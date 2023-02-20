import numpy as np
import imageio.v3 as iio # to read and write images
import matplotlib.pyplot as plt
import os


def threshold(value):
    if value > .3:  # Change threshold value here
        return 1
    else:
        return 0


def rgb_to_gray(rgb):

    # compute a weighted average of RGB colors to obtain a greyscale value
    # weights correspond to the luminosity of each color channel
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def padGreyscaleImageToRGBImage(greyImage):
    rgbImage = []
    for i in range(len(greyImage)):
        rgbImage.append([])
        for j in range(len(greyImage[0])):
            rgbImage[i].append([greyImage[i][j], greyImage[i][j], greyImage[i][j]])

def apply_BOX_filter(img, dim):

    # create a kernel for a "dim" x "dim" BOX filter
    kernel = np.zeros((dim,dim))
    kernel.fill(1./(dim*dim))

    nrow=img.shape[0]
    ncol=img.shape[1]

    filt_img = np.zeros_like(img)

    pad_size = int(dim/2)
    pad_image = np.pad(img, pad_size, mode='constant')

    for i in range(pad_size, nrow + pad_size):
        for j in range(pad_size, ncol + pad_size):
                filt_img[i-pad_size,j-pad_size] = (kernel*pad_image[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1]).sum()

    return filt_img

def compute_Gauss_filter(std):
    size=5*std

    # the number should be integer
    if not size.is_integer():
        size=int(5*std) + 1
        
    # we want an odd size
    if size%2==0:
        size+=1

    size = int(size)

    mask = np.zeros((size,size))

    half_s = int(size/2)

    for i in range(-half_s,half_s+1):
        for j in range(-half_s,half_s+1):
            mask[i+half_s,j+half_s] = np.exp((-i*i - j*j)/(2*std*std))

    # Factor to normalize the mask
    k = np.sum(mask)

    # to get the separable filter, we just take the first row
    d_mask = (1/(k**(1/2)))*mask[0]

    return d_mask, half_s



def apply_Gauss_filter(img,std):

    #create a kernel for a separated gaussian filter
    kernel, padding_size = compute_Gauss_filter(std)

    nrow=img.shape[0]
    ncol=img.shape[1]

    # create a "horizontal" 0 padding
    padded_image = np.pad(img, ((0, 0), (padding_size, padding_size)), mode='constant')

    
    vertical_filt_img = np.zeros_like(img)

    # apply the horizontal filter
    for i in range(nrow):
        for j in range(padding_size, ncol + padding_size):
                vertical_filt_img[i,j-padding_size] = (kernel*padded_image[i, j-padding_size:j+padding_size+1]).sum()


    # create a "vertical" 0 padding
    vertical_filt_img = np.pad(vertical_filt_img, ((padding_size, padding_size), (0, 0)), mode='constant')
    
    filt_img = np.zeros_like(img)

    # apply the vertical filter
    for i in range(padding_size, nrow + padding_size):
        for j in range(ncol):
                filt_img[i-padding_size,j] = (kernel*vertical_filt_img[i-padding_size:i+padding_size+1, j]).sum()

    return filt_img

def compute_derivative_Gauss(std):
    # TODO
    pass

def compute_temporal_derivatives(all_images, filter, modifier=1.0):
    # The function takes as input the array of smoothed images
    output = []
    for count in range(0, len(all_images) - 2):
        trio_images = [all_images[count], all_images[count+1], all_images[count+2]]

        mask = []  # new image of size 320x240
        for row in range(240):
            mask.append([])
            for col in range(320):
                value = 0
                for location in range(3):
                    value += filter[location]*trio_images[location][row][col]
                mask[row].append(threshold(value*modifier))
        output.append(mask)
    return output


def applyMasksToOriginalFrames(masks, frames):
    output = []
    for imageNum in range(len(masks)):  # For each image
        maskedImage = []
        for row in range(240):
            maskedImage.append([])
            for col in range(320):
                maskedImage[row].append(np.array(frames[imageNum][row][col])*masks[imageNum][row][col])
        output.append(np.array(maskedImage))
    return output


#####################
# Read all the images in the directory
#####################

directory='RedChair'
original_images = []

for filename in os.listdir(directory):
    print("Reading image "+filename)
    original_images.append(iio.imread(uri=directory+'/'+filename))

#####################
# Convert images to black and white
#####################

greyImages = []

for i in range(len(original_images)):
    print("Greyscaling image "+str(i))
    greyImages.append(rgb_to_gray(original_images[i]))

#####################
# Apply a smoothing filter to all images
#####################

smoothing = input("Would you like to apply a smoothing filter? (no/gaussian/box3/box5)").lower()

smoothedImages = []

l = len(greyImages)
counter = 1

match smoothing:
    case "no":
        smoothedImages = greyImages
    case "gaussian":
        std = float(input("Which standard deviation (sigma)?").lower())
        for img in greyImages:
            print(f'Smoothing...({counter}/{l})')
            smoothedImages.append((apply_Gauss_filter(img, std)))
            counter += 1
    case "box3":
        for img in greyImages:
            print(f'Smoothing...({counter}/{l})')
            smoothedImages.append((apply_BOX_filter(img, 3)))
            counter += 1
    case "box5":
        for img in greyImages:
            print(f'Smoothing...({counter}/{l})')
            smoothedImages.append((apply_BOX_filter(img, 5)))
            counter += 1
    case _:
        smoothedImages = greyImages

# check smoothing results:

#plt.imshow(smoothedImages[0])
#plt.show()
# plt.imshow(smoothed_BOX_images_5_dim[0])
# plt.show()
# plt.imshow(smoothedImages[0])
# plt.show()


#####################
# Compute temporal derivatives
#####################

temporal = input("Would you like to use a simple or gaussian filter? (simple/gaussian)").lower()

match temporal:
    case "simple":
        filter = [-1, 0, 1]
    case "gaussian":
        std = float(input("Which standard deviation (sigma)?").lower())
        filter = compute_derivative_Gauss(std)

motionMasks = compute_temporal_derivatives(smoothedImages, filter, .5)
maskedImages = applyMasksToOriginalFrames(motionMasks, original_images)

#####################
# Output final images
#####################
for image in range(len(maskedImages)):
    filePath = "output/name" + str(image) + ".png"
    iio.imwrite(uri=filePath, image=maskedImages[image])

