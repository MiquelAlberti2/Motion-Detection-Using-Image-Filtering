import numpy as np
import imageio.v3 as iio # to read and write images
import matplotlib.pyplot as plt
import os

def est_noise(images):
    #estimate the noise using the EST_NOISE procedure

    n=len(images)

    for i in range(n):
        images[i] = np.array(images[i])

    dim=images[0].shape[0] #suppose its square

    # Use the procedure EST_NOISE to estimate the noise
    estim_mu = np.zeros((dim,dim))
    estim_sigma = np.zeros((dim,dim))

    for i in range(dim):
        for j in range(dim):

            for img in images:
                estim_mu[i,j] += img[i,j]

            estim_mu[i,j] /= n

            for img in images:
                estim_sigma[i,j] += (estim_mu[i,j] - img[i,j])**2

            estim_sigma[i,j] = (estim_sigma[i,j]/(n-1))**(1/2)

    return np.mean(estim_mu), np.mean(estim_sigma)


def threshold(value, th):
    if value > th:  # Change threshold value here
        return 1
    else:
        return 0


def rgb_to_gray(rgb):

    # compute a weighted average of RGB colors to obtain a greyscale value
    # weights correspond to the luminosity of each color channel
    # we also normalize the image
    return (1/255)*np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


# Takes an image of mxn size with only greyscale values, and creating an image of the same size but with r, g, and b in
# each pixel, where r/g/b all equal the grey value from the original image
def padGreyscaleImageToRGBImage(greyImage):
    rgbImage = []
    for i in range(len(greyImage)):
        rgbImage.append([])
        for j in range(len(greyImage[0])):
            rgbImage[i].append([round(greyImage[i][j]), round(greyImage[i][j]), round(greyImage[i][j])])
    output = np.array(rgbImage)
    return output


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

def size_gaussian_mask(std):

    size=5*std

    # the number should be integer
    if not size.is_integer():
        size=int(5*std) + 1
        
    # we want an odd size
    if size%2==0:
        size+=1

    size = int(size)

    return size, int(size/2)
    
def compute_Gauss_filter(std):
    
    size, half_s = size_gaussian_mask(std)

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

    size, half_s = size_gaussian_mask(std)
    
    dg = np.zeros(size)
    for i in range(-half_s, half_s + 1):
        dg[i + half_s] = -(i/std**2)*np.exp(-(i**2)/(2*std**2))

    return dg

def compute_temporal_derivatives(all_images, filter):
    l = len(filter)

    # suppose all images have the same size
    nrow=all_images[0].shape[0]
    ncol=all_images[0].shape[1]

    # The function takes as input the array of smoothed images
    output = []
    
    n_output_frames = len(all_images) - l + 1
    for count in range(0, n_output_frames):
        print(f"Computing derivatives...({count+1}/{n_output_frames})")
        
        compared_images=[]
        for i in range(l):
            compared_images.append(all_images[count+i])

        mask = []  # new image
        for row in range(nrow):
            mask.append([])
            for col in range(ncol):
                value = 0
                for location in range(l):
                    value += filter[location]*compared_images[location][row][col]
                mask[row].append(value)
        output.append(mask)
    return output


def applyMasksToOriginalFrames(masks, frames, th):
    # suppose all images have the same size
    nrow=frames[0].shape[0]
    ncol=frames[0].shape[1]

    output = []
    offset = int((len(frames)-len(masks))/2)

    for imageNum in range(len(masks)):  # For each image
        print(f"Applying masks to original frames...({imageNum+1}/{len(masks)})")
        maskedImage = []
        for row in range(nrow):
            maskedImage.append([])
            for col in range(ncol):
                maskedImage[row].append(np.array(frames[imageNum+offset][row][col])*threshold(masks[imageNum][row][col], th))
        output.append(np.array(maskedImage))
    return output


#####################
# Read all the images in the directory
#####################

directory = 'RedChair'
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

numImages = len(greyImages)
counter = 1

match smoothing:
    case "no":
        smoothedImages = greyImages
    case "gaussian":
        std = float(input("Which standard deviation (sigma)?").lower())
        for img in greyImages:
            print(f'Smoothing...({counter}/{numImages})')
            smoothedImages.append((apply_Gauss_filter(img, std)))
            counter += 1
    case "box3":
        for img in greyImages:
            print(f'Smoothing...({counter}/{numImages})')
            smoothedImages.append((apply_BOX_filter(img, 3)))
            counter += 1
    case "box5":
        for img in greyImages:
            print(f'Smoothing...({counter}/{numImages})')
            smoothedImages.append((apply_BOX_filter(img, 5)))
            counter += 1
    case _:
        smoothedImages = greyImages

# check smoothing results:

plt.gray()
plt.imshow(smoothedImages[0])
plt.show()

# save an example of the smoothing (or not smoothing)
iio.imwrite(uri="smoothed.png", image=padGreyscaleImageToRGBImage(smoothedImages[3]).astype(np.uint8))

#####################
# Compute temporal derivatives
#####################

temporal = input("Would you like to use a simple or gaussian filter? (simple/gaussian)").lower()

match temporal:
    case "simple":
        filter = [-0.5, 0, 0.5]
    case "gaussian":
        std = float(input("Which standard deviation (sigma)?").lower())
        filter = compute_derivative_Gauss(std)
    case _:
        filter = [-0.5, 0, 0.5]

motionMasks = compute_temporal_derivatives(smoothedImages, filter)

# For both directories, there is no motion in the first 22 frames
# Therefore, the resulting motion masks should be absolutely black
# And we can use them to estimate the noise

mu, sigma = est_noise(motionMasks[:18]) #we cannot choose the 22 frames because 
                                        #we discard the first frames due to the size of the filter

th = mu + 3*sigma 

print('-----------------------------------------------------')
print("Chosen threshold: "+str(th))
print('-----------------------------------------------------')

maskedImages = applyMasksToOriginalFrames(motionMasks, original_images, th)

#####################
# Output final images
#####################
for image in range(len(maskedImages)):
    print(f"Saving images in disk...({image+1}/{len(maskedImages)})")
    filePath = "output/name" + str(image) + ".png"
    iio.imwrite(uri=filePath, image=maskedImages[image])