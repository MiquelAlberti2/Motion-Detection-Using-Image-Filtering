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


def apply_Gauss_filter(img):

    #create a kernel for a separated gaussian filter
    kernel=np.array([1,4,8,10,8,4,1])
    kernel=(1/36)*kernel

    nrow=img.shape[0]
    ncol=img.shape[1]

    # add a padding to the orignal image
    padding_size = 3

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


def compute_temporal_derivatives(all_images, filter, modifier=1):
    # The function takes as input the array of smoothed images

    for count in range(0, len(all_images) - 1):
        trio_images = [all_images[count], all_images[count+1], all_images[count+2]]

        mask = []  # new image of size 320x240
        for row in range(320):
            mask.append([])
            for col in range(240):
                value = 0
                for location in range(3):
                    value += filter[location]*trio_images[location][row][col]
                mask[row].append(threshold(value*modifier))
        maskedImage = []  # new image of size 320x240
        for row in range(320):
            maskedImage.append([])
            for col in range(240):
                maskedImage[row].append(trio_images[1][row][col]*mask[row][col])


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

for i in range(len(original_images)):
    print("Greyscaling image "+str(i))
    original_images[i] = rgb_to_gray(original_images[i])

#####################
# Apply a smoothing filter to all images
#####################

smoothing = input("Would you like to apply a smoothing filter? (no/gaussian/box3/box5)").lower()

smoothedImages = []

match smoothing:
    case "no":
        smoothedImages = original_images
    case "gaussian":
        for img in original_images:
            smoothedImages.append((apply_Gauss_filter(img)))
    case "box3":
        for img in original_images:
            smoothedImages.append((apply_BOX_filter(img, 3)))
    case "box5":
        for img in original_images:
            smoothedImages.append((apply_BOX_filter(img, 5)))

# check smoothing results:

# plt.imshow(smoothed_BOX_images_3_dim[0])
# plt.show()
# plt.imshow(smoothed_BOX_images_5_dim[0])
# plt.show()
# plt.imshow(apply_Gauss_filter[0])
# plt.show()


#####################
# Compute temporal derivatives
#####################

#compute_temporal_derivatives(smoothed_images) # TODO think of what should the function return


# This doesn't work, I'm trying to figure out the error
iio.imwrite(uri="name1.png", image=smoothedImages[0])
iio.imwrite(uri="name2.png", image=smoothedImages[1])

