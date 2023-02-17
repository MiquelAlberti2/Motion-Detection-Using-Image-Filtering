import numpy as np
import imageio.v3 as iio # to read and write images
import matplotlib.pyplot as plt
import os


def threshold(value):
    if value > .3: # Change threshold value here
        return 1
    else:
        return 0

def rgb_to_gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def apply_filter(kernel, img):

    nrow=img.shape[0]
    ncol=img.shape[1]

    filt_img = np.zeros((nrow-2,ncol-2)) #ignore borders

    for i in range(nrow):
        for j in range(ncol):
            if i>0 and i<nrow-1 and j>0 and j<ncol-1:
                filt_img[i-1,j-1] = (kernel*img[i-1:i+2, j-1:j+2]).sum()


    return filt_img

def apply_BOX_filter(img):

    # create a kernel for a 3x3 BOX filter
    kernel = np.zeros((3,3))
    kernel.fill(1./9)

    return apply_filter(kernel,img)


def compute_temporal_derivatives(all_images):
    # The function takes as input the array of smoothed images

    for i in range(0,len(all_images),3):
        trio_images = [all_images[i], all_images[i+1], all_images[i+2]]

        """ There's no need for this now
        images = [] # array of 3 images, each 320x240
        for i in range(1,1069):
            with open('Office/image01_'+str(i).zfill(4)+'.jpg', 'rb') as f:
                images.append(f.read())
            with open('Office/image01_'+str(i+1).zfill(4)+'.jpg', 'rb') as f:
                images.append(f.read())
            with open('Office/image01_'+str(i+2).zfill(4)+'.jpg', 'rb') as f:
                images.append(f.read())
        """
        mask = [] # new image of size 320x240
        for row in range(320):
            mask.append([])
            for col in range(240):
                mask[row].append(threshold(.5*(-1*trio_images[0][row][col] + 1*trio_images[2][row][col])))
        maskedImage = [] # new image of size 320x240
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
    original_images.append(iio.imread(uri=directory+'/'+filename))

#####################
# Convert images to black and white
#####################

for i in range(len(original_images)):
    original_images[i] = rgb_to_gray(original_images[i])

#####################
# Apply a smoothing filter to all images
#####################

smoothed_images = []

for img in original_images:
    smoothed_images.append(apply_BOX_filter(img))

#####################
# Compute temporal derivatives
#####################

compute_temporal_derivatives(smoothed_images) # TODO think of what should the function return



# plt.imshow(image)
# iio.imwrite(uri="ws.bmp", image=image)
