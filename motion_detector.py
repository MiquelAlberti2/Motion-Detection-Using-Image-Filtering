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

    # compute a weighted average of RGB colors to obtain a greyscale value
    # weights correspond to the luminosity of each color channel
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def apply_BOX_filter(img):

    # create a kernel for a 3x3 BOX filter
    kernel = np.zeros((3,3))
    kernel.fill(1./9)

    nrow=img.shape[0]
    ncol=img.shape[1]

    filt_img = np.zeros((nrow-2,ncol-2)) #ignore borders

    for i in range(nrow):
        for j in range(ncol):
            if i>0 and i<nrow-1 and j>0 and j<ncol-1:
                filt_img[i-1,j-1] = (kernel*img[i-1:i+2, j-1:j+2]).sum()


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

smoothed_BOX_images = []

for img in original_images:
    smoothed_BOX_images.append(apply_BOX_filter(img))

smoothed_Gauss_images = []

for img in original_images:
    smoothed_Gauss_images.append(apply_Gauss_filter(img))

# check smoothing results:
# plt.imshow(smoothed_BOX_images[0])
# plt.show()
# plt.imshow(smoothed_Gauss_images[0])
# plt.show()


#####################
# Compute temporal derivatives
#####################

#compute_temporal_derivatives(smoothed_images) # TODO think of what should the function return



#iio.imwrite(uri="name1.png", image=smoothed_BOX_images[0])
#iio.imwrite(uri="name2.png", image=smoothed_Gauss_images[0])

