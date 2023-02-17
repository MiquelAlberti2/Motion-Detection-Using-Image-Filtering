import numpy as np
import imageio.v3 as iio # to read and write images
import matplotlib.pyplot as plt


def apply_filter(kernel, img):

    nrow=img.shape[0] #suppose its square
    ncol=img.shape[1] #suppose its square

    filt_img = np.zeros((nrow-2,ncol-2)) #ignore borders

    for i in range(nrow):
        for j in range(ncol):
            if i>0 and i<nrow-1 and j>0 and j<ncol-1:
                filt_img[i-1,j-1] = (kernel*img[i-1:i+2, j-1:j+2]).sum()


    return filt_img







directory='RedChair'

image = iio.imread(uri="data/eight.tif")
plt.imshow(image)


iio.imwrite(uri="ws.bmp", image=image)
