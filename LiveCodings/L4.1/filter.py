
from skimage import io
import numpy as np
import matplotlib.pyplot as plt


image = io.imread('./pim4files/freckles.jpg')/255.
plt.imshow(image)
plt.gcf().set_size_inches((10,6))
sz = 21
half_sz=int(np.floor(sz/2.)+1)
xx,yy = np.meshgrid(np.linspace(-4,4,sz),np.linspace(-4,4,sz))
mask = np.exp(-0.5*(xx*xx+yy*yy))
mask_r = mask.ravel()
beta=180.

def my_filter(x):
    return np.median(x)

from scipy import ndimage
def rgb_generic_filter(image, func, size):
    r=ndimage.generic_filter(image[:,:,0],func,size)
    g=ndimage.generic_filter(image[:,:,1],func,size)
    b=ndimage.generic_filter(image[:,:,2],func,size)
    return np.dstack((r,g,b))

def bilateral_filter(x):
    pc = int(x[np.floor(len(x)/2.)+1])
    mask_i=np.exp(-0.5*(x-pc)*(x-pc)*beta)
    mask=mask_r*mask_i.ravel()
    mask=mask/np.sum(mask)
    return np.sum(mask*x)

def bilateral_filter2(x):
    print (x.shape)
    #pc = int(x[int(np.floor(len(x)/2))+1])
    #pc = int(x[np.floor(len(x)/2.)+1])
    #mask_i=np.where(np.abs(x-pc)<0.2,1.,0.)
    #mask=mask_r*mask_i.ravel()
    #mask=mask/np.sum(mask)
    #return np.sum(mask*x)
    return 1


import math
def distance(x, y, i, j):
    return np.sqrt((x-i)**2 + (y-j)**2)


def gaussian(x, sigma):
    return (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x ** 2) / (2 * sigma ** 2))


def apply_bilateral_filter(source, filtered_image, x, y, diameter, sigma_i, sigma_s):
    hl = diameter/2
    i_filtered = 0
    Wp = 0
    i = 0
    while i < diameter:
        j = 0
        while j < diameter:
            neighbour_x = x - (hl - i)
            neighbour_y = y - (hl - j)
            if neighbour_x >= len(source):
                neighbour_x -= len(source)
            if neighbour_y >= len(source[0]):
                neighbour_y -= len(source[0])
            n_x=int(np.floor(neighbour_x))
            n_y=int(np.floor(neighbour_y))
            gi = gaussian(source[n_x][n_y] - source[x][y], sigma_i)
            gs = gaussian(distance(n_x, n_y, x, y), sigma_s)
            w = gi * gs
            i_filtered += source[n_x][n_y] * w
            Wp += w
            j += 1
        i += 1
    i_filtered = i_filtered / Wp
    filtered_image[x][y] = int(round(i_filtered))


def bilateral_filter_own(source, filter_diameter, sigma_i, sigma_s):
    filtered_image = np.zeros(source.shape)

    i = 0
    while i < len(source):
        j = 0
        while j < len(source[0]):
            apply_bilateral_filter(source, filtered_image, i, j, filter_diameter, sigma_i, sigma_s)
            j += 1
        i += 1
        print(i, end=' ')
    return filtered_image

image2=image[::3,::3,:]
res_max = np.zeros(image2.shape, 'uint8')
res_max[..., 0] = bilateral_filter_own(image2[:,:,0]*256, sz, 12.0, 16.0)
res_max[..., 1]= bilateral_filter_own(image2[:,:,1]*256, sz, 12.0, 16.0)
res_max[..., 2] = bilateral_filter_own(image2[:,:,2]*256, sz, 12.0, 16.0)

plt.imshow(res_max,cmap='jet')
#res_max = rgb_generic_filter(image,bilateral_filter2,[sz,sz])