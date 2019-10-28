from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
import matplotlib.image as mpimg
import os
import itertools

#displacement variables
dx_from = -10
dx_to = 10
dy_from = -10
dy_to = 10

#images
images = ['00029u.png']#, '00087u.png', '00106v.jpg', '00128u.png', '00458u.png', '00737u.png', '00757v.jpg', '00822u.png', '00888v.jpg', '00889v.jpg', '00892u.png', '00907v.jpg', '00911v.jpg', '01031v.jpg', '01043u.png', '01047u.png', '01657v.jpg', '01880v.jpg']

def NCC (image1, image2):
    # non-zero mean NCC
    
    i1 = image1.flatten()
    i2 = image2.flatten()
    
    norm1 = np.linalg.norm(i1)
    norm2 = np.linalg.norm(i2)
    
    ncc = np.dot(i1, i2)/(norm1*norm2)
    return ncc

def SSD (i1, i2):
    # Using SSD to match the pictures
        
    ssd = np.sum((np.absolute(i1 - i2)) ** 2)
    return ssd

def displace(d_img1, d_img2, m, n):
    #displaces the image height by m, width by n

    if m < 0:
        d_img1 = d_img1[:d_img1.shape[0]+m, :]
        d_img2 = d_img2[np.absolute(m):, :]
    elif m > 0:
        d_img1 = d_img1[m:, :]
        d_img2 = d_img2[:d_img2.shape[0]-m, :]
    if n < 0:
        d_img1 = d_img1[:, np.absolute(n):]
        d_img2 = d_img2[:, :d_img2.shape[1]+n]
    elif n > 0:
        d_img1 = d_img1[:, :d_img1.shape[1]-n]
        d_img2 = d_img2[:, n:]
        
    return (d_img1, d_img2)

def best_alignment(image1, image2):
    displacement = [0, 0]
    result = NCC(image1, image2)
    #displacing height
    for m in range(dy_from, dy_to+1, 1):
        #displacing width
        for n in range(dx_from, dx_to+1, 1):
            # image2 can't cover all of image1 so compute
            # only comparing overlapping parts
            d_img1 = image1
            d_img2 = image2
            
            d_img1, d_img2 = displace(d_img1, d_img2, m, n)
            
            #result2 = NCC(d_img1, d_img2)
            result2 = SSD(d_img1, d_img2)

            if result2 > result:
                result = result2
                displacement = [m, n]
            
        
    return displacement
                
if __name__ == '__main__':
    os.chdir(os.getcwd())
    
    ########################
    #Run the following for grayscale images to show up correctly and for images
    #to display immediately
    gray()
    ########################
    
    # part 1
    for n in range(0, len(images)):    
        #Read in an image
        i = imread(images[n])
        
        i = i.astype(float)
        
        #Crop out borders (width) by approximately 10%
        borders = i.shape[1] * 0.10
        i = i[:, (borders/2):(i.shape[1]-borders/2)]
        
        #Crop out borders (height) by approximately 4%
        borders = i.shape[0] * 0.04
        i = i[(borders/2):(i.shape[0]-borders/2), :]
        
        #Distribute the 3 different photos to their respective layers
        firstend = (i.shape[0]/3)
        secondend= 2*firstend
        thirdend = 3*firstend
        
        b = i[:firstend, :]
        g = i[firstend:secondend, :]
        r = i[secondend:thirdend, :]
        
        # compare green to blue
        align = best_alignment(b, g)
        print(align)
        newb, newg = displace(b, g, align[0], align[1])
        
        # compare green to red
        align = best_alignment(g, r)
        print(align)
        newg, newr = displace(g, r, align[0], align[1])
        
        #Create a new array with the same size/dimensions as i
        height = min(newr.shape[0], newg.shape[0], newb.shape[0])
        width = min(newr.shape[1], newg.shape[1], newb.shape[1])
        newr = newb[:height, :width]
        newg = newg[:height, :width]
        newb = newb[:height, :width]
        channel = 3
    
        # convert numbers to be in a smaller range to display properly
        newr = newr/float(max(newr.flatten()))
        newg = newg/float(max(newg.flatten()))
        newb = newb/float(max(newb.flatten()))
    
        z = zeros((height, width, channel)).astype(float)
        z[:,:, 0] = newr
        z[:,:, 1] = newg
        z[:,:, 2] = newb
        
        name = 'pic2'+str(n)+'.jpg'
        imsave(name, z)
    
    #img resize version (part2)
    '''    
    for n in range(0, len(images)):    
        #Read in an image
        i = imread(images[n])
       
        i = i.astype(float)
        
        #Crop out borders (width) by approximately 10%
        borders = i.shape[1] * 0.10
        i = i[:, (borders/2):(i.shape[1]-borders/2)]
        
        #Crop out borders (height) by approximately 4%
        borders = i.shape[0] * 0.04
        i = i[(borders/2):(i.shape[0]-borders/2), :]
        
        #Distribute the 3 different photos to their respective layers
        firstend = (i.shape[0]/3)
        secondend= 2*firstend
        thirdend = 3*firstend
        
        b = i[:firstend, :]
        g = i[firstend:secondend, :]
        r = i[secondend:thirdend, :]
        
        #resize by scaling 75% so that it's small enough
        b2 = imresize(b, 0.75)
        g2 = imresize(g, 0.75)
        r2 = imresize(r, 0.75)
        
        # compare green to blue
        align1 = best_alignment(b2, g2)
        
        # compare green to red
        align2 = best_alignment(g2, r2)
        
        #get general displacements 
        dx_from = min(align1[0], align2[0])
        dx_to = max(align1[0], align2[0])
        dy_from = min(align1[1], align2[1])
        dy_to = max(align1[0], align2[0])
        
        newb, newr = displace(g, r, align[0], align[1])
        newg, newr = displace(g, r, align[0], align[1])
        
        #Create a new array with the same size/dimensions as i
        height = min(newr.shape[0], newg.shape[0], newb.shape[0])
        width = min(newr.shape[1], newg.shape[1], newb.shape[1])
        newr = newb[:height, :width]
        newg = newg[:height, :width]
        newb = newb[:height, :width]
        channel = 3
    
        # convert numbers to be in a smaller range to display properly
        newr = newr/float(max(newr.flatten()))
        newg = newg/float(max(newg.flatten()))
        newb = newb/float(max(newb.flatten()))
    
        z = zeros((height, width, channel)).astype(float)
        z[:,:, 0] = newr
        z[:,:, 1] = newg
        z[:,:, 2] = newb
        
        name = 'pic2'+str(n)+'_resized.jpg'
        imsave(name, z)
    '''