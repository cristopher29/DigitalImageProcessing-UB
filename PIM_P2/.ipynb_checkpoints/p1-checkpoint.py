from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
import copy

def ncc(p1, p2):
    '''
    Take two matrix at same size,
    return the Normalized Cross-Correlation(NCC) between them.
    '''
    up = sum((p1 - p1.mean())*(p2 - p2.mean()))
    down_p1 = sqrt(sum((p1 - p1.mean())*(p1 - p1.mean())))
    down_p2 = sqrt(sum((p2 - p2.mean())*(p2 - p2.mean())))
    output = up / down_p1 / down_p2
    return output 

def ssd(p1, p2):
    '''
    Take two matrix at same size,
    return the Sum of Squared Differences(SSD) between them.
    '''
    p1 = p1.astype(float)
    p2 = p2.astype(float)
    output = (p1/255 - p2/255)**2
    return output.sum()
    
def max_ncc(i1, i2, p1, p2, edge_length):
    '''
    Takes two images i1, i2; two square areas p1, p2 in i1, i2;
    take length of any p.
    From -10 to 10 pixels along both the x and the y directions, move p2 and
    culculate the ncc between moved p2 and p1.
    Return the x,y position with the largest ncc value. 
    '''
    n = ncc(p1, p2)
    l = [0, 0]
    for ci in range(-10, 11, 1):
        for ri in range(-10, 11, 1):
            p = i2[(50+ci):(edge_length-50+ci), (50+ri):(edge_length-50+ri)]
            new_n = ncc(p1, p)
            if n < new_n:
                n = new_n
                l = [ci, ri]
    return l

def min_ssd(i1, i2, p1, p2, edge_length):
    '''
    Takes two images i1, i2; two square areas p1, p2 in i1, i2;
    take length of any p.
    From -10 to 10 pixels along both the x and the y directions, move p2 and
    culculate the ssd between moved p2 and p1.
    Return the x,y position with the minimum ssd value. 
    '''
    s = ssd(p1, p2)
    l = [0, 0]
    for ci in range(-10, 11, 1):
        for ri in range(-10, 11, 1):
            p = i2[(50+ci):(edge_length-50+ci), (50+ri):(edge_length-50+ri)]
            new_s = ssd(p1, p)
            if s > new_s:
                s = new_s
                l = [ci, ri]
    return l

def conduct(i):
    '''
    Takes any gray image, change the value of every pixel under 255 and return it
    '''
    max_in_i = max(i.flatten())
    index = 0
    while 2**index-1 < max_in_i:
        index += 1
    output = i/(2**index-1)*255
    return output.astype(int)

def PGmagic(i, is_ncc):
    '''
    Takes an gray image of three inverted negatives;
    Takes an value to determine which kind of matching used;
    Return a color image by combining the three colour channels
    '''
    #if i should be conduct to values under 255 so that it can represent the
    #value of a color channel correctly.
    i = conduct(i)
    #Record the initial size information of input image.
    is_large = 0
    height = int(len(i)/3)
    b_y = [0, height - 1]
    g_y = [height, 2*height-1]
    r_y = [2*height, 3*height - 1]
    b_x = [0, len(i[0])]
    g_x = [0, len(i[0])]
    r_x = [0, len(i[0])]
    
    #Check if input image is large. I it is large, then resize it.
    if max(shape(i)) > 1024:
        is_large = 1
        self_ratio = (len(i) * 1.0) / (len(i[0]) * 1.0)
        ratio = 1024.0 / (len(i) * 1.0)
        i_shape = [1024, int(len(i[1]) * ratio)]
        if i_shape[1] == 0:
            i_shape[1] = 1
        buffer = imresize(i, i_shape)
        real_i = i
        i = buffer
        height = int(len(i)/3)

    #obataining the initial guess of three inverted negatives by trisecting i
    b_guess = i[: height-1]
    g_guess = i[height : 2*height-1]
    r_guess = i[2*height : 3*height -1]

    #pick a large square area of each inverted negatives with the same size
    #This would be used as sample area for matching.
    edge_length = min(shape(b_guess))
    pb = b_guess[50:edge_length-50, 50:edge_length-50]
    pg = g_guess[50:edge_length-50, 50:edge_length-50]
    pr = r_guess[50:edge_length-50, 50:edge_length-50]
    
    #matching green and red inverted negatives to the blue inverted negative
    #find the best matched position by culculating ncc or ssd.
    if is_ncc == 1:
        lg = max_ncc(b_guess, g_guess, pb, pg, edge_length)
        lr = max_ncc(b_guess, r_guess, pb, pr, edge_length)
    else:
        lg = min_ssd(b_guess, g_guess, pb, pg, edge_length)
        lr = min_ssd(b_guess, r_guess, pb, pr, edge_length)

    #If i is large, then extend matched position by dividing resize ratio.
    if is_large == 1:
        lg = [int(round(lg[0]/ratio)), int(round(lg[1]/ratio))]
        lr = [int(round(lr[0]/ratio)), int(round(lr[1]/ratio))]
        i = real_i
        height = int(len(i)/3)

    #By culculated mathced position, culculate correct cut posistions for
    #three inverted negatives
    b_y = [b_y[0], b_y[1]]
    g_y = [g_y[0]+lg[0], g_y[1]+lg[0]]
    r_y = [r_y[0]+lr[0], r_y[1]+lr[0]]
    b_x = [b_x[0], b_x[1]]
    g_x = [g_x[0]+lg[1], g_x[1]+lg[1]]
    r_x = [r_x[0]+lr[1], r_x[1]+lr[1]]

    #for the red inverted negative, if matched possition goes over the bottom
    #of i, fill that part with 0
    if (r_y[1] >= len(i)):
        i_expend = zeros([11, len(i[1])], dtype=uint8)
        i = vstack((i, i_expend))
    i1 = i[b_y[0]:b_y[1]]
    i2 = i[g_y[0]:g_y[1]]
    i3 = i[r_y[0]:r_y[1]]

    #move each inverted negatives horizontally with the matched possition.
    #for part out of the scale of i, fill that part with 0
    if lg[1] < 0:
        g_expend = zeros((height-1, -lg[1]),dtype=uint8)
        i2 = np.append(i2, g_expend, axis=1)[:,:lg[1]]
    else:
        g_expend = zeros((height-1, lg[1]), dtype=uint8)
        i2 = np.append(g_expend, i2, axis=1)[:,lg[1]:]
    if lr[1] < 0:
        r_expend = zeros((height-1, -lr[1]),dtype=uint8)
        i3 = np.append(i3, r_expend, axis=1)[:,:lr[1]]
    else:
        r_expend = zeros((height-1, lr[1]), dtype=uint8)
        i3 = np.append(r_expend, i3, axis=1)[:,lr[1]:]

    #combine three inverted negatives to RGB channel and return a colored image
    output = zeros((len(i1), len(i1[0]), 3), uint8)
    output[:, :, 0] = i3
    output[:, :, 1] = i2
    output[:, :, 2] = i1
    return output
    
'''
i1 = imread('/h/u12/g4/00/g3yangyl/Downloads/320/a1/images/00106v.jpg')
i2 = imread('/h/u12/g4/00/g3yangyl/Downloads/320/a1/images/00757v.jpg')
i3 = imread('/h/u12/g4/00/g3yangyl/Downloads/320/a1/images/00888v.jpg')
i4 = imread('/h/u12/g4/00/g3yangyl/Downloads/320/a1/images/00889v.jpg')
i5 = imread('/h/u12/g4/00/g3yangyl/Downloads/320/a1/images/00907v.jpg')
i6 = imread('/h/u12/g4/00/g3yangyl/Downloads/320/a1/images/00911v.jpg')
i7 = imread('/h/u12/g4/00/g3yangyl/Downloads/320/a1/images/01031v.jpg')
i8 = imread('/h/u12/g4/00/g3yangyl/Downloads/320/a1/images/01657v.jpg')
i9 = imread('/h/u12/g4/00/g3yangyl/Downloads/320/a1/images/01880v.jpg')

j1 = imread('/h/u12/g4/00/g3yangyl/Downloads/320/a1/images/00029u.png')
j2 = imread('/h/u12/g4/00/g3yangyl/Downloads/320/a1/images/00128u.png')
j3 = imread('/h/u12/g4/00/g3yangyl/Downloads/320/a1/images/00458u.png')

o1n = PGmagic(j1, 1)
o2n = PGmagic(j2, 1)
o3n = PGmagic(j3, 1)
o1s = PGmagic(j1, 0)
o2s = PGmagic(j2, 0)
o3s = PGmagic(j3, 0)

imsave('/h/u12/g4/00/g3yangyl/Downloads/320/a1/00029uncc.jpg', o1n)
imsave('/h/u12/g4/00/g3yangyl/Downloads/320/a1/00128uncc.jpg', o2n)
imsave('/h/u12/g4/00/g3yangyl/Downloads/320/a1/00458uncc.jpg', o3n)
imsave('/h/u12/g4/00/g3yangyl/Downloads/320/a1/00029ussd.jpg', o1s)
imsave('/h/u12/g4/00/g3yangyl/Downloads/320/a1/00128ussd.jpg', o2s)
imsave('/h/u12/g4/00/g3yangyl/Downloads/320/a1/00458ussd.jpg', o3s)
'''
