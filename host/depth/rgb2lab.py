#!/usr/bin/env python

from scipy import *
from scipy.ndimage import *

import Image

def rgb2xyz(im):
    ''' Convert RGB image to XYZ image. Algorithm from www.easyrgb.com'''
    # Normalize the image first.
    imtemp = im.copy()/255.0
    x1, y1, z1 = where(imtemp > 0.0405)
    x2, y2, z2 = where(imtemp <= 0.0405)

    imtemp[x1, y1, z1] = pow((imtemp[x1, y1, z1] + 0.055)/1.055, 2.4)
    imtemp[x2, y2, z2] = imtemp[x2, y2, z2]/12.92
    
    imtemp *= 100
    
    imxyz = zeros_like(imtemp)
    imxyz[:,:,0] = imtemp[:,:,0]*0.4124 + (
                   imtemp[:,:,1]*0.3576) + (
                   imtemp[:,:,2]*0.1805)
    imxyz[:,:,1] = imtemp[:,:,0]*0.2126 + (
                   imtemp[:,:,1]*0.7152) + (
                   imtemp[:,:,2]*0.0722)
    imxyz[:,:,2] = imtemp[:,:,0]*0.0193 + (
                   imtemp[:,:,1]*0.1192) + (
                   imtemp[:,:,2]*0.9505)
    return imxyz

def xyz2lab(im):
    ''' Convert XYZ image to CIE-L*ab format'''
    imxyz = im.copy()
    imxyz[:,:,0] /= 95.047
    imxyz[:,:,1] /= 100.00
    imxyz[:,:,2] /= 108.883

    x1, y1, z1 = where(imxyz > 0.008856)
    x2, y2, z2 = where(imxyz <= 0.008856)
    imxyz[x1, y1, z1] = pow(imxyz[x1, y1, z1], 1.0/3.0)
    imxyz[x2, y2, z2] = (7.787*imxyz[x2, y2, z2]) + (16/116)

    imlab = zeros_like(imxyz)

    imlab[:,:,0] = (116*imxyz[:,:,1]) - 16
    imlab[:,:,1] = 500*(imxyz[:,:,0]-imxyz[:,:,1])
    imlab[:,:,2] = 200*(imxyz[:,:,1]-imxyz[:,:,2])

    return imlab

def rgb2lab(im):
    ''' Convert RGB image to CIE-L*ab format'''
    imxyz = rgb2xyz(im)
    imlab = xyz2lab(imxyz)

    return imlab

if __name__ == '__main__':
    im = imread('../output/cam/saved_im_pure.bmp')
    imlab = rgb2lab(im)
    Image.fromarray(imlab[:,:,2]).show()
     
