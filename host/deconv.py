#!/usr/bin/env python

import os, sys

from scipy import *
from scipy.signal import *
from scipy.ndimage import *
from numpy import fft

import Image

kernel_name = 'output/blur_kernel.bmp'
testim_name = 'output/saved_im_noshake.bmp'

def wiener_deconv(kernel, im, nsr=0.1):
    """ Wiener deconvolution method"""
    xdim, ydim = im.shape
    K = fft.fft2(kernel, s=(xdim, ydim))
    IM = fft.fft2(im, s=(xdim, ydim))

    IMOUT = IM*conj(K)/(K*conj(K) + nsr)
    imout = fft.ifft2(IMOUT)
    
    return (imout*255.0/imout.max()).astype(uint8)

if __name__ == '__main__':
    kernel = imread(kernel_name, flatten=True)
    testim = imread(testim_name, flatten=True)
    
    kernel = kernel/sum(kernel)
    blurim = convolve2d(testim, kernel, mode='same')
    Image.fromarray(blurim).show()
    Image.fromarray(testim).show()
    deblur_im = wiener_deconv(kernel, blurim, 0.1)
    Image.fromarray(deblur_im).show()

    
