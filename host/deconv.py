#!/usr/bin/env python

import os, sys

from scipy import *
from scipy.signal import *
from scipy.ndimage import *
from numpy import fft

import Image

kernel_name = 'output/robust_kernel.bmp'
testim_name = 'output/saved_im.bmp'

def wiener_deconv(kernel, im, nsr=0.1):
    """ Wiener deconvolution method"""
    xdim, ydim = im.shape
    K = fft.fft2(kernel, s=(xdim, ydim))
    IM = fft.fft2(im, s=(xdim, ydim))

    IMOUT = IM*conj(K)/(K*conj(K) + nsr)
    imout = fft.ifft2(IMOUT)
    
    return (imout*255.0/imout.max()).astype(uint8)

def reg_deconv(kernel, im, reg_filter = None, alpha=0.1):
    '''Deconvolution using regularization'''
    # Default filter is Laplacian
    if reg_filter == None:
        reg_filter = array([[0, 1,0],
                            [1,-4,1],
                            [0, 1,0]])
    xdim, ydim = im.shape
    IM = fft.fft2(im, s=(xdim, ydim))
    K  = fft.fft2(kernel, s=(xdim, ydim))
    Y  = fft.fft2(reg_filter, s=(xdim, ydim))
    IMOUT = K*IM/(K*conj(K) + alpha*abs(Y)**2)
    imout = fft.ifft2(IMOUT)
    return imout*255.0/imout.max()

if __name__ == '__main__':
    kernel = imread(kernel_name, flatten=True)
    testim = imread(testim_name, flatten=True)
    
    kernel = kernel/sum(kernel)
    blurim = convolve2d(testim, kernel, mode='same')
    Image.fromarray(blurim).show()
    Image.fromarray(testim).show()
    #deblur_im = wiener_deconv(kernel, blurim, 0.1)
    count = 0
    for alpha in arange(0.001, 0.5, 0.001):
        #deblur_im = reg_deconv(kernel, blurim, alpha=alpha)
        deblur_im = wiener_deconv(kernel, blurim, alpha)
        Image.fromarray(deblur_im.astype(uint8)).save('tmp/im'+str(count)+
            '.bmp')
        count += 1

    
