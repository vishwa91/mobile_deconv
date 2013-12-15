#!/usr/bin/env python

import os, sys

from scipy import *
from scipy.signal import *
from scipy.ndimage import *
from numpy import fft

import Image

kernel_name = 'output/blur_kernel.bmp'
testim_name = 'output/synthetic.bmp'

def wiener_deconv(kernel, im, nsr=0.1):
    """ Wiener deconvolution method"""
    xdim, ydim = im.shape
    K = fft.fft2(kernel, s=(xdim, ydim))
    IM = fft.fft2(im, s=(xdim, ydim))

    IMOUT = IM*conj(K)/(K*conj(K) + nsr)
    imout = fft.ifft2(IMOUT)
    
    return (imout*255.0/imout.max()).astype(uint8)

def reg_deconv(kernel, im, reg_filters = None, alpha=0.1):
    '''Deconvolution using regularization. The concept is to use 
       gaussian priors.
    '''
    # Default filter is Laplacian
    if reg_filters == None:
        
        filter1 = array([[1, 1, 1],
                         [0, 0, 0],
                         [-1,-1,-1]])
        filter2 = array([[1, 0,-1],
                         [1, 0,-1],
                         [1, 0,-1]])
        '''
        filter1 = array([[0, 1, 0],
                         [1,-4, 1],
                         [0, 1, 0]])
        '''
        reg_filters = [filter1, filter2]
		
    xdim, ydim = im.shape
    IM = fft.fft2(im, s=(xdim, ydim))
    K  = fft.fft2(kernel, s=(xdim, ydim))
    Y = zeros_like(IM)
    for reg_filter in reg_filters:
        temp = fft.fft2(reg_filter, s=(xdim, ydim))
        Y += abs(temp)**2
    IMOUT = conj(K)*IM/(abs(K)**2 + alpha*Y)
    imout = fft.ifft2(IMOUT)
    return imout*255.0/imout.max()

if __name__ == '__main__':
    kernel = imread(kernel_name, flatten=True)
    testim = imread(testim_name, flatten=True)
    
    kernel = kernel/sum(kernel)
    #blurim = convolve2d(testim, kernel, mode='same')
    #Image.fromarray(blurim).show()
    #Image.fromarray(testim).show()
    #deblur_im = wiener_deconv(kernel, blurim, 0.1)
    count = 0
    # Try making the tmp directory
    try:
		os.mkdir('tmp')
    except OSError:
		pass
    for alpha in arange(0.001, 0.01, 0.001):
        #deblur_im = wiener_deconv(kernel, testim, nsr=alpha)
        deblur_im = reg_deconv(kernel, testim, None, alpha)
        Image.fromarray(deblur_im.astype(uint8)
            ).save('tmp/im%d.bmp'%count)
        count += 1
