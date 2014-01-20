#!/usr/bin/env python

import os, sys

from scipy import *
from scipy.signal import *
from scipy.ndimage import *
from scipy import random
from numpy import fft

import Image

TMP_DIR = '../tmp/deconv'
kernel_name = '../tmp/cam/0_kernel_0.002000_0.018000_20000.000000.bmp'
testim_name = '../tmp/cam/0_im.bmp'
goodim_name = '../output/cam/dot.bmp'

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
        
        filter3 = array([[0, 1, 0],
                         [1,-4, 1],
                         [0, 1, 0]])
        
        reg_filters = [filter1, filter2, filter3]
		
    xdim, ydim = im.shape
    IM = fft.fft2(im, s=(xdim, ydim))
    K  = fft.fft2(kernel, s=(xdim, ydim))
    Y = zeros_like(IM)
    for reg_filter in reg_filters:
        temp = fft.fft2(reg_filter, s=(xdim, ydim))
        Y += (abs(temp)**2)/3.0
    IMOUT = conj(K)*IM/(abs(K)**2 + alpha*Y)
    imout = fft.ifft2(IMOUT)
    return imout*255.0/imout.max()

if __name__ == '__main__':
    kernel = imread(kernel_name, flatten=True)
    #testim = imread(testim_name, flatten=True)
    goodim = imread(goodim_name, flatten=True)
    
    kernel = kernel/sum(kernel)
    
    # Synthetically create the image.
    testim = fftconvolve(goodim, kernel)
    
    # Add noise.
    s = testim.shape
    noise = random.normal(0, 3, s)
    testim += noise
    count = 0
    # Try making the tmp directory
    try:
		os.mkdir(TMP_DIR)
    except OSError:
		pass
    alpha = 0.0001
    Image.fromarray(testim).convert('RGB').save(
                os.path.join(TMP_DIR, 'blurred.bmp'))
    for scale in arange(0.5, 2, 0.1):
        print 'Deblurring for scale =', scale
        deblur_im = wiener_deconv(zoom(kernel, scale), testim, nsr=alpha)
        #deblur_im = reg_deconv(zoom(kernel, scale), testim, None, alpha)
        Image.fromarray(deblur_im.astype(uint8)
            ).save(os.path.join(TMP_DIR, 'im%d.bmp'%count))
        count += 1
