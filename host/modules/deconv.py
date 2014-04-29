#!/usr/bin/env python

'''
    Routines in this file are mostly concerned with deconvolution of an image
    using non-blind methods.
'''
import os, sys, commands

from scipy import *
from scipy.signal import *
from scipy.ndimage import *
from scipy import random
from numpy import fft

import Image

def non_blind_deconv(kernel, im, output_name):
    ''' Deconvolve using the non_blind_deconv executable'''
    
    # Save the data first
    Image.fromarray(kernel).convert('L').save('tmp/kernel.bmp')
    Image.fromarray(im).convert('L').save('tmp/imblur.bmp')
    
    # Execute the command
    exec_cmd = ('wine tools/non_blind_deblur/robust_deconv.exe ' 
                'tmp/imblur.bmp tmp/kernel.bmp %s 0 0.001 10'%output_name) 
    print exec_cmd
    cmd_op = commands.getoutput(exec_cmd)
    return cmd_op
    
def wiener_deconv(kernel, im, alpha=0.1):
    """ Wiener deconvolution method"""
    xdim, ydim = im.shape
    K = fft.fft2(kernel, s=(xdim, ydim))
    IM = fft.fft2(im, s=(xdim, ydim))

    IMOUT = IM*conj(K)/(K*conj(K) + alpha)
    imout = fft.ifft2(IMOUT)
    
    return (imout).astype(uint8)

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
        reg_filters = [filter1, filter2]		
    xdim, ydim = im.shape
    IM = fft.fft2(im, s=(xdim, ydim))
    K  = fft.fft2(kernel, s=(xdim, ydim))
    Y = zeros_like(IM)
    for reg_filter in reg_filters:
        temp = fft.fft2(reg_filter, s=(xdim, ydim))
        Y += (abs(temp)**2)
    IMOUT = conj(K)*IM/(abs(K)**2 + alpha*Y)
    imout = fft.ifft2(IMOUT)
    return imout#*255.0/imout.max()
