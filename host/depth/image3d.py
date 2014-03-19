#!/usr/bin/env python

import os, sys
import commands

from matplotlib.pyplot import *
from scipy import *
import numpy
from scipy.signal import *
from scipy.linalg import *
from scipy.interpolate import spline
from scipy.ndimage import *
from scipy.special import *
from numpy import fft

import Image

def register3d(impure, imblur, kernel=None):
    ''' Register and shift two 3d images using fourier correlation method.'''
    IMPURE = fft.fft2(impure[:,:,0])
    IMBLUR = fft.fft2(imblur[:,:,0])

    IMSHIFT = IMPURE*conj(IMBLUR)/(abs(IMPURE)*abs(IMBLUR))
    if kernel is not None:
        xdim, ydim = impure.shape
        KERNEL = fft.fft2(kernel, s=(xdim, ydim))
        IMSHIFT *= abs(KERNEL)/KERNEL
    imshift = real(fft.ifft2(IMSHIFT))
    imshift *= 255.0/imshift.max()
    x, y = where(imshift == imshift.max())
    xdim, ydim = imshift.shape
    if x >= xdim//2:
        x = x- xdim
    if y >= ydim//2:
        y = y - ydim
    
    shift_kernel = zeros((2*abs(x)+1, 2*abs(y)+1))
    shift_kernel[abs(x)-x,abs(y)-y] = 1
    shifted_im0 = fftconvolve(impure[:,:,0], shift_kernel, mode='same')
    shifted_im1 = fftconvolve(impure[:,:,1], shift_kernel, mode='same')
    shifted_im2 = fftconvolve(impure[:,:,2], shift_kernel, mode='same')

    output_im = zeros_like(impure)
    output_im[:,:,0] = shifted_im0
    output_im[:,:,1] = shifted_im1
    output_im[:,:,2] = shifted_im2

    return output_im

def shift3d(im, xyshift):
	''' Shift a 3D image'''
	x, y = xyshift
	imout = zeros_like(im)
	imout[:,:,0] = shift(im[:,:,0], xyshift)
	imout[:,:,1] = shift(im[:,:,1], xyshift)
	imout[:,:,2] = shift(im[:,:,2], xyshift)

	return imout

def linear_convolve3d(im, kernel):
    ''' Convolve the two 3d images in linear fashion but return only the 
        actual image size
    '''
    imout = zeros_like(im)
    imout0 = fftconvolve(im[:,:,0], kernel, 'full')
    imout1 = fftconvolve(im[:,:,1], kernel, 'full')
    imout2 = fftconvolve(im[:,:,2], kernel, 'full')
    kx, ky = kernel.shape
    xdim, ydim = imout0.shape
    if kx%2 == 1:
        kx1, kx2 = kx//2, kx//2
    else:
        kx1 = (kx-1)//2; kx2 = kx//2
    if ky%2 == 1:
        ky1, ky2 = ky//2, ky//2
    else:
        ky1 = (ky-1)//2; ky2 = ky//2
    imout[:,:,0] = imout0[kx1:xdim-kx2, ky1:ydim-ky2]
    imout[:,:,1] = imout1[kx1:xdim-kx2, ky1:ydim-ky2]
    imout[:,:,2] = imout2[kx1:xdim-kx2, ky1:ydim-ky2]
    return imout

def _norm_diff(im1, im2):
	''' Difference of two images after normalization'''
	m1 = mean(im1); m2 = mean(im2)
	v1 = variance(im1); v2 = variance(im2)
	imnorm1 = (im1-m1)/sqrt(v1); imnorm2 = (im2-m2)/sqrt(v2)
	diff_im = (imnorm1 - imnorm2)**2
	return diff_im

def norm_diff3d(im1, im2):
    ''' Difference of two images after normalization'''
    diff_im = zeros_like(im1)
    diff_im[:,:,0] = _norm_diff(im1[:,:,0], im2[:,:,0])
    diff_im[:,:,1] = _norm_diff(im1[:,:,1], im2[:,:,1])
    diff_im[:,:,2] = _norm_diff(im1[:,:,2], im2[:,:,2])

    return mean(diff_im, axis=2)
