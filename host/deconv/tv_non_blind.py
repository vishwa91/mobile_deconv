#!/usr/bin/env python

import os, sys

from scipy import *
from scipy.ndimage import *
from scipy.signal import *
from numpy import fft

import Image

'''
	The concept of Total Variation (TV) deconvolution is to rely on the fact
	that images are piecewise smooth. This would mean that the total variation
	of the signal is small. Hence we try to minimize this. However, we also 
	need to remember the fact that the output signal has to be close to the 
	input signal. Hence, we try to minimize the L2 norm. Overall, we have

	Minimize Z = |y-h*x|^2 + lambda X TV(x)
	Where TV(x) is defined as hypot(diffx(x), diffy(x))
	Here, diffx(x) is the differentiation of x along x axis
'''
def tv(im):
	''' Return the total variance of the image'''
	diffx = zeros_like(im); diffy = zeros_like(im)
	diffx[:-1, :] = im[1:, :] - im[:-1, :]
	diffy[:, :-1] = im[:, 1:] - im[:, :-1]
	tv_mat = hypot(diffx, diffy)
	return tv_mat.sum()
	
def tv_deconv(im, kernel):
	''' Deconvolve the image using Total Variation. '''
