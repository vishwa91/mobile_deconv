#!/usr/bin/env python

import os, sys

from scipy import *
from scipy.ndimage import *
from scipy.signal import *
from numpy import fft
from scipy.interpolate import spline

import Image

accel_data_file = '../output/cam/saved_ac.dat'
T = 10e-3
G = 9.8

def shock_filter(im, tstep=0.2, sigma=2, niters=10):
	'''Return an image with shock filtering'''
	laplace_kernel = array([[0,-1,0],
							[-1,4,-1],
							[0,-1,0]])/4.0
	sobelx = array([[ 1, 2, 1],
					[ 0, 0, 0],
					[-1,-2,-1]])
	sobely = array([[-1, 0, 1],
					[-2, 0, 2],
					[-1, 0, 1]])
	imout = im.copy()
	for i in range(niters):
		imout = gaussian_filter(imout, sigma)
		diffx = convolve2d(imout, sobelx)[1:-1, 1:-1]
		diffy = convolve2d(imout, sobely)[1:-1, 1:-1]
		normxy = hypot(diffx, diffy)
		imlap = convolve2d(imout, laplace_kernel)[1:-1, 1:-1]
		sgn = -sign(imlap)
		imout = imout + tstep*sgn*normxy

	return imout

if __name__ == '__main__':
	im = imread('../output/cam/saved_im.bmp', flatten=True)
	imout = shock_filter(im, sigma = 0.5, niters=100)
	Image.fromarray(imout).show()