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

def sfilter_rect(im, window_min=7, window_max=63, nlevels=3):
	''' Filter the input image using space variant filtering'''
	max_level, min_level = im.max(), im.min()
	n_im_levels = linspace(min_level, max_level, nlevels)
	n_sigma_levels = linspace(window_min, window_max, nlevels)[::-1]
	im_filtered = zeros_like(im)
	im_mask = zeros_like(im)
	immask_now = zeros_like(im)
	imnow = zeros_like(im)
	imtemp = im.copy()
	for i in range(nlevels):
		x, y = where(imtemp <= n_im_levels[i])
		imnow[:,:] = 0
		imnow[x, y] = imtemp[x, y]
		imtemp[x, y] = float('inf')
		immask_now[:,:] = 0
		immask_now[x, y] = 1.0
		w = int(n_sigma_levels[i])
		kernel = ones((w,w))
		im_mask += fftconvolve(immask_now, kernel/kernel.sum(), mode='same')
		im_filtered += fftconvolve(imnow, kernel/kernel.sum(), mode='same')
	return im_filtered/im_mask

def sfilter_gauss(im, sigma_min=0.1, sigma_max=3, nlevels=3):
	''' Filter the input image using space variant filtering'''
	max_level, min_level = im.max(), im.min()
	n_im_levels = linspace(min_level, max_level, nlevels)
	n_sigma_levels = linspace(sigma_min, sigma_max, nlevels)[::-1]
	im_filtered = zeros_like(im)
	im_mask = zeros_like(im)
	immask_now = zeros_like(im)
	imnow = zeros_like(im)
	imtemp = im.copy()
	for i in range(nlevels):
		x, y = where(imtemp <= n_im_levels[i])
		imnow[:,:] = 0
		imnow[x, y] = imtemp[x, y]
		imtemp[x, y] = float('inf')
		immask_now[:,:] = 0
		immask_now[x, y] = 1.0
		im_mask += gaussian_filter(immask_now, n_sigma_levels[i])
		im_filtered += gaussian_filter(imnow, n_sigma_levels[i])

	return im_filtered/im_mask

if __name__ == '__main__':
	im = imread('../output/cam/saved_im.bmp', flatten=True)
	laplace_kernel = array([[0,-1,0],
							[-1,4,-1],
							[0,-1,0]])
	imlap = convolve2d(im, laplace_kernel)[1:-1, 1:-1]
	im_filtered = sfilter(imlap)
	Image.fromarray(imlap*100).show()
	Image.fromarray(im_filtered*100).show()
