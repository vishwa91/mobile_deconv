#!/usr/bin/env python

from scipy import *
from scipy.ndimage import *
from scipy.signal import *

import Image

def _ssim(im1, im2):
	''' Calculate SSIM for a single patch'''
	m1 = mean(im1); m2 = mean(im2)
	v1 = variance(im1); v2 = variance(im2)
	cv = mean(im1*im2) - m1*m2
	c1 = (0.01*255)**2; c2 = (0.03*255)**2
	numerator = (2*m1*m2 + c1)*(2*cv+c2)
	denominator = (m1**2 + m2**2 + c1)*(v1 + v2 + c2)
	return numerator/denominator

def calculate_ssim(im1, im2, w=8):
	''' Calculate the structural similarity of two images'''
	xdim, ydim = im1.shape
	d = w//2
	im_ssim = zeros_like(im1)
	for x in range(d, xdim-d, d):
		for y in range(d, ydim-d, d):
			im_ssim[x-d:x+d, y-d:y+d] = _ssim(im1[x-d:x+d, y-d:y+d], im2[x-d:x+d, y-d:y+d])
	return im_ssim

if __name__ == '__main__':
	im1 = imread('../synthetic/test.jpg', flatten=True)
	im2 = imread('../tmp/space_variant_blur.bmp', flatten=True)
	im_ssim = calculate_ssim(im1, im1)
	Image.fromarray(im_ssim*255.0/im_ssim.max()).show()