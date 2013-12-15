#!/usr/bin/env python

import os
import sys

from scipy import *
from scipy.ndimage import *
from scipy.signal import *
from numpy import fft
import matplotlib.pyplot as plt
import Image

# global constants
IMTEST = 'imblur.bmp'
KERNEL = 'blur_kernel.bmp'
TMPDIR = 'tmp'

def _wiener_deconv(kernel, im, nsr):
	'''
		Deconvolve the image using wiener deconvolution method. The reason for
		using a simple and ineffective method is that implementation of any
		better method on a mobile is not feasible.

		This function deconvoles only a grayscale image. Don't use this function
		directly. Instead, use wiener_deconv.
	'''
	xdim, ydim = im.shape
	IM = fft.fft2(im, s=(xdim, ydim))
	H  = fft.fft2(kernel, s=(xdim, ydim))

	# Normalise kernel.
	kernel /= sum(kernel)
	# At this stage, we won't care about boundary. That is a problem for another
	# day.
	IMOUT = conj(H)*IM/(abs(H)**2 + nsr)
	imout = fft.ifft2(IMOUT)
	return imout

def wiener_deconv(kernel, im, nsr):
	'''
		Deconvolve the image using wiener deconvolution method. The reason for
		using a simple and ineffective method is that implementation of any
		better method on a mobile is not feasible.
	'''
	imshape = im.shape
	if len(imshape) == 2:
		return _wiener_deconv(kernel, im, nsr)
	else:
		# An RGB image. Deconvolve all the three channels separately
		imr = im[:,:,0]; img = im[:,:,1]; imb = im[:,:,2]
		imnew = zeros_like(im)
		imrout = _wiener_deconv(kernel, imr, nsr)
		imgout = _wiener_deconv(kernel, img, nsr)
		imbout = _wiener_deconv(kernel, imb, nsr)	
		imnew[:,:,0] = imrout; imnew[:,:,1] = imgout; imnew[:,:,2] = imbout 

		# Typecast it to uint8. Some problem with Image module.
		return imnew.astype(uint8)	

if __name__ == '__main__':
	# Create the temporary directory if necessary
	try:
		os.mkdir(TMPDIR)
	except OSError:
		print 'Temporary directory exists.'
		pass
	# Load image and kernel.
	kernel = imread(KERNEL, flatten=True)
	im = imread(IMTEST)

	nsr = 0.08
	count = 0
	for scale in arange(0.1, 4, 0.1):
		# Deblur at various scales.
		print 'Deblurring at %f scale'%scale
		imout = wiener_deconv(zoom(kernel, scale), im, nsr)
		imname = '%s/im_%d.bmp'%(TMPDIR, int(scale*10))
		Image.fromarray(imout).save(imname)