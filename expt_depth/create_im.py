#!/usr/bin/env python

from scipy import *
from scipy.ndimage import *
from numpy.fft import *
from scipy.signal import convolve2d
import Image

def _blur_im(im, blur_kernel):
	''' Blur a grayscale image'''
	bx, by = blur_kernel.shape
	imx, imy = im.shape
	return convolve2d(im, blur_kernel)

def blur_im(im, blur_kernel):
	'''
		Function to blur an image.
	'''
	imshape = im.shape
	if len(imshape) == 2:
		# A grayscale image
		return _blur_im(im, blur_kernel)
	else:
		# An RGB image. Blur each channel and add it up
		imr = im[:,:,0]; img = im[:,:,1]; imb = im[:,:,2]
		imrblur = _blur_im(imr, blur_kernel)
		imgblur = _blur_im(img, blur_kernel)
		imbblur = _blur_im(imb, blur_kernel)
		x, y = imrblur.shape
		imnew = zeros((x, y, 3))
		imnew[:,:,0]=imrblur; imnew[:,:,0]=imgblur; imnew[:,:,0]=imbblur; 
		return imnew

if __name__ == '__main__':
	blur_kernel = zoom(imread('blur_kernel.bmp', flatten='True'), 2)
	imfront = imread('imfront.bmp')
	imback = imread('imback.bmp')
	# The blur kernel for the back will be a little smaller
	blur_back_temp = zoom(blur_kernel, 0.4)
	blur_back = zeros_like(blur_kernel)
	x1, y1 = blur_kernel.shape
	x2, y2 = blur_back_temp.shape
	cx, cy = x1//2, y1//2
	blur_back[cx-x2//2:cx+x2//2, cy-y2//2:cy+y2//2] = blur_back_temp
	imfront_blur = blur_im(imfront, blur_kernel)
	imback_blur = blur_im(imback, blur_back)
	imblur = imfront_blur+imback_blur
	pmax = imblur.max()
	imblur *= 255.0/pmax
	Image.fromarray((imfront_blur*255.0/imfront_blur.max()).astype(uint8)).convert('RGB'
		).save('imbfront_blur.bmp')
	Image.fromarray((imback_blur*255.0/imback_blur.max()).astype(uint8)).convert('RGB'
		).save('imbback_blur.bmp')
	Image.fromarray(imblur.astype(uint8)).convert('RGB').save('imblur.bmp')
