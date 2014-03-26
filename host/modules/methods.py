#!/usr/bin/env python

'''
    Routines in this file are miscellaneous methods for operating on images.
'''
import os, sys
import commands

from matplotlib.pyplot import *
from scipy import *
from scipy.signal import *
from scipy.linalg import *
from scipy.interpolate import spline
from scipy.ndimage import *
from scipy.special import *
from numpy import fft

import Image

def register(impure, imblur, kernel=None):
    ''' Register and shift the pure image using fourier correlation method.'''
    IMPURE = fft.fft2(impure)
    IMBLUR = fft.fft2(imblur)

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
    shifted_im = fftconvolve(impure, shift_kernel, mode='same')

    return shifted_im
    
def computer_path_diff(im, x, y):
    ''' A very experimental function. We calcluate what we call the path 
        differential of the image, given x vector and y vector.'''
    #Create the basis kernels for a steerable filter.
    sobelx = array([[ 1, 2, 1],
                 [ 0, 0, 0],
                 [-1,-2,-1]])
    sobely = array([[-1, 0, 1],
                 [-2, 0, 2],
                 [-1, 0, 1]])
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    imfinal = zeros_like(im)
    imfinal[:,:] = float("inf")
    for i in range(len(dx)):
        cosx = dx[i]/hypot(dx[i],dy[i])
        sinx = dy[i]/hypot(dx[i],dy[i])
        diff_kernel = sobelx*cosx + sobely*sinx
        imdiff = convolve2d(impure, diff_kernel)[1:-1, 1:-1]
        xmin, ymin = where(imdiff <= imfinal)
        imfinal[xmin, ymin] = imdiff[xmin, ymin]
    imfinal *= 255.0/imfinal.max()
    return imfinal

def compute_var(im, window=5):
    ''' Compute the variance map of the image'''
    var_map = zeros_like(im)
    xdim, ydim = im.shape

    for x in range(0, xdim, window):
        for y in range(0, ydim, window):
            mvar = variance(im[x:x+window, y:y+window])
            var_map[x:x+window, y:y+window] = mvar

    return var_map

def sconv(im, xcoords, ycoords, dmap):
    '''
        Convolve the image using space variant convolution. The xcoords and the 
        ycoords will be scaled by the value of dmap 
    '''
    xdim, ydim = im.shape
    final_im = zeros_like(im)
    avg_map = zeros_like(im)
    w = float(len(xcoords))
    for xidx in range(xdim):
        for yidx in range(ydim):
            # For each pixel, 'Spread' it and add it to the empty image.
            xshifts = xidx + dmap[xidx, yidx]*xcoords
            yshifts = yidx - dmap[xidx, yidx]*ycoords
            illegalx = where((xshifts>=xdim))
            illegaly = where((yshifts>=ydim))
            xshifts[illegalx] = xdim-1; yshifts[illegaly] = ydim-1;
            final_im[xshifts.astype(int), yshifts.astype(int)] += (
                im[xidx, yidx])
            #final_im[xidx, yidx] += (
            #   im[xshifts.astype(int), yshifts.astype(int)]).sum()
            avg_map[xshifts.astype(int), yshifts.astype(int)] += 1
    xz, yz = where(avg_map == 0)
    avg_map[xz, yz] = 1
    return final_im/avg_map

def max_filter(im, w):
    ''' Filter the image using maximum filter'''
    d = w//2
    xdim, ydim = im.shape
    imfiltered = zeros_like(im)
    for x in range(d, xdim-d):
        for y in range(d, ydim-d):
            imfiltered[x-d:x+d, y-d:y+d] = im[x-d:x+d, y-d:y+d].min()
    return imfiltered
    
def mquantize(im, nlevels=5):
    ''' Quantize the image for the given number of levels'''
    vmin = im.min(); vmax = im.max()
    levels = linspace(vmin, vmax, nlevels)
    curr_min = 0
    for level in levels:
        xd, yd = where((im>curr_min) * (im<level))
        im[xd, yd] = curr_min
        curr_min = level
    return im, levels

def linear_convolve(im, kernel):
    ''' Convolve the two images in linear fashion but return only the 
        actual image size
    '''
    imout = fftconvolve(im, kernel, 'full')
    kx, ky = kernel.shape
    xdim, ydim = imout.shape
    if kx%2 == 1:
        kx1, kx2 = kx//2, kx//2
    else:
        kx1 = (kx-1)//2; kx2 = kx//2
    if ky%2 == 1:
        ky1, ky2 = ky//2, ky//2
    else:
        ky1 = (ky-1)//2; ky2 = ky//2
    return imout[kx1:xdim-kx2, ky1:ydim-ky2]

def spacial_fft(im, xw=8, yw=8):
    ''' Compute the spacial fft of the image with given window size'''
    imfft = zeros_like(im, dtype=complex)
    xdim, ydim = im.shape
    for x in range(0, xdim, xw):
        for y in range(0, ydim, yw):
            imfft[x:x+xw, y:y+yw] = fft.fft2(im[x:x+xw, y:y+yw])
    return imfft

def spacial_ifft(im, xw=8, yw=8):
    ''' Compute the spacial ifft of the image with given window size'''
    imifft = zeros_like(im, dtype=complex)
    xdim, ydim = im.shape
    for x in range(0, xdim, xw):
        for y in range(0, ydim, yw):
            imifft[x:x+xw, y:y+yw] = fft.ifft2(im[x:x+xw, y:y+yw])
    return imifft

def spacial_corr(imblur, imreblur, xw=8, yw=8):
    '''Find the correlation map of the two images'''
    IMBLUR = spacial_fft(imblur, xw, yw)
    IMREBLUR = spacial_fft(imreblur, xw, yw)

    IMCORR = IMBLUR*conj(IMREBLUR)/(abs(IMBLUR*IMREBLUR))
    imcorr = spacial_ifft(IMCORR, xw, yw)
    xdim, ydim = imcorr.shape
    for x in range(0, xdim, xw):
        for y in range(0, ydim, yw):
            imcorr[x:x+xw,y:y+yw] = imcorr[x,y]
    return real(imcorr)

def do_clustering(im, ncenters=3):
    ''' Segment the image using kmeans clustering. The image is expected
        to be in Lab format
        '''
    xdim, ydim, nchannels = im.shape
    im_vec = im.copy().reshape((xdim*ydim, nchannels))
    centroids, _ = kmeans(im_vec, ncenters)
    idx, _ = vq(im_vec, centroids)
    for i in range(ncenters):
        x = where(idx == i)
        im_vec[x] = i
    imout = im_vec.reshape((xdim, ydim, nchannels))

    return imout*255.0/ncenters
    
def rgb2xyz(im):
    ''' Convert RGB image to XYZ image. Algorithm from www.easyrgb.com'''
    # Normalize the image first.
    imtemp = im.copy()/255.0
    x1, y1, z1 = where(imtemp > 0.0405)
    x2, y2, z2 = where(imtemp <= 0.0405)

    imtemp[x1, y1, z1] = pow((imtemp[x1, y1, z1] + 0.055)/1.055, 2.4)
    imtemp[x2, y2, z2] = imtemp[x2, y2, z2]/12.92
    
    imtemp *= 100
    
    imxyz = zeros_like(imtemp)
    imxyz[:,:,0] = imtemp[:,:,0]*0.4124 + (
                   imtemp[:,:,1]*0.3576) + (
                   imtemp[:,:,2]*0.1805)
    imxyz[:,:,1] = imtemp[:,:,0]*0.2126 + (
                   imtemp[:,:,1]*0.7152) + (
                   imtemp[:,:,2]*0.0722)
    imxyz[:,:,2] = imtemp[:,:,0]*0.0193 + (
                   imtemp[:,:,1]*0.1192) + (
                   imtemp[:,:,2]*0.9505)
    return imxyz

def xyz2lab(im):
    ''' Convert XYZ image to CIE-L*ab format'''
    imxyz = im.copy()
    imxyz[:,:,0] /= 95.047
    imxyz[:,:,1] /= 100.00
    imxyz[:,:,2] /= 108.883

    x1, y1, z1 = where(imxyz > 0.008856)
    x2, y2, z2 = where(imxyz <= 0.008856)
    imxyz[x1, y1, z1] = pow(imxyz[x1, y1, z1], 1.0/3.0)
    imxyz[x2, y2, z2] = (7.787*imxyz[x2, y2, z2]) + (16/116)

    imlab = zeros_like(imxyz)

    imlab[:,:,0] = (116*imxyz[:,:,1]) - 16
    imlab[:,:,1] = 500*(imxyz[:,:,0]-imxyz[:,:,1])
    imlab[:,:,2] = 200*(imxyz[:,:,1]-imxyz[:,:,2])

    return imlab

def rgb2lab(im):
    ''' Convert RGB image to CIE-L*ab format'''
    imxyz = rgb2xyz(im)
    imlab = xyz2lab(imxyz)

    return imlab
    
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
    
def linear_convolve(im, kernel):
    ''' Convolve the two images in linear fashion but return only the 
        actual image size
    '''
    imout = fftconvolve(im, kernel, 'full')
    kx, ky = kernel.shape
    xdim, ydim = imout.shape
    if kx%2 == 1:
        kx1, kx2 = kx//2, kx//2
    else:
        kx1 = (kx-1)//2; kx2 = kx//2
    if ky%2 == 1:
        ky1, ky2 = ky//2, ky//2
    else:
        ky1 = (ky-1)//2; ky2 = ky//2
    return imout[kx1:xdim-kx2, ky1:ydim-ky2]

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
    
