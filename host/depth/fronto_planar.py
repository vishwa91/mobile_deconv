#!/usr/bin/env python

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
import ssim

accel_data_file = '../output/cam/saved_ac.dat'
T = 10e-3
G = 9.8

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

def estimate_g(data, start=0, end=-1):
    '''Estimate gravity vector using projection onto unit sphere
       What we need to do is minimize the error given that every point lies
       on a unit sphere.
    '''
    if end == len(data[:,0]):
        end = -1
    # Assume that the data is a stack of x,y, z acceleration
    mx = mean(data[start:end, 0])
    my = mean(data[start:end, 1])
    mz = mean(data[start:end, 2])
    norm_g = sqrt(mx**2 + my**2 + mz**2)
    output = zeros_like(data)
    output[:,0] = mx/norm_g; output[:,1] = my/norm_g; output[:,2] = mz/norm_g; 

    return output

def construct_kernel(xpos, ypos, d=1.0, interpolate_scale = 1):
    '''Construct the kernel from the position data'''
    ntime = len(xpos)
    xpos = d*spline(range(ntime), xpos,
        linspace(0, ntime, ntime*interpolate_scale))
    ypos = d*spline(range(ntime), ypos,
        linspace(0, ntime, ntime*interpolate_scale))
    ntime *= interpolate_scale
    #xpos -= mean(xpos); ypos -= mean(ypos)
    xmax = max(abs(xpos)); ymax = max(abs(ypos))
    kernel = zeros((2*xmax+1, 2*ymax+1), dtype=uint8)
    for i in range(ntime):
        kernel[int(xmax+xpos[i]), int(ymax-ypos[i])] += 1
    return kernel.astype(float)/(kernel.sum()*1.0)

def estimate_simple_pos(accel, start, end):
    ''' Simple calculation of position using just integration'''
    if end == len(accel[:,0]):
        end = -1
    # Estimate g first
    g_vector = estimate_g(accel)
    accel -= g_vector
    # Convert acceleration into m/s^2
    accel *= G
    xaccel = accel[:,0]; yaccel = accel[:,1]; zaccel = accel[:,2]
    raw_xpos = cumsum(cumsum(xaccel[start:end]))*T*T
    raw_ypos = cumsum(cumsum(yaccel[start:end]))*T*T
    raw_zpos = cumsum(cumsum(zaccel[start:end]))*T*T

    return raw_xpos, raw_ypos, raw_zpos, g_vector

def norm_diff(im1, im2):
    ''' Difference of two images after normalization'''
    m1 = mean(im1); m2 = mean(im2)
    v1 = variance(im1); v2 = variance(im2)
    imnorm1 = (im1-m1)/sqrt(v1); imnorm2 = (im2-m2)/sqrt(v2)
    diff_im = (imnorm1 - imnorm2)**2
    return diff_im

def iterative_depth(impure, imblur, xpos, ypos, mkernel=None):
    ''' Estimate the depth using multiple iterations. Rudimentary, but expected
        to work.
    '''
    best_depth = float('inf')
    w = 15
    avg_filter = ones((w,w))/(w*w*1.0)
    xdim, ydim = impure.shape
    xw = 32; yw = 32
    dmax = hypot(xpos, ypos).max()
    count = 0
    nlevels = 40
    curr_diff = float('inf')
    for depth in linspace(0, nlevels/dmax, nlevels):
        print 'Iteration for %f depth'%depth
        if mkernel == None:
            kernel = construct_kernel(xpos, ypos, depth)
        else:
            kernel = zoom(mkernel, depth)
            kernel /= (1e-5+kernel.sum())
        imreblur = fftconvolve(impure, kernel, mode='same')
        imreblur = register(imreblur, imblur, kernel)
        imsave = norm_diff(imreblur, imblur).sum()
        if imsave < curr_diff:
        	best_depth = depth
        	curr_diff = imsave
        count += 1
    return best_depth

if __name__ == '__main__':
	# Load images
	impure = imread('../output/cam/preview_im.bmp', flatten=True)
	imblur = imread('../output/cam/saved_im.bmp', flatten=True)
	# Load the acceleration data.
	data = loadtxt(accel_data_file)
	start = 41
	end = 93
	x, y, z, g = estimate_simple_pos(data, start, end)
	x -= mean(x); y -= mean(y)

	best_depth = iterative_depth(impure, imblur, x, y)
	print best_depth
	kernel = construct_kernel(x, y, best_depth, 10)
	imreblur = fftconvolve(kernel, impure)
	Image.fromarray(imreblur).convert('RGB').save('../tmp/imreblur.bmp')