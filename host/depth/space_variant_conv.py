#!/usr/bin/env python

import os, sys

from scipy import *
from scipy.ndimage import *
from scipy.signal import *
from numpy import fft
from scipy.interpolate import spline

import Image
from numba import jit, double

accel_data_file = '../output/cam/saved_ac.dat'
T = 10e-3
G = 9.8

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
			yshifts = yidx + dmap[xidx, yidx]*ycoords
			illegalx = where((xshifts>=xdim))
			illegaly = where((yshifts>=ydim))
			xshifts[illegalx] = xdim-1; yshifts[illegaly] = ydim-1;
			final_im[xshifts.astype(int), yshifts.astype(int)] += (
				im[xidx, yidx])
			#final_im[xidx, yidx] += (
			#	im[xshifts.astype(int), yshifts.astype(int)]).sum()
			avg_map[xshifts.astype(int), yshifts.astype(int)] += 1
	x, y = where(avg_map == 0)
	avg_map[x, y] = 1
	return final_im/avg_map

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

def sconv2(impure, dmap, xpos, ypos, nlevels=2, scale=1):
    ''' Return spacially convolved image'''
    dmap /= dmap.max()
    levels = linspace(0, 1.1, nlevels)
    xdim, ydim = impure.shape
    imblurred = zeros_like(impure)
    for level in levels:
        x, y = where(dmap < level)
        depth = level
        dmap[x, y] = float('inf')
        kernel = construct_kernel(xpos, ypos, depth*scale)
        imtemp = linear_convolve(impure, kernel)
        imblurred[x, y] = imtemp[x, y]
    return imblurred

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

def construct_kernel(xpos, ypos, d=1.0, interpolate_scale = 10):
    '''Construct the kernel from the position data'''
    ntime = len(xpos)
    xpos = d*spline(range(ntime), xpos,
        linspace(0, ntime, ntime*interpolate_scale))
    ypos = d*spline(range(ntime), ypos,
        linspace(0, ntime, ntime*interpolate_scale))
    ntime *= interpolate_scale
    xpos -= mean(xpos); ypos -= mean(ypos)
    xmax = max(abs(xpos)); ymax = max(abs(ypos))
    kernel = zeros((2*xmax+1, 2*ymax+1), dtype=uint8)
    for i in range(ntime):
        kernel[xmax+int(xpos[i]), ymax-int(ypos[i])] += 1
    return kernel.astype(float)/(kernel.sum()*1.0)

if __name__ == '__main__':
    impure = imread('../synthetic/test.jpg', flatten=True)
    data = loadtxt(accel_data_file)
    start = 41
    end = 63
    xpos, ypos, zpos, g = estimate_simple_pos(data, start, end)
    
    xdim, ydim = impure.shape
    dmax = hypot(xpos, ypos).max()
    dmap = imread('../synthetic/depth.gif', flatten=True)
    dmax = hypot(xpos,ypos).max()
    # Restrict yourself to a maximum kernel diameter of 10
    Image.fromarray(dmap*255.0/dmap.max()).show()
    dmap /= dmap.max()
    maxscale = 20
    imblur = sconv(impure, xpos, ypos, dmap*maxscale/dmax)
    imblur = sconv2(impure, dmap, xpos, ypos, nlevels=4, scale=maxscale/dmax)
    Image.fromarray(imblur).convert('RGB').save(
        '../tmp/synthetic_blur/space_variant_blur.bmp')
    print (xpos*maxscale/dmax).astype(int)
    print (ypos*maxscale/dmax).astype(int)