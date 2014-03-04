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

numba_sconv = jit(double[:,:](double[:,:], double[:],
				  double[:], double[:,:]))(sconv)
if __name__ == '__main__':
    impure = imread('../synthetic/random_dot.jpg', flatten=True)
    data = loadtxt(accel_data_file)
    start = 41
    end = 63
    ypos, xpos, zpos, g = estimate_simple_pos(data, start, end)
    # Remove the mean
    xpos -= mean(xpos); ypos -= mean(ypos)
    xdim, ydim = impure.shape
    im = construct_kernel(xpos, ypos, 60000, 10)
    im *= 255.0/im.max()
    Image.fromarray(im.astype(uint8)).save('../tmp/kernel.bmp')
    # Create a dmap
    dmap = imread('../synthetic/depth.gif', flatten=True)
    #dmap[:,:] = 1
    dmax = hypot(xpos,ypos).max() * dmap.max()
    # Restrict yourself to a maximum kernel diameter of 3
    Image.fromarray(dmap*255.0/dmap.max()).show()
    for maxscale in range(1, 15):
        print 'Creating synthetic image for %d max scale'%maxscale
        imblur = sconv(impure, xpos, ypos, dmap*maxscale/dmax)
        Image.fromarray(imblur).convert('RGB').save(
            '../tmp/synthetic_blur/space_variant_blur%d.bmp'%maxscale)
