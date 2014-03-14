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

def create_blurred_images(IMPURE, imblur, xpos, ypos):
    max_dist = hypot(x, y).max()
    max_pixel = 10

    count = 0
    xdim, ydim = imblur.shape
    imtemp = zeros((xdim, 2*ydim), dtype=uint8)
    imtemp[:,:ydim] = mag_imblur

    for depth in linspace(0, max_pixel/max_dist, max_pixel):
        print 'Depth = %f'%depth
        kernel = construct_kernel(xpos, ypos, depth)
        KERNEL = abs(fft.fft2(kernel, s=(xdim, ydim)))
        IMREBLUR = KERNEL*IMPURE
        imreblur = real(fft.ifft2(IMREBLUR))
        imtemp[:,ydim:] = imreblur
        Image.fromarray(imtemp).save('../tmp/depth/im%d.bmp'%count)
        count += 1

if __name__ == '__main__':
    try:
        os.mkdir('../tmp/steer')
    except OSError:
        pass
    impure = imread('../output/cam/preview_im.bmp', flatten=True)
    imblur = imread('../output/cam/saved_im.bmp', flatten=True)

    # Load the acceleration data.
    data = loadtxt(accel_data_file)
    start = 41
    end = 63
    x, y, z, g = estimate_simple_pos(data, start, end)

    # Take only the magnitude of the blurred image
    mag_imblur = real(fft.ifft2(abs(fft.fft2(imblur))))
    IMPURE = abs(fft.fft2(impure))
    mag_impure = real(fft.ifft2(IMPURE))

    