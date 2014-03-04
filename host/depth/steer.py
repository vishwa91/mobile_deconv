#!/usr/bin/env python

import os, sys

from scipy import *
from scipy.ndimage import *
from scipy.signal import *
from numpy import fft

import Image

accel_data_file = '../output/cam/saved_ac.dat'
T = 10e-3
G = 9.8

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

def computer_path_diff(im, x, y, mode='along'):
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
        if mode == 'along':
            cosx = dx[i]/hypot(dx[i],dy[i])
            sinx = dy[i]/hypot(dx[i],dy[i])
        elif mode == 'perpendicular':
            sinx = dx[i]/hypot(dx[i],dy[i])
            cosx = dy[i]/hypot(dx[i],dy[i])
        diff_kernel = sobelx*cosx + sobely*sinx
        imdiff = convolve2d(im, diff_kernel)[1:-1, 1:-1]
        xmin, ymin = where(imdiff <= imfinal)
        imfinal[xmin, ymin] = imdiff[xmin, ymin]
    imfinal *= 255.0/imfinal.max()
    return imfinal

if __name__ == '__main__':
    # Load image
    imblur = imread('../output/cam/saved_im.bmp', flatten=True) 
    # Load the acceleration data.
    data = loadtxt(accel_data_file)
    start = 41; end = 63
    x, y, z, g = estimate_simple_pos(data, start, end)
    diff_im = computer_path_diff(imblur, x, y, 'perpendicular')
    Image.fromarray(diff_im).show()