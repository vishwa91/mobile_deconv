#!/usr/bin/env python

'''
    Routines in this file are used for constructing kernel from accelerometer
    data, visualizing them and manipulating them. 
'''
import os, sys
import commands

from matplotlib.pyplot import *
from scipy import *
from scipy.signal import *
from scipy.linalg import *
from scipy.interpolate import spline
from scipy.integrate import *
from scipy.ndimage import *
from scipy.integrate import *
from numpy import fft

import Image

T = 10e-3
G = 9.8

def estimate_g(data, start=0, end=-1):
    '''
        Estimate gravity vector using projection onto unit sphere
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
    # Convert acceleration into m/s^2
    accel *= G
    xaccel = accel[:,0]; yaccel = accel[:,1]; zaccel = accel[:,2]
    gx = accel[:,3]; gy = accel[:,4]; gz = accel[:,5]
    x = (xaccel - gx)[start:end]
    y = (yaccel - gy)[start:end]
    z = (zaccel - gz)[start:end]
<<<<<<< HEAD
    #raw_xpos = cumsum(cumsum(x))*T*T
    #raw_ypos = cumsum(cumsum(y))*T*T
    #raw_zpos = cumsum(cumsum(z))*T*T

    raw_xpos = cumtrapz(cumtrapz(x))*T*T
    raw_ypos = cumtrapz(cumtrapz(y))*T*T
    raw_zpos = cumtrapz(cumtrapz(z))*T*T    

=======
    raw_xpos = cumtrapz(cumtrapz(x))*T*T
    raw_ypos = cumtrapz(cumtrapz(y))*T*T
    raw_zpos = cumtrapz(cumtrapz(z))*T*T
>>>>>>> 848c7a7fca4a9f526cb6647caacca0b399ae4cac
    return raw_xpos, raw_ypos, raw_zpos

def construct_kernel(xpos, ypos, d=1.0, interpolate_scale = 1):
    '''Construct the kernel from the position data'''
    ntime = len(xpos)
    xpos = d*spline(range(ntime), xpos,
        linspace(0, ntime, ntime*interpolate_scale))
    ypos = d*spline(range(ntime), ypos,
        linspace(0, ntime, ntime*interpolate_scale))
    ntime *= interpolate_scale
    xmax = ceil(max(abs(xpos))); ymax = ceil(max(abs(ypos)))
    kernel = zeros((2*xmax+1, 2*ymax+1), dtype=uint8)
    for i in range(ntime):
        kernel[int(xmax-xpos[i]), int(ymax+ypos[i])] += 1
    return kernel.astype(float)/(kernel.sum()*1.0)
