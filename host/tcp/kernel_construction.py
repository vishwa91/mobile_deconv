#!/usr/bin/env python

import os, sys

from scipy import *
from scipy.signal import *
from scipy.linalg import *
from numpy import fft

import Image

accel_data_file = '../output/cam/saved_ac.dat'

def estimate_g(data):
    '''Estimate gravity vector using projection onto unit sphere
       What we need to do is minimize the error given that every point lies
       on a unit sphere.
    '''
    # Assume that the data is a stack of x,y, z acceleration
    norm_vector = sqrt(data[:,0]**2 + data[:,1]**2 + data[:,2]**2)
    return data/norm_vector
    
def _estimate_position(accel_vector, nestim=5):
    '''Estimate vector along a single axis'''
    # Model each parabola using nestim samples. If length of accel_vector is not
    # divisible by nestim, discard the tail.
    tail = accel_vector%nestim
    accel_vector = accel_vector[:-tail]
    xlen = len(accel_vector)
    nvar = xlen/nestim
    # our objective function is min. Xt.Q.X + Ct.X where,
    # X = [a1, a2, ...., b1, b2....., c1, c2, .....]
    temp = eye(xlen)    # Stuck here.
    C = -2*accel_vector
    
    
def estimate_position(accel):
    ''' Estimate the trajectory using information from the acceleration values.
        We model chunks of trajectory as a parabola. Using the fact that the 
        position values and velocity values are continuous, we minimize the 
        error between modeled acceleration and actual acceleration.
    '''
    # Estimate g first
    g_vector = estimate_g(accel)
    accel -= g_vector
    xaccel = accel[:,0]; yaccel = accel[:,1]; zaccel = accel[:,2]
    xpos = _estimate_position(xaccel)
    ypos = _estimate_position(yaccel)
    zpos = _estimate_position(zaccel)
