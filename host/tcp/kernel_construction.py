#!/usr/bin/env python

import os, sys

from scipy import *
from scipy.signal import *
from scipy.linalg import *
from numpy import fft

import Image

accel_data_file = '../output/cam/saved_ac.dat'
t = 10e-3

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
    temp = eye(nvar)     
    Q = zeros((3*nvar, 3*nvar))
    Q[2*nvar:, 2*nvar:] = temp
    C = zeros((3*nvar, 1))
    for i in range(0, xlen, nestim):
        C[2*nvar+i/nestim, 0] = sum(accel_vector[i:i+nestim])
    # E is our constrain matrix. It is a 2nvar x 3nvar matrix.
    E = zeros((2*nvar, 3*nvar))
    # RHS of the constrain.
    d = zeros((2*nvar, 1))
    
    # Create constrain matrix.
    for i in range(0, nvar):
        # ai are 0->nvar-1; bi are nvar->2nvar-1, ci are 2nvar->3nvar-1 

        # First nvar equations are position continuity equations and the
        # next nvar equations are velocity continuity equations.

        # Position continuity equation:
        # ai - ai+1 + bi(ti-nestim*t*i) - bi+1(-nestim*t*i+1) +
        # ci(tj-nestim*t*j)^2 - ci+1(-nestim*t*i+1)^2 = 0
        E[i, i] += 1; E[i, i+1] += -1;
        E[i, nvar+i] += t*i - nestim*t*i; E[i, nvar+i+1] += nestim*t*(i+1);
        E[i, 2*nvar+i] += (t*i - nestim*t*i)**2;
        E[i, 2*nvar+i+1] -= (nestim*t*(i+1))**2

        # Velocity continuity equation:
        # bi - bi+1 + ci(ti-nestim*t*i) - ci+1(-nestim*t*i+1) = 0
        E[i+nvar, i+nvar] += 1; E[i+nvar, i+nvar+1] -= 1;
        E[i+nvar, i+2*nvar] += t*i - nestim*t*i;
        E[i+nvar, i+2*nvar+1] += nestim*t*(i+1);
    # Now that we have all our matrices, we can form the final matrix.
    augmented_mat = zeros((5*nvar, 5*nvar))
    augmented_mat[:3*nvar, :3*nvar] = Q
    augmented_mat[3*nvar:, :3*nvar] = E
    augmented_mat[:3*nvar, 3*nvar:] = E.T
    # Create the RHS vector
    rhs_vector = vstack((C, d))

    
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
