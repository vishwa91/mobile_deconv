#!/usr/bin/env python

import os, sys
import commands

from matplotlib.pyplot import *
from scipy import *
from scipy.signal import *
from scipy.linalg import *
from scipy.interpolate import spline
from scipy.ndimage import *
from numpy import fft

import Image
from canny import Canny

accel_data_file = '../output/cam/saved_ac.dat'
T = 10e-3
G = 9.8

def _deblur(kernel, im, nsr, mode):
    ''' Deblur a single channel image'''
    x, y= im.shape
    if mode in ['wien', 'reg']:
        kernel /= (1.0*kernel.sum())
        diffx = array([[1,  1],
                       [-1,-1]])
        diffy = array([[-1, 1],
                       [-1, 1]])
        diffxy = array([[0,-1, 0],
                       [-1,4,-1],
                       [0,-1, 0]])
        #x *= 2; y *= 2
        DX = fft.fft2(diffx, s=(x,y))
        DY = fft.fft2(diffy, s=(x,y))
        DXY = fft.fft2(diffxy, s=(x,y))
        REG = abs(DX)**2 + abs(DY)**2 + abs(DXY)**2
        #x2, y2 = kernel.shape
        #x = x1+x2; y = y1+y2
        F = fft.fft2(kernel, s=(x,y))
        IM = fft.fft2(im, s=(x,y))
        if mode == 'reg':
            nsr *= REG
        IMOUT = conj(F)*IM/(abs(F)**2 + nsr)
        imout = real(fft.ifft2(IMOUT))
    elif mode == 'rl':
        flipped_kernel = flipud(fliplr(kernel))
        imout = ones_like(im)
        for i in range(50):
            temp = im/(nsr+fftconvolve(imout, kernel, mode='same'))
            imout = imout*fftconvolve(temp, flipped_kernel, mode='same')
    elif mode =='tik':
        im_nms = Canny(im, 0.2, 50, 20).nms_im[1:-1, 1:-1]
        xt,yt = where(im_nms > 20)
        im_nms[:,:] = 0
        im_nms[xt,yt] = 20
        IM = fft.fft2(im, s=(x,y))
        REG = abs(fft.fft2(im_nms, s=(x,y))/IM)**2
        F = fft.fft2(kernel, s=(x,y))
        IMOUT = conj(F)*IM/(abs(F)**2 + nsr*REG)
        imout = real(fft.ifft2(IMOUT))
    else:
        raise('mode needs to be wien, reg, tik or rl. Type help(deblur) for help')
    #return (imout*255.0/imout.max()).astype(float)
    return imout.astype(float)

def estimate_g(data):
    '''Estimate gravity vector using projection onto unit sphere
       What we need to do is minimize the error given that every point lies
       on a unit sphere.
    '''
    # Assume that the data is a stack of x,y, z acceleration
    norm_vector = sqrt(data[:,0]**2 + data[:,1]**2 + data[:,2]**2)
    data[:,0] = data[:,0]/norm_vector
    data[:,1] = data[:,1]/norm_vector
    data[:,2] = data[:,2]/norm_vector 
    return data
    
def _estimate_coefficients(accel_vector, nestim=5):
    '''Estimate the coefficients of each parabola.'''
    # Model each parabola using nestim samples. If length of accel_vector is
    # not divisible by nestim, discard the tail.
    tail = len(accel_vector)%nestim
    if tail != 0:
        accel_vector = accel_vector[:-tail]
    xlen = len(accel_vector)
    nvar = xlen/nestim
    # our objective function is min. Xt.Q.X + Ct.X where,
    # X = [a1, a2, ...., b1, b2....., c1, c2, .....]
    temp = eye(nvar)     
    Q = zeros((3*nvar, 3*nvar))
    Q[2*nvar:, 2*nvar:] = 4*temp
    C = zeros((3*nvar, 1))
    for i in range(0, xlen, nestim):
        C[2*nvar+i/nestim, 0] = -2*sum(accel_vector[i:i+nestim])
    # E is our constrain matrix. It is a 2nvar x 3nvar matrix.
    Epos = zeros((nvar, 3*nvar))
    Evel = zeros((nvar, 3*nvar))
    # RHS of the constrain.
    d = zeros((2*nvar, 1))
    
    # Create constrain matrix.
    for i in range(0, nvar-1):
        # ai are 0->nvar-1; bi are nvar->2nvar-1, ci are 2nvar->3nvar-1 

        # First nvar equations are position continuity equations and the
        # next nvar equations are velocity continuity equations.

        # Position continuity equation:
        # ai - ai+1 + bi(ti-nestim*t*i) - bi+1(-nestim*t*i+1) +
        # ci(tj-nestim*t*j)^2 - ci+1(-nestim*t*i+1)^2 = 0
        Epos[i, i] += 1; Epos[i, i+1] += -1;
        Epos[i, nvar+i] += (nestim - 1)*T
        Epos[i, 2*nvar+i] += ((nestim-1)*T)**2

        # Velocity continuity equation:
        # bi - bi+1 + ci(ti-nestim*t*i) - ci+1(-nestim*t*i+1) = 0
        Evel[i, i+nvar] += 1; Evel[i, i+nvar+1] -= 1;
        Evel[i, i+2*nvar] += (nestim-1)*T
    # Initial position and velocity are zero.
    Epos[-1, 0] = 1
    Evel[-1, nvar] = 1
    E = vstack((Epos, Evel))
    # Now that we have all our matrices, we can form the final matrix.
    augmented_mat = zeros((5*nvar, 5*nvar))
    augmented_mat[:3*nvar, :3*nvar] = Q
    augmented_mat[3*nvar:, :-2*nvar] = E
    augmented_mat[:-2*nvar, 3*nvar:] = E.T
    # Create the RHS vector
    rhs_vector = vstack((C, d))
    output = dot(inv(augmented_mat), rhs_vector)
    X = output[:3*nvar, 0]
    lamda = output[3*nvar:, 0]
    return X, xlen, nestim

def _estimate_position(accel, nestim):
    '''Estimate the position for a single axis'''
    X, xlen, nestim = _estimate_coefficients(accel, nestim)
    nvars = len(X)/3
    a_coeffs = X[:nvars]
    b_coeffs = X[nvars:2*nvars]
    c_coeffs = X[2*nvars:3*nvars]
    t_vec = arange(0, nestim*T, T)
    pos_vec = zeros(nvars*size(t_vec))
    for i in range(nvars):
        temp_vec = a_coeffs[i] + b_coeffs[i]*t_vec + c_coeffs[i]*t_vec*t_vec
        pos_vec[i*len(t_vec):(i+1)*len(t_vec)] = temp_vec
    return pos_vec
    
def estimate_position(accel, nestim):
    ''' Estimate the trajectory using information from the acceleration values.
        We model chunks of trajectory as a parabola. Using the fact that the 
        position values and velocity values are continuous, we minimize the 
        error between modeled acceleration and actual acceleration.
    '''
    # Estimate g first
    g_vector = estimate_g(accel.copy())
    accel -= g_vector
    # Convert acceleration into m/s^2
    accel *= G
    xaccel = accel[:,0]; yaccel = accel[:,1]; zaccel = accel[:,2]
    xpos = _estimate_position(xaccel, nestim)
    ypos = _estimate_position(yaccel, nestim)
    zpos = _estimate_position(zaccel, nestim)
    raw_xpos = cumsum(cumsum(xaccel))*T*T
    raw_ypos = cumsum(cumsum(yaccel))*T*T
    raw_zpos = cumsum(cumsum(zaccel))*T*T
    return xpos, ypos, zpos, raw_xpos, raw_ypos, raw_zpos

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
        kernel[int(xpos[i]), int(ypos[i])] += 1
    return kernel

if __name__ == '__main__':
    try:
        os.mkdir('../tmp/kernel')
    except OSError:
        pass
    data = loadtxt(accel_data_file)
    temp = array([0.1, 0.4, -0.1, 0.5, 0.1])
    im = imread('../tmp/cam/imtest.bmp', flatten=True)
    x, y, z, xr, yr, zr = estimate_position(data.copy(), 5)
    subplot(3,1,1); plot(xr); plot(x)
    subplot(3,1,2); plot(yr); plot(y)
    subplot(3,1,3); plot(zr); plot(z)
    start = 41
    end = 63
    x_imp = x[start:end]; y_imp = y[start:end]
    #show()
    drange = arange(100**2, 6000**2, 300**2)
    for depth in sqrt(drange).astype(int):
        print 'Deconvolving for %d depth'%depth
        kernel = construct_kernel(x_imp, y_imp, depth, 10)
        kernel = kernel.astype(float)/kernel.sum()
        #Image.fromarray(flipud(kernel)*255.0/kernel.sum()).convert('RGB').save(
        #    '../tmp/kernel/kernel_%d.bmp'%depth)
        imout = _deblur(flipud(kernel), im, 0.001, mode='reg')
        #out = commands.getoutput('../output/cam/robust_deconv.exe ../tmp/cam/imtest.bmp ../tmp/kernel/kernel_%d.bmp ../tmp/kernel/im_%d.bmp 0 0.1 1'%(depth, depth))
        #print out
        # (fftconvolve(imout, flipud(kernel), mode='same')-im
        Image.fromarray(imout).convert(
            'RGB').save('../tmp/kernel/im_%d.bmp'%depth)
