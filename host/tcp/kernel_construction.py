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

def _deblur(kernel, im, nsr, niters=4):
    ''' Deblur a single channel image'''
    x1, y1= im.shape
    x2, y2 = kernel.shape
    # Pad the image. We are getting unnecessary shift otherwise.
    x = x1+2*x2; y=y1+2*y2
    imtemp = zeros((x, y), dtype='uint8')
    imtemp[x2:-x2, y2:-y2] = im # Main image
    imtemp[:x2, :y2] = im[-x2:,-y2:]  # Left top
    imtemp[-x2:, :y2] = im[:x2, -y2:] # Right top
    imtemp[-x2:, -y2:] = im[:x2, :y2] # Right bottom
    imtemp[:x2, -y2:] = im[-x2:, :y2] # Left bottom
    imtemp[x2:-x2, :y2] = im[:, -y2:] # top
    imtemp[x2:-x2, -y2:] = im[:, :y2] # bottom
    imtemp[:x2, y2:-y2] = im[-x2:, :] # left
    imtemp[-x2:, y2:-y2] = im[:x2, :] # right
    im = imtemp

    # Create the ffts
    IM = fft.fft2(im, s=(x,y))
    H  = fft.fft2(kernel, s=(x,y))
    # First time transformation is just wiener.
    IMOUT = conj(H)*IM/(abs(H)**2+nsr)

    # Now we do reguralization.
    for i in range(niters):
        IMDIFF = (IM - H*IMOUT)/IM
        IMOUT = conj(H)*IM/(abs(H)**2+nsr*IMDIFF)
    imout = fft.ifft2(IMOUT)
    return imout.astype(float)[:x1, :y1]

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
    return xpos, ypos, zpos, raw_xpos, raw_ypos, raw_zpos, g_vector

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
    return kernel

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

def compute_var(im, window=5):
    ''' Compute the variance map of the image. window should be odd'''
    if window%2 != 1:
        raise("window value should be odd")
    imvar = zeros_like(im)
    xdim, ydim = imvar.shape
    mshift = window//2
    for x in range(mshift, xdim-mshift):
        for y in range(mshift, ydim-mshift):
            imvar[x,y] = var(im[x-mshift:x+mshift, y-mshift:y+mshift])
    return imvar*255.0/imvar.max()

if __name__ == '__main__':
    try:
        os.mkdir('../tmp/kernel')
    except OSError:
        pass
    data = loadtxt(accel_data_file)
    temp = array([0.1, 0.4, -0.1, 0.5, 0.1])
    im = imread('../tmp/cam/imtest.bmp', flatten=True)
    start = 41
    end = 63
    #x, y, z, xr, yr, zr, g = estimate_position(data.copy(), 5)
    #x_imp = x[start:end]; y_imp = y[start:end]
    x, y, z, g = estimate_simple_pos(data, start, end)
    subplot(3,1,1); plot(x); plot(data[start:end,0]-g[start:end, 0])
    subplot(3,1,2); plot(y); plot(data[start:end,1]-g[start:end, 1])
    subplot(3,1,3); plot(z); plot(data[start:end,2]-g[start:end, 2])
    show()
    drange = arange(100**2, 1000**2, 200**2)
    imdepth = ones_like(im)
    imdepth[:,:] = float('inf')
    var_old = zeros_like(im)
    var_old[:,:] = float('inf')
    for depth in range(100, 800, 10):
        print 'Deconvolving for %d depth'%depth
        kernel = construct_kernel(x, y, depth, 10)
        kernel = kernel.astype(float)/kernel.sum()
        Image.fromarray(kernel*255.0/kernel.max()).convert('RGB').save(
            '../tmp/kernel/kernel_%d.bmp'%depth)
        imout = _deblur(kernel, im, 0.001, niters=3)
        #out = commands.getoutput('../output/cam/robust_deconv.exe ../tmp/cam/imtest.bmp ../tmp/kernel/kernel_%d.bmp ../tmp/kernel/im_%d.bmp 0 0.1 1'%(depth, depth))
        #print out
        # fftconvolve(imout, kernel, mode='same')-im
        # Construct the variance map.
        imdiff = im - fftconvolve(imout, kernel, mode='same')
        var_image = compute_var(imdiff, 15)
        xd, yd = where(var_image < var_old)
        var_old[xd, yd] = var_image[xd, yd]
        imdepth[xd, yd] = depth
        Image.fromarray(var_image).convert('RGB').save(
            '../tmp/kernel/im_%d.bmp'%depth)
    Image.fromarray(imdepth*255.0/imdepth.max()).convert('RGB').save(
        'depth_map.bmp')
