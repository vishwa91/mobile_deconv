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

accel_data_file = '../output/cam/saved_ac.dat'
T = 10e-3
G = 9.8

def register(impure, imblur):
    ''' Register and shift the pure image using fourier correlation method.'''
    IMPURE = fft.fft2(impure)
    IMBLUR = fft.fft2(imblur)

    IMSHIFT = IMPURE*conj(IMBLUR)/(abs(IMPURE)*abs(IMBLUR))

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
    shifted_im = fftconvolve(shift_kernel, impure, mode='same')

    return shifted_im

def _try_deblur(kernel, im, nsr, mfilter):
    ''' Another try at deblurring'''
    kernel = kernel.astype(float)/kernel.sum()
    x, y = im.shape
    IM = fft.fft2(im, s=(x,y))
    H  = fft.fft2(kernel, s=(x,y))
    F  = fft.fft2(mfilter, s=(x,y))/IM
    
    IMOUT = conj(H)*IM/(abs(H)**2 + nsr*(abs(F)**2))
    imout = real(fft.ifft2(IMOUT))
    return imout.astype(float)
    
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
    #x, y = im.shape
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

def compute_diff(impure, imblur, kernel, window=5, zero_thres=10):
    ''' Compute the difference of the two images. The difference will also be
        averaged to reduce the effect of noise.'''
    imreblur = fftconvolve(impure, kernel, mode='same')
    imdiff = abs(imblur - imreblur)
    avg_kernel = ones((window, window), dtype=float)/float(window**2)
    x, y = imdiff.shape
    startx = max(0, window//2-1); starty = max(0, window//2-1)
    endx = x + startx; endy = y + starty
    imavg = fftconvolve(imdiff, avg_kernel, mode='same')#[startx:endx, starty:endy]
    xz, yz = where(imavg <= zero_thres)
    #imavg[xz, yz] = 0

    return imavg, xz, yz

def computer_path_diff(im, x, y):
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
        cosx = dx[i]/hypot(dx[i],dy[i])
        sinx = dy[i]/hypot(dx[i],dy[i])
        diff_kernel = sobelx*cosx + sobely*sinx
        imdiff = convolve2d(impure, diff_kernel)[1:-1, 1:-1]
        xmin, ymin = where(imdiff <= imfinal)
        imfinal[xmin, ymin] = imdiff[xmin, ymin]
    imfinal *= 255.0/imfinal.max()
    return imfinal

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
    impure = register(impure, imblur)
    # Save the blur image also
    Image.fromarray(imblur).convert('RGB').save('../tmp/steer/imblur.bmp')
    Image.fromarray(impure).convert('RGB').save('../tmp/steer/imbure.bmp')
    imdepth = zeros_like(imblur)
    imdepth[:,:] = float('inf')
    old_diff = zeros_like(imblur)
    old_diff[:,:] = float('inf')
    x1, y1 = imblur.shape
    for depth in range(10, 7000, 10):
        print 'Deconvolving for %d depth'%depth
        kernel = construct_kernel(x, y, depth, 1)
        imdiff, xz, yz = compute_diff(impure, imblur, kernel, 15, 20)
        xd, yd = where(imdiff < old_diff)
        imdepth[xd,yd] = depth
        #imdepth[xz, yz] = 0
        old_diff[xd,yd] = imdiff[xd,yd]
        Image.fromarray(imdiff*255.0/imdiff.max()).convert('RGB').save(
            '../tmp/steer/im%d.bmp'%depth)
    imdepth *= 255.0/imdepth.max()
    Image.fromarray(imdepth).convert('RGB').save('../tmp/steer/imdepth.bmp')
