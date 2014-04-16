#!/usr/bin/env python

'''
    Routines in this file are related to estimating depth from an image, given
    a blurred and non-blurred image pair and the shifts from accelerometer data.
'''
import os, sys
import commands

from matplotlib.pyplot import *
from scipy import *
from scipy.signal import *
from scipy.linalg import *
from scipy.interpolate import spline
from scipy.ndimage import *
from scipy.special import *
from numpy import fft

import Image
from kernel import *
from methods import *

def norm_diff(im1, im2):
    ''' Difference of two images after normalization'''
    m1 = mean(im1); m2 = mean(im2)
    v1 = variance(im1); v2 = variance(im2)
    imnorm1 = (im1-m1)/sqrt(v1); imnorm2 = (im2-m2)/sqrt(v2)
    diff_im = (imnorm1 - imnorm2)**2
    return diff_im
    
def patchy_depth(impure, imblur, xpos, ypos, w=31):
    ''' As Prof. ANR suggested, let us check at patches and take that patches
        which has the least error energy
    '''
    dmax = hypot(xpos, ypos).max()
    imdepth = zeros_like(impure)
    xdim, ydim = impure.shape
    nlevels = 40
    d_array = linspace(0, nlevels/dmax, nlevels)
    dstack = []
    # Create blur copies
    for depth in d_array:
        print 'Creating new blur image'
        kernel = construct_kernel(xpos, ypos, depth, 10)
        imreblur = fftconvolve(impure, kernel, mode='same')
        #imreblur = register(imreblur, imblur)
        imdiff = (imreblur - imblur)**2
        dstack.append(imdiff)
    imbest = zeros_like(impure); imbest[:,:] = float('inf')
    for count in range(len(d_array)):
        print 'Estimating new depth'
        for x in range(w, xdim, w):
            for y in range(w, ydim, w):
                if imbest[x:x+w, y:y+w].sum() > dstack[count][x:x+w, y:y+w].sum():
                    imdepth[x:x+w, y:y+w] = count
                    imbest[x:x+w, y:y+w] = dstack[count][x:x+w, y:y+w]
    return imdepth
    
def iterative_depth(impure, imblur, xpos, ypos, mkernel=None):
    ''' Estimate the depth using multiple iterations. Rudimentary, but expected
        to work.
    '''
    w = 15
    avg_filter = ones((w,w))/(w*w*1.0)
    xdim, ydim = impure.shape
    imdepth = zeros((xdim, ydim))
    imdiff = zeros((xdim, ydim)); imdiff[:,:] = float('inf')
    imdiff_curr = zeros((xdim, ydim))
    xw = 32; yw = 32
    dmax = hypot(xpos, ypos).max()
    count = 0
    diff_array1 = []; diff_array2 = []
    nlevels = 40
    save_data = zeros((xdim, ydim, nlevels))
    for depth in linspace(0, nlevels/dmax, nlevels):
        print 'Iteration for %f depth'%depth
        if mkernel == None:
            kernel = construct_kernel(xpos, ypos, depth)
        else:
            kernel = zoom(mkernel, depth)
            kernel /= (1e-5+kernel.sum())
        imreblur = linear_convolve(impure, kernel)
        #imreblur = register(imreblur, imblur, kernel)
        # imsave is a 2d image
        imsave = norm_diff(imreblur, imblur)
        imdiff_curr = sqrt(linear_convolve(imsave, avg_filter))
        #imdiff_curr = gaussian_filter(imsave, 3.1)
        #save_data[:,:,count] = imdiff_curr
        imtemp = zeros((xdim, ydim*2), dtype=uint8)
        imtemp[:, :ydim] = imblur
        imtemp[:, ydim:] = imreblur#imsave*255.0/imsave.max()
        Image.fromarray(imtemp).convert('RGB').save(
            'tmp/depth/im%d.bmp'%count)
        x, y = where(imdiff_curr <= imdiff)
        imdepth[x, y] = depth
        imdiff[x, y] = imdiff_curr[x, y]
        count += 1
    return imdepth, save_data

def estimate_planar_depth(impure, imblur, xpos, ypos, mkernel=None):
    ''' Estimate the depth using multiple iterations. Rudimentary, but expected
        to work.
    '''
    best_depth = float('inf')
    w = 15
    avg_filter = ones((w,w))/(w*w*1.0)
    xdim, ydim = impure.shape
    xw = 32; yw = 32
    dmax = hypot(xpos, ypos).max()
    count = 0
    nlevels = 40
    curr_diff = float('inf')
    for depth in linspace(0, nlevels/dmax, nlevels):
        print 'Iteration for %f depth'%depth
        if mkernel == None:
            kernel = construct_kernel(xpos, ypos, depth)
        else:
            kernel = zoom(mkernel, depth)
            kernel /= (1e-5+kernel.sum())
        imreblur = fftconvolve(impure, kernel, mode='same')
        imreblur = register(imreblur, imblur, kernel)
        imsave = norm_diff(imreblur, imblur).sum()
        if imsave < curr_diff:
        	best_depth = depth
        	curr_diff = imsave
        count += 1
    return best_depth

def bp_depth(impure, imblur, x, y, scale = 0.4):
    ''' Estimate depth using belief propagation. The executable has been 
        provided by Karthik and is based on Prof. ANR's paper on
        Depth from motion blur using Unscented Kalman Filter (name not accurate)
    '''
    # Large images give a memory out error
    im1 = zoom(impure, scale)
    im2 = zoom(imblur, scale)

    # save the data
    savetxt('tmp/bf_depth/tx.txt', x)
    savetxt('tmp/bf_depth/ty.txt', y)

    Image.fromarray(im1).convert('L').save('tmp/bf_depth/im1.pgm')
    Image.fromarray(im2).convert('L').save('tmp/bf_depth/im2.pgm')

    # Execute the commands
    os.system('tools/depth_ukf/depth tmp/bf_depth/tx.txt tmp/bf_depth/ty.txt\
        tmp/bf_depth/im1.pgm tmp/bf_depth/im2.pgm tmp/bf_depth/depth.pgm')

def sml_focus_depth(imdir, smldir, idx1, idx2):
    ''' Calculate the depth of a scene using Sum Modified Laplacian operator,
        given a set of images, which are refocused in every image
    '''
    imdepth = zeros((480, 640))
    imdepth[:,:] = float('inf')
    im_max = zeros_like(imdepth)
    imfocus = zeros_like(imdepth)
    window = 9
    kx = array([[0,1,0],
                [0,-2,0],
                [0,1,0]])
    ky = array([[0,0,0],
                [1,-2,1],
                [0,0,0]])
    scale_array = linspace(1.0, 1.05, idx2+1)[:-1]
    for i in range(idx1, idx2):
        print 'Reading image %d'%i
        im = imread(os.path.join(imdir, 'im%d.pgm'%i), flatten=True)
        x1, y1 = im.shape; tx = x1//2; ty = y1//2
        imzoom = zoom(im, scale_array[i])
        x2, y2 = imzoom.shape; cx = x2//2; cy = y2//2
        im = imzoom[cx-tx:cx+tx, cy-ty:cy+ty]
        gxx = linear_convolve(im, kx)
        gyy = linear_convolve(im, ky)
        imlap = abs(gxx) + abs(gyy)
        mfilter = ones((window, window), dtype=float)/float(window*window)
        #imlap = linear_convolve(imlap, mfilter)
        imlap = gaussian_filter(imlap, 3.0)
        x, y = where(imlap > im_max)
        im_max[x, y] = imlap[x, y]
        imdepth[x, y] = i
        imfocus[x,y] = im[x,y]
        Image.fromarray((imlap*255.0/imlap.max()).astype(uint8)).save(
            os.path.join(smldir, 'im%d.pgm'%i))
    return imdepth*255.0/imdepth.max(), imfocus

def _calc_local_var(im, window):
    ''' Calculate the local variance of an image'''
    xdim, ydim = im.shape
    tx = window//2
    imvar = zeros_like(im)
    for i in range(tx,xdim-tx):
        for j in range(tx,ydim-tx):
            imvar[i, j] = variance(im[i-tx:i+tx,i-tx:i+tx])
    return imvar

def var_focus_depth(imdir, smldir, idx1, idx2, window=8):
    ''' Calculate the depth of a scene using local variance,
        given a set of images, which are refocused in every image
    '''
    imdepth = zeros((480, 640))
    imdepth[:,:] = float('inf')
    im_max = zeros_like(imdepth)
    imfocus = zeros_like(imdepth)
    scale_array = linspace(1.0, 1.05, idx2+1)[:-1]
    for i in range(idx1, idx2):
        print 'Reading image %d'%i
        im = imread(os.path.join(imdir, 'im%d.pgm'%i), flatten=True)
        x1, y1 = im.shape; tx = x1//2; ty = y1//2
        imzoom = zoom(im, scale_array[i])
        x2, y2 = imzoom.shape; cx = x2//2; cy = y2//2
        im = imzoom[cx-tx:cx+tx, cy-ty:cy+ty]
        imlap = _calc_local_var(im, window)
        mfilter = ones((window, window), dtype=float)/float(window*window)
        imlap = linear_convolve(imlap, mfilter)
        #imlap = gaussian_filter(imlap, 3.0)
        x, y = where(imlap > im_max)
        im_max[x, y] = imlap[x, y]
        imdepth[x, y] = i
        imfocus[x,y] = im[x,y]
        Image.fromarray((imlap*255.0/imlap.max()).astype(uint8)).save(
            os.path.join(smldir, 'im%d.pgm'%i))
    return imdepth*255.0/imdepth.max(), imfocus