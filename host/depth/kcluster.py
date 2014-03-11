#!/usr/bin/env python

import os, sys

from scipy import *
from scipy.ndimage import *
from scipy.signal import *
from numpy import fft
from scipy.interpolate import spline
from scipy.cluster.vq import kmeans, vq
import Image

from rgb2lab import *

accel_data_file = '../output/cam/saved_ac.dat'
T = 10e-3
G = 9.8

def do_clustering(im, ncenters=3):
    ''' Segment the image using kmeans clustering. The image is expected
        to be in Lab format
        '''
    xdim, ydim, nchannels = im.shape
    im_vec = im.copy().reshape((xdim*ydim, nchannels))
    centroids, _ = kmeans(im_vec, ncenters)
    idx, _ = vq(im_vec, centroids)
    for i in range(ncenters):
        x = where(idx == i)
        im_vec[x] = i
    imout = im_vec.reshape((xdim, ydim, nchannels))

    return imout*255.0/ncenters

if __name__ == '__main__':
    im = imread('../test_output/real8/saved_im.bmp')
    xdim, ydim, _ = im.shape
    x = range(xdim); y = range(ydim)
    X, Y = meshgrid(x,y)
    X -= xdim//2; Y -= ydim//2
    dist_mat = 0.1*hypot(X,Y).T
    im_to_cluster = zeros((xdim, ydim, 4))
    imsmooth = gaussian_filter(im, 1.1)
    im_to_cluster[:,:,:3] = imsmooth
    im_to_cluster[:,:,3] = dist_mat
    imlab = rgb2lab(im)
    imseg = do_clustering(im_to_cluster)[:,:,0]
    imseg = filters.median_filter(imseg, (11,11))
    Image.fromarray(imseg).show()
                           

