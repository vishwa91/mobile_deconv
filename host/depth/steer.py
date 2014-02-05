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

if __name__ == '__main__':
	try:
		os.mkdir('../tmp/steer')
	except OSError:
		pass
	impure = imread('../output/cam/saved_im_noshake.bmp', flatten=True)
	# Load the acceleration data.
	data = loadtxt(accel_data_file)
	start = 41
 	end = 63
	x, y, z, g = estimate_simple_pos(data, start, end)
	kernel = construct_kernel(x, y, 4000, 10)
	kernel = kernel.astype('float')/(1.0*kernel.sum())
	# Blur the image
	imblur = convolve2d(impure, kernel)
	imblur = imread('../output/cam/saved_im.bmp', flatten=True)
	#Create the basis kernels for a steerable filter.
	sobelx = array([[ 1, 2, 1],
					[ 0, 0, 0],
					[-1,-2,-1]])
	sobely = array([[-1, 0, 1],
					[-2, 0, 2],
					[-1, 0, 1]])
	count = 0
	# Save the blur image also
	Image.fromarray(imblur).convert('RGB').save('../tmp/steer/imblur.bmp')
	dx = x[1:] - x[:-1]
	dy = y[1:] - y[:-1]
	temp = convolve2d(imblur, sobelx)
	imfinal = zeros_like(imblur)
	for i in range(len(dx)):
		cosx = dy[i]/hypot(dx[i],dy[i])
		sinx = dx[i]/hypot(dx[i],dy[i])
		diff_kernel = sobelx*cosx + sobely*sinx
		imdiff = convolve2d(imblur, diff_kernel)[1:-1, 1:-1]
		imfinal += imdiff
		#Image.fromarray(imdiff).convert('RGB').save(
		#	'../tmp/steer/im_%d.bmp'%count)
		count += 1
	#imfinal  = (imfinal - imfinal.min())*255.0/imfinal.max()
	imfinal *= 255.0/imfinal.max()

	for depth in range(100, 7000, 100):
		print 'Deconvolving for %d depth'%depth
		kernel = construct_kernel(x, y, depth, 10)
		imdeblur = _deblur(kernel, imfinal, 0.001, 0)
		imreblur = fftconvolve(imdeblur, kernel, mode='same')
		Image.fromarray(imfinal-imreblur).convert('RGB').save(
			'../tmp/steer/im%d.bmp'%depth)

	print imfinal.shape
	Image.fromarray(imfinal).convert('RGB').save('../tmp/steer/imfinal.bmp')
