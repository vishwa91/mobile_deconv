
#!/usr/bin/env python
import os
import sys

from scipy import *
from scipy.ndimage import *
from scipy.signal import *
from numpy import fft
import matplotlib.pyplot as plt
import Image

# global constants
IMTEST = 'imblur.bmp'
KERNEL = 'blur_kernel.bmp'
TMPDIR = 'tmp'

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

if __name__ == '__main__':
	# Create the temporary directory if necessary
	try:
		os.mkdir(TMPDIR)
	except OSError:
		print 'Temporary directory exists.'
		pass
	# Load image and kernel.
	kernel = imread(KERNEL, flatten=True)
	im = imread(IMTEST, flatten=True)
	nsr = 0.001
	count = 0
	for scale in arange(0.1, 2, 0.1):
		# Deblur at various scales.
		print 'Deblurring at %f scale'%scale
		imout = _deblur(zoom(kernel, scale), im, nsr)
		imname = '%s/im_%d.bmp'%(TMPDIR, int(scale*10))
		Image.fromarray(imout.astype(uint8)).save(imname)