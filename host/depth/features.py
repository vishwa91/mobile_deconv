#!/usr/bin/env python

from scipy import *
from scipy.ndimage import *
from scipy.signal import *

from numpy import fft

import Image
import lbp

if __name__=='__main__':
	im1 = imread('../tmp/space_variant_blur.bmp', flatten=True)
	im2 = imread('../synthetic/random_dot.jpg', flatten=True)
	imlbp1 = lbp.lbp(im1.astype(int)); imlbp2 = lbp.lbp(im2.astype(int))
	klaplace = array([[ 0,-1, 0],
                      [-1, 4,-1],
                      [ 0,-1, 0]])/4.0
	imdiff = convolve2d(im1, klaplace)[1:-1, 1:-1]
	Image.fromarray(imlbp1).show()
	Image.fromarray(imdiff*10).show()
