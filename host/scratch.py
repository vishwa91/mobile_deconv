#!/usr/bin/env python

from modules.methods import *
from modules.tcp import *
from modules.kernel import *

if __name__ == '__main__':
	start_token = 'S\x00T\x00F\x00S\x00'
	end_token = 'E\x00D\x00F\x00S\x00'
	frame_token = 'S\x00T\x00F\x00R\x00'
	fstart = 'STFR'
	fend = 'EDFR'
	#dummy_recv(start_token, end_token, frame_token, 'tmp/focus/tokens.dat')
	#extract_images('tmp/focus/tokens.dat', 100, fstart, fend, 'tmp/focus/src')

	imdepth = zeros((480, 640))
	imdepth[:,:] = float('inf')
	im_max = zeros_like(imdepth)
	window = 15
	for i in range(3, 100):
		print 'Reading image %d'%i
		im = imread('tmp/focus/src/im%d.bmp'%i, flatten=True)
		imlap = abs(laplace(im))
		mfilter = ones((window, window), dtype=float)/(8.0*8.0)
		imlap = linear_convolve(imlap, mfilter)
		x, y = where(imlap > im_max)
		im_max[x, y] = imlap[x, y]
		imdepth[x, y] = i
		Image.fromarray((imlap*255.0/imlap.max()).astype(uint8)).save(
			'tmp/focus/sml/im%d.bmp'%i)
	Image.fromarray(imdepth).convert('RGB').save('tmp/focus/imdepth.bmp')
