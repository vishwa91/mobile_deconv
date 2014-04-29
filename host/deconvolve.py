#!/usr/bin/env python

from modules.deconv import *
from modules.kernel import *
from modules.methods import *
from modules.depth import *

T = 10e-3
G = 9.8

if __name__ == '__main__':
	main_dir = 'output/cam'
	im_name = 'saved_im.bmp'
	ac_name = 'saved_ac.dat'

	im = imread(os.path.join(main_dir, im_name), flatten=True)
	data = loadtxt(os.path.join(main_dir, ac_name))
	xpos, ypos, zpos = estimate_simple_pos(data, 5, 27)

	dmax = hypot(xpos, ypos).max()

	xshifts = linspace(0, abs(xpos).max(), 5)
	yshifts = linspace(0, abs(ypos).max(), 5)

	depths  = linspace(0, 10/dmax, 10)
	count = 0
	for xshift in xshifts:
		for yshift in yshifts:
			for depth in depths:
				print xshift, yshift, depth
				xtemp = xpos - linspace(0, xshift, len(xpos))
				ytemp = ypos - linspace(0, yshift, len(xpos))

				kernel = construct_kernel(xtemp, ytemp, depth, 10)

				#imout = wiener_deconv(kernel, im, alpha=0.0005)
				#Image.fromarray(imout.astype(uint8)).convert('RGB').save(
				#	'tmp/deconv/im%d.bmp'%count)
				cmd_op = non_blind_deconv(kernel, im, 
				                'tmp/deconv/im%d.bmp'%count)
				print cmd_op
				count += 1