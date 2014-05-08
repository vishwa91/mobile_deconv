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
	ypos, xpos, zpos = estimate_simple_pos(data, 10, 110)

	dmax = hypot(xpos, ypos).max()

	xshifts = linspace(0, abs(xpos).max()*0.5, 4)
	yshifts = linspace(0, abs(ypos).max()*0.5, 4)

	depths  = linspace(0, 20/dmax, 10)
	count = 0
	imvarmin = float('inf')
	best_feat = None
	for xshift in xshifts:
		for yshift in yshifts:
			for depth in depths:
				print xshift, yshift, depth
				xtemp = xpos - linspace(0, xshift, len(xpos))
				ytemp = ypos - linspace(0, yshift, len(xpos))

				kernel = construct_kernel(xtemp, ytemp, depth, 10)

				#imout = real(wiener_deconv(kernel, im, alpha=0.005))
				#Image.fromarray(imout.astype(uint8)).convert('RGB').save(
				#	'tmp/deconv/im%d.bmp'%count)
				cmd_op = non_blind_deconv(kernel, im, 
				                'tmp/deconv/im%d.bmp'%count)
				print cmd_op
				#imx = prewitt(imout, -1)
				#imy = prewitt(imout, 0)
				#imvar = hypot(imx, imy).sum()
				#print imvar
				#if imvar < imvarmin:
				#	imvarmin = imvar
				#	best_feat = [xshift, yshift, depth, count]
				count += 1
	#print best_feat