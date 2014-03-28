#!/usr/bin/env python

from modules.methods import *
from modules.kernel import *
from modules.depth import *

if __name__ == '__main__':
	main_dir = 'output/cam'
	impname = 'preview_im.bmp'
	imbname = 'saved_im.bmp'
	acname = 'saved_ac.dat'

	imp = imread(os.path.join(main_dir, impname), flatten=True)
	imb = imread(os.path.join(main_dir, imbname), flatten=True)

	data = loadtxt(os.path.join(main_dir, acname))
	xshifts, yshifts, z = estimate_simple_pos(data, 10, 60) 

	# Does having no scale really help? Yes it does. The search space is 
	# reduced from being a circle to along a small radius sector. Now in our 
	# initial attempt, we assume that it is only along and line and we estimate
	# the depth using the least difference.

	dist = hypot(xshifts, yshifts)
	idx = where(dist == dist.max())
	dist = dist.max()
	xshift = xshifts[idx]
	yshift = yshifts[idx]

	imdiff_best = float('inf')
	best_shift = [0,0]

	# Assume maximum shift is 50 pixels in any axis => ~70 along diag.
	xdim, ydim = imp.shape
	for depth in linspace(0,200/dist, 50):
		x = int(-xshift*depth); y = int(-yshift*depth)
		print x, y
		imshifted = shift(imp, [x, y])
		if (x >= 0) and (y >= 0):
			imdiff = (abs(imshifted - imb))[x:, y:]/(xdim - x)*(ydim - y)
		elif (x < 0) and (y >= 0):
			imdiff = (abs(imshifted - imb))[:x, y:]/(xdim + x)*(ydim - y)
		elif (x >= 0) and (y < 0):
			imdiff = (abs(imshifted - imb))[x:, :y]/(xdim - x)*(ydim + y)
		else:
			imdiff = (abs(imshifted - imb))[:x, :y]/(xdim + x)*(ydim + y)
		Image.fromarray(imdiff*255.0/imdiff.max()).convert('RGB').save(
			'tmp/register/im_%d_%d.bmp'%(x,y))
		if sqrt(imdiff.sum()) < imdiff_best:
			imdiff_best = sqrt(imdiff.sum())
			best_shift = [x, y]
	imshifted = shift(imp, best_shift)
	imsave = hstack((imb, imshifted))
	Image.fromarray(imsave).convert('RGB').save('tmp/register/imreg.bmp')
	print best_shift


