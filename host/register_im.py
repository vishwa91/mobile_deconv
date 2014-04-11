#!/usr/bin/env python

from modules.methods import *
from modules.kernel import *
from modules.depth import *

def shift_register(xshift, yshift, imp, imb):
	'''Register two images which are just shifted, given the position vector,
		the latent image and the blurred image
	'''
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
	return best_shift

def rotate_register(ax, ay, imp, imb):
	''' Regiter two images which have been rotated and translated. In the first
		step, to remove rotation, we use the data from the accelerometers, which
		could give a clue about the orientation. Once this is done, we use
		fourier correlation to retrieve the translation.
	'''

	best_shift = [0,0]
	best_rotation = 0
	best_diff = 0#float('inf')
	IMB = fft.fft2(imb)
	# The motion is expected between 15 and 35. We don't exactly know when it 
	# happens since the camera and the accelerometer are not synchronized. 
	# Hence, we sweep.
	theta = arctan2(ay, ax)
	for i in range(10,36):
		imrotated = rotate(imp, -180 + theta[i]*180/pi, reshape = False)
		IMR = fft.fft2(imrotated)
		IMCORR = IMR*conj(IMB)/abs(IMR*IMB)
		imcorr = fft.ifft2(IMCORR)
		x, y = where(imcorr == imcorr.max())
		corr_max = imcorr.max()
		xdim, ydim = imcorr.shape
		if x >= xdim//2:
			x = x - xdim
		if y >= ydim//2:
			y = y - ydim
		imcorrected = shift(imrotated, [x, y])
		Image.fromarray(abs(imcorrected - imb)).convert('RGB').save(
			'tmp/register/im%d.bmp'%i)
		imdiff = sqrt(pow((imcorrected-imb), 2).sum())
		if corr_max > best_diff:
			best_diff = corr_max
			best_rotation = theta[i]
			best_shift = [x, y]
	return best_rotation, best_shift

if __name__ == '__main__':
	main_dir = 'output/cam'
	impname = 'preview_im.bmp'
	imbname = 'saved_im.bmp'
	acname = 'saved_ac.dat'

	imp = imread(os.path.join(main_dir, impname), flatten=True)
	imb = imread(os.path.join(main_dir, imbname), flatten=True)

	data = loadtxt(os.path.join(main_dir, acname))
	'''
	xshifts, yshifts, z = estimate_simple_pos(data, 10, 60) 

	best_shift = shift_register(xshifts, yshifts, imp, imb)

	imshifted = shift(imp, best_shift)
	imsave = hstack((imb, imshifted))
	Image.fromarray(imsave).convert('RGB').save('tmp/register/imreg.bmp')
	'''
	ax = data[:,0]; ay = data[:,1]; az = data[:,2] 
	gx = data[:,3]; gy = data[:,4]; gz = data[:,5]

	figure(1)
	subplot(3,1,1); plot(gx)
	subplot(3,1,2); plot(gy)
	subplot(3,1,3); plot(gz)

	figure(2)
	subplot(3,1,1); plot(ax - gx)
	subplot(3,1,2); plot(ay - gy)
	subplot(3,1,3); plot(az - gz)

	# 35-55 for gx, gy; 15-35 for ax, ay
	figure(3)
	thetas = unwrap(arctan2(gy, gx))
	plot(thetas)
	
	figure(4)
	vx = cumsum(ax-gx)
	vy = cumsum(ay-gy)
	jerk_theta = unwrap(arctan2(vy, vx))
	plot(jerk_theta)

	#show()

	theta, mshift = rotate_register(ax, ay, imp, imb)
	print theta, mshift
	imrotated = rotate(imp, -180 + 180*theta/pi, reshape=False)
	imshift = shift(imrotated, mshift)
	imsave = hstack((imb, imshift))
	Image.fromarray(imsave).convert('RGB').save('tmp/register/imreg.bmp')