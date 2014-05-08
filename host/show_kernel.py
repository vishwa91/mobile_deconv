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

	xblur, yblur, z = estimate_simple_pos(data, 5, 60) # Blur

	dmax_blur = hypot(xblur, yblur).max()

	if True:
		count = 0
		xshifts = linspace(0, max(abs(xblur)), 10)
		yshifts = linspace(0, max(abs(yblur)), 10)
		for xshift in xshifts:
			for yshift in yshifts:
				x = xblur - linspace(0, sqrt(xshift), len(xblur))**2
				y = yblur - linspace(0, sqrt(yshift), len(yblur))**2
				kernel = construct_kernel(x, y, 200/dmax_blur, 10)
				Image.fromarray(kernel*255.0/kernel.max()).convert('RGB').save(
					'tmp/kernel/im_%f_%f.png'%(xshift, yshift))
				count += 1
	xblur -= mean(xblur); yblur -= mean(yblur)
	blur_kernel = construct_kernel(xblur, yblur, 100/dmax_blur, 10)

	#Image.fromarray(imp).show()
	Image.fromarray(imb).show()
	#Image.fromarray(shift_kernel*255.0/shift_kernel.max()).show()
	Image.fromarray(blur_kernel*255.0/blur_kernel.max()).show()

	# Save the data
	Image.fromarray(imb).convert('RGB').save('tmp/ground_truth.png')
	Image.fromarray(blur_kernel*255.0/blur_kernel.max()).convert('RGB').save(
		'tmp/constructed.png')
