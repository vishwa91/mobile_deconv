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
	#xshift, yshift, z = estimate_simple_pos(data, 10, 110) # Shifts
	xblur, yblur, z = estimate_simple_pos(data, 10, 110) # Blur

	#dmax_shift = hypot(xshift, yshift).max()
	dmax_blur = hypot(xblur, -yblur).max()

	#shift_kernel = construct_kernel(xshift, yshift, 300/dmax_shift, 10)
	blur_kernel = construct_kernel(xblur, yblur, 100/dmax_blur, 10)

	imblur = linear_convolve(imp, blur_kernel)
	Image.fromarray(imp).show()
	Image.fromarray(imb).show()
	#Image.fromarray(shift_kernel*255.0/shift_kernel.max()).show()
	Image.fromarray(blur_kernel*255.0/blur_kernel.max()).show()