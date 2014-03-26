#!/usr/bin/env python

from modules.methods import *
from modules.kernel import *

def _equalize(data_blind, data_accel):
	''' Find the scale and shift which would reduce the error between the 
	two data sets'''

	if len(data_blind) > len(data_accel):
		data_accel = spline(range(len(data_accel)), data_accel,
        linspace(0, , range(len(data_blind)))
    else:
    	data_blind = spline(range(len(data_blind)), data_blind,
        linspace(0, , range(len(data_accel)))

    N = len(data_accel)
    # Form the constants
    a1 = (data_accel*data_accel).sum()
    b1 = (-array(range(len(data_blind)))*data_accel).sum()
    c1 = (data_blind*data_accel).sum()

    a2 = b1
    b2 = N*(N+1)*(2*N+1)/6
    b1 = (array(range(len(data_blind)))*data_blind).sum()

    k2 = (a1*c2 - b1*c1)/(b1*b1 - a1*b2)
    k1 = (b1*c2 - b2*c1)/(b2*a1 - b1*b1)

    lshift = arange(0, N, 1.0)

    data_return = k1*data_accel - k2*shift

    return data_return

def equalize(blind_kernel, xpos, ypos):
	'''Find the scale and linear shift that would reduce the error between the
	   blind kernel and the measured kernel'''

	xhist_blind = sum(blind_kernel, axis=0)
	yhist_blind = sum(blind_kernel, axis=1)

	xdim, ydim = blind_kernel.shape

	xpos *= 0.5*xdim/abs(xpos).max()
	ypos *= 0.5*ydim/abs(ypos).max()

	mkernel = construct_kernel(xpos, ypos, 1.0, 10.0)

	xhist_accel = sum(mkernel, axis=0)
	yhist_accel = sum(mkernel, axis=1)

	xhist_new = _equalize(xhist_blind, xhist_accel)
	yhist_new = _equalize(yhist_blind, yhist_accel)

	


