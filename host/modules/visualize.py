#!/usr/bin/env python

'''
    Routines in this file are used for visualizing data.
'''
import os, sys
import commands

from matplotlib.pyplot import *
from scipy import *
from scipy.signal import *
from scipy.linalg import *
from scipy.interpolate import spline
from scipy.ndimage import *
from scipy.special import *
from numpy import fft

import Image

class DiffDatahandle(object):
	''' Class to handle the data for visualizing image difference'''
	def __init__(self, fname):
		'''Initialize the class by loading the data'''
		self.data = load(fname)

	def view(self, x, y):
		'''View the variation of the difference for the given x and y coord'''
		plot_data = self.data[x,y,:]
		clf()
		plot(plot_data)
		show()

	def multi_view(self, coord_list):
		''' View variation of difference at multiple points'''
		clf()
		for coord in coord_list:
			x, y = coord
			plot_data = self.data[x, y, :]
			plot(plot_data, label='(%d, %d)'%(x,y))
		legend()
		show()
