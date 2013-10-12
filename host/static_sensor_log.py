#!/usr/bin/python

import os, sys
import socket

from scipy import *
from scipy.linalg import *

from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D

# Global constants
HOST = '192.168.151.1'
PORT = 1991

BUFFER_SIZE = 1024

# Time information
TSTEP = 20e-3
G = 9.8

# Sharp image dimensions for focus=50
# Width = 7cm
# Height = 5.5cm

# Output files and directory
OUTPUT_DIR = 'output'
SENSOR_FILE = 'sensor_log.dat'

def tcp_listen():
	""" Listen to the TCP socket and get data"""
	# Wait till you get a socket client

	global HOST, PORT
	global STRT, STIM, EDIM, STAC, EDAC, ENDT
	global SENSOR_FILE

	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	print 'Waiting for a client'
	sock.bind((HOST, PORT))
	sock.listen(1)
	conn, addr = sock.accept()
	print 'Accepted client ', addr

	dstring = ''
	sensorfile = open(os.path.join(OUTPUT_DIR, SENSOR_FILE), 'w')
	# Start getting data
	while 1:
		data = conn.recv(BUFFER_SIZE).replace('\x00', '')
		sensorfile.write(data)
		dstring += data
		if 'STLG' in data:
			print 'Starting static sensor log.'
		if 'EDLG' in data:
			print 'Completed static sensor log.'
			conn.close()
			return dstring

class DataHandle(object):
	""" Class to handle the static sensor log data"""
	def __init__(self, dstring, fname=None, window_size=30):

		# Read either the string data or data from file.
		if fname == None:
			tokens = dstring.split('\n')
		else:
			tokens = open(fname).readlines()
		try:
			index = tokens.index('STLG\n')
		except ValueError:
			index = tokens.index('STLG')
		acdat = tokens[index+1]
		acvals = acdat.split(';;')

		self.xaccel, self.yaccel, self.zaccel = [], [], []
		for ac in acvals:
			try:
				ax, ay, az = [float(i) for i in ac.split(';')]
				self.xaccel.append(ax)
				self.yaccel.append(ay)
				self.zaccel.append(az)
			except ValueError:
				print 'Invalid acceleration data. Skipping'

		self.window_size = window_size
		self.ndata = len(self.xaccel)
		residue = self.ndata%window_size
		self.ndata -= residue
		self.nsnaps = self.ndata//window_size
		# Create a matrix of the accelerations with 30 snapshots each.
		self.ax_matrix = array(self.xaccel[:-residue]).reshape(
			(self.nsnaps, window_size))
		self.ay_matrix = array(self.yaccel[:-residue]).reshape(
			(self.nsnaps, window_size))
		self.az_matrix = array(self.zaccel[:-residue]).reshape(
			(self.nsnaps, window_size))

	def calculate_position(self, subtract_drift=True):
		""" Calculate the position from the given acceleration data.
		Note the following assumptions:
		1. The average acceleration has to be zero.
		2. The drift is assumed to be linear. Hence, we do a least
		   square fitting of the position and subtract.
		"""

		global G, TSTEP
		# Subtract the mean.
		ax = self.ax_matrix - mean(self.ax_matrix, 1).reshape(
			(self.nsnaps,1))
		ay = self.ay_matrix - mean(self.ay_matrix, 1).reshape(
			(self.nsnaps,1))
		az = self.az_matrix - mean(self.az_matrix, 1).reshape(
			(self.nsnaps,1))

		# Integrate twice to get the position.
		xpos = cumsum(cumsum(ax, 1), 1)*G*TSTEP*TSTEP
		ypos = cumsum(cumsum(ay, 1), 1)*G*TSTEP*TSTEP
		zpos = cumsum(cumsum(az, 1), 1)*G*TSTEP*TSTEP

		# Find the slope of the line drift = m*time
		time = range(self.window_size)
		norm_time = (array(time)*TSTEP).reshape((self.window_size, 1))

		self.mx, res, rank, sing = lstsq(norm_time, xpos.T)
		self.my, res, rank, sing = lstsq(norm_time, ypos.T)
		self.mz, res, rank, sing = lstsq(norm_time, zpos.T)

		if subtract_drift:
		# Subtract the drift
			self.xpos = xpos - dot(norm_time, self.mx).T
			self.ypos = ypos - dot(norm_time, self.my).T
			self.zpos = zpos - dot(norm_time, self.mz).T
		else:
			self.xpos = xpos
			self.ypos = ypos
			self.zpos = zpos

	def plot_position(self):
		""" Plot the positions in X, Y, Z direction. Note that the
		method calculate_position has to be called first"""

		global TSTEP

		fig = figure()
		time = arange(0, TSTEP*self.window_size, TSTEP)[:self.window_size]

		print self.xpos.shape, time.shape

		subplot(3,1,1)
		for i in range(self.nsnaps):
			plot(time, self.xpos[i, :])
		title('X position')

		subplot(3,1,2)
		for i in range(self.nsnaps):
			plot(time, self.ypos[i, :])
		title('Y position')

		subplot(3,1,3)
		for i in range(self.nsnaps):
			plot(time, self.zpos[i, :])
		title('Z position')

		show()

if __name__ == '__main__':
	#dstring = tcp_listen()
	#dhandle = DataHandle(dstring)
	dhandle = DataHandle(dstring=None, fname='output/sensor_log.dat')
	dhandle.calculate_position(subtract_drift=True)
	dhandle.plot_position()