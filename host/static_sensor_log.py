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
	sensorfile = open(os.path.join(OUTPUT_DIR, TOKEN_FILE), 'w')
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