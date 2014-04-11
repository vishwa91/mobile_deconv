#!/usr/bin/env python

'''
    Routines in this file are used for visualizing data.
'''
import os, sys, time
import socket
import commands

# Seems like we have some problem with matplotlib
import matplotlib
matplotlib.use('TkAgg')

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

class AttitudeHandle(object):
    ''' Class to visualize attitude of the camera. We only look at
        in plane translation and Z axis rotation
    '''
    def __init__(self):
        # Initialize matplotlib interactive plot
        ion()
        
        # Initialize all the constants
        self.host = '192.168.151.1'     # TCP Host
        self.port = 1991                # TCP Port
        self.bufsize = 1024             # TCP buffer size
        self.t = 10e-3                  # Sampling time
        self.g = 9.8                    # Acceleration due to gravity
        self.alpha = 0.90               # LPF constant
        self.window = 10                # Window for velocity calculation

        # Data arrays
        self.xaccel = [0]
        self.yaccel = [0]
        self.zaccel = [0]

        # Old velocity values.
        self.vx = 0; self.vy = 0

        # Acceleration due to gravity
        self.gx = 0; self.gy = 0; self.gz = 0

        # plots
        self.ax_orient = subplot(2,1,1, polar=True)
        self.ax_orient.set_rmax(2.0)
        self.ax_orient.grid(True)

        self.ax_pos = subplot(2,1,2, polar=False)
        self.ax_pos.set_xlim(-1,1)
        self.ax_pos.set_ylim(-1,1)
        self.ax_pos.grid(True)            

    def connect(self):
        # Connect to the mobile
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print 'Waiting for a client'
        s.bind((self.host, self.port))
        s.listen(1)
        self.conn, self.addr = s.accept()

        print 'Connection address:', self.addr

    def start_log(self):
        # Start the logging process. Update as and when we get a new
        # value
        start_plot = False
        while True:
            data = self.conn.recv(self.bufsize).replace('\x00', '')
            # See if we need to stop the logging.
            if 'EDLG' in data:
                print 'Stopping data log'
                self.conn.close()
                print 'Connection closed'
                return 0
            if 'STLG' in data:
                start_plot = True
                print 'Started logging data'
                
            if start_plot:
                ac_tokens = data.split(';;')
                for i in ac_tokens[:-1]:
                    # The first value might be wrong. Try and catch it
                    try:
                        acx, acy, acz = i.split(';')
                        if abs(float(acx)) > 2:
                            continue
                        self.gx = (self.alpha*self.gx +
                                   (1-self.alpha)*float(acx))
                        self.gy = (self.alpha*self.gy +
                                   (1-self.alpha)*float(acy))
                        self.gz = (self.alpha*self.gz +
                                   (1-self.alpha)*float(acz))
                        self.xaccel.append(float(acx) - self.gx)
                        self.yaccel.append(float(acy) - self.gy)
                        self.zaccel.append(float(acz) - self.gz)
                        #self.update_plots()
                    except ValueError:
                        pass
                theta = 180 - arctan2(self.gy, self.gx)*180/pi
                if theta > 180:
                    print 360 - theta
                else:
                    print theta
                self.update_plots()
    def update_plots(self):
        '''
            Update the plot. The plot consists of orientation and position
        '''
        theta = arctan2(self.gy, self.gx)
        # Clear the plots first.
        self.ax_orient.clear(); self.ax_pos.clear()
        # Set the axes again.
        self.ax_orient.set_rmax(2.0); self.ax_orient.grid(True)
        self.ax_pos.set_xlim(-0.2,0.2); self.ax_pos.set_ylim(-0.2,0.2)
        self.ax_pos.grid(True); self.ax_pos.set_aspect('equal')

        # Plot data
        self.ax_orient.plot(theta, 1.0, 'bo')
        vx = cumsum(self.xaccel)[-1]*self.g*self.t
        vy = cumsum(self.yaccel)[-1]*self.g*self.t
        self.ax_pos.plot(vy-self.vy, self.vx-vx, 'bo')
        self.vx = 0.9*vx+(1-0.9)*self.vx
        self.vy = 0.9*vy+(1-0.9)*self.vy
        draw()
            
if __name__ == '__main__':
    ahandle = AttitudeHandle()
    ahandle.connect()
    ahandle.start_log()
