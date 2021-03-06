#!/usr/bin/python

'''
    Modules in this file are used for TCP communication with the mobile.
'''

import os, sys
import shutil
import commands
import socket
import time
from StringIO import StringIO

import matplotlib
matplotlib.use('TkAgg')

from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D

from scipy import *
from scipy.ndimage import *
from scipy.linalg import *
from scipy.signal import *
from numpy import fft
from scipy.interpolate import spline

import Image

# Global constants
HOST = '192.168.151.1'
PORT = 1991

BUFFER_SIZE = 1024

# Time information
TSTEP = 10e-3
G = 9.8

# Sharp image dimensions for focus=50
# Width = 7cm
# Height = 5.5cm

WORLD_WIDTH = 7e-2
WORLD_HEIGHT = WORLD_WIDTH
WINDOW_SIZE = 100

IM_WIDTH = 640
IM_HEIGHT = 640

INTERPOLATE_SCALE = 40

# Output files and directory
OUTPUT_DIR = 'output/cam'
IMAGE_NAME = 'saved_im.bmp'
PREVIEW_IMAGE_NAME = 'preview_im.bmp'
ACCEL_FILE = 'saved_ac.dat'
TOKEN_FILE = 'tokens.dat'
TMP_DIR = 'tmp/cam'

# Delimiting tokens
STRT = '\x00S\x00T\x00R\x00T\x00\n'
STIP = '\x00S\x00T\x00I\x00P\x00\n'
EDIP = '\x00E\x00D\x00I\x00P\x00\n'
STIM = '\x00S\x00T\x00I\x00M\x00\n'
EDIM = '\x00E\x00D\x00I\x00M\x00\n'
STAC = '\x00S\x00T\x00A\x00C\x00\n'
EDAC = '\x00E\x00D\x00A\x00C\x00\n'
ENDT = '\x00E\x00N\x00D\x00T\x00\n'

def save_data(dstring):
    """ Function to extract and save the image and acceleration data
    from the incoming tcp data"""
    global STIM, EDIM, STAC, EDAC
    global OUTPUT_DIR, IMAGE_NAME, ACCEL_FILE

    # Check if we need to create the output directory
    try:
        os.mkdir(OUTPUT_DIR)
    except OSError:
        pass

    # Get index of image preview delimiters
    stip_index = dstring.index(STIP)
    edip_index = dstring.index(EDIP)

    # Get the index of the image delimitors
    stim_index = dstring.index(STIM)
    edim_index = dstring.index(EDIM)

    # Get the index of the acceleration data delimitors
    stac_index = dstring.index(STAC)
    edac_index = dstring.index(EDAC)

    acstring = dstring[stac_index+len(STAC)+1:edac_index-1]
    acstring = acstring.replace('\x00', '')

    acvals = acstring.split(';;')
    afile = open(os.path.join(OUTPUT_DIR, ACCEL_FILE), 'w')
    for ac in acvals:
        afile.write(ac.replace(';', ' ')+'\n')

    # Save preview image
    preview_string = dstring[stip_index+len(STIP)+1:edip_index-1]
    preview_im_flat = array([ord(i) for i in preview_string])
    preview_im = preview_im_flat.reshape((480,640,4)).astype(uint8)
    preview_im[:,:,2], preview_im[:,:,0] = preview_im[:,:,0], preview_im[:,:,2]
    Image.fromarray(preview_im[:,:,:3]).convert('RGB').save(
        os.path.join(OUTPUT_DIR, PREVIEW_IMAGE_NAME))
    # Save the image and the data
    imstring = dstring[stim_index+len(STIM)+1:edim_index-1]
    im = Image.open(StringIO(imstring[:-2]))
    im.save(os.path.join(OUTPUT_DIR, IMAGE_NAME))
    afile.close()

class TCPDataHandle(object):
    """ Class to get the polished data handle. This includes the 
    image, the acceleration, position and velocity. In future when
    the gyroscope will be incorporated, the handle will have rotation
    data also"""

    def __init__(self, dstring, acname=None, imname=None, impreview=None):
        """ Extract the data and store the image and other values"""
        global STIM, EDIM, STAC, EDAC

        if acname == None:
            # Get the index of the image delimiters
            stim_index = dstring.index(STIM)
            edim_index = dstring.index(EDIM)

            # Get index of image preview delimiters
            stip_index = dstring.index(STIP)
            edip_index = dstring.index(EDIP)

            # Get the index of the acceleration data delimiters
            stac_index = dstring.index(STAC)
            edac_index = dstring.index(EDAC)

            acstring = dstring[stac_index+len(STAC)+1:edac_index-1]
            acstring = acstring.replace('\x00', '')
            self.xaccel, self.yaccel, self.zaccel = [], [], []
            self.gx, self.gy, self.gz = [], [], []

            acvals = acstring.split(';;')
            for ac in acvals:
                try:
                    ax, ay, az, gx, gy, gz = [float(i) for i in ac.split(';')]
                    self.xaccel.append(ax); self.gx.append(gx) 
                    self.yaccel.append(ay); self.gy.append(gy)
                    self.zaccel.append(az); self.gz.append(gz)
                except ValueError:
                    print 'Invalid acceleration value. Skipping'
        else:
            acdat = loadtxt(acname)
            self.xaccel, self.yaccel, self.zaccel = [acdat[:, 0],
                                                     acdat[:, 1],
                                                     acdat[:, 2]]

        self.ndata = len(self.xaccel)
        if impreview == None:
            # Get the preview image.
            preview_string = dstring[stip_index+len(STIP)+1:edip_index-1]
            preview_im_flat = array([ord(i) for i in preview_string])
            self.preview_im = preview_im_flat.reshape((480,640,4)).astype(uint8)
        else:
            self.preview_im = Image.open(impreview)
        if imname == None:
            # Get the image.
            imstring = dstring[stim_index+len(STIM)+1:edim_index-1]
            self.im = Image.open(StringIO(imstring[:-2]))
        else:
            self.im = Image.open(imname)

    def compute_kernel(self, fpos=None, depth=1.0, shift_type='linear', 
                        start=0, end=-1):
        '''
            Compute the blur kernel given the final position and depth.
            Note that the running average of the acceleration data will
            be taken as the effect of gravity.
        '''
        global G, TSTEP, INTERPOLATE_SCALE
        
        if end == -1:
            end = len(self.xaccel)
        #remove the average gravity effect.
        mean_vector = sqrt(self.xaccel**2 + self.yaccel**2 + self.zaccel**2)
        avgx = mean(array(self.xaccel[start:end]))
        avgy = mean(array(self.yaccel[start:end]))
        #avgx = mean(array(self.xaccel[start:end]))
        #avgy = mean(array(self.yaccel[start:end]))
        xaccel = array(self.xaccel[start:end]) - avgx
        yaccel = array(self.yaccel[start:end]) - avgy
        
        ntime = len(xaccel)
        
        xpos_temp = cumsum(cumsum(xaccel))*G*TSTEP*TSTEP
        ypos_temp = cumsum(cumsum(yaccel))*G*TSTEP*TSTEP
        
        xpos_temp -= mean(xpos_temp)
        ypos_temp -= mean(ypos_temp)
        
        xpos_temp = spline(range(ntime), xpos_temp,
                        linspace(0, ntime, ntime*INTERPOLATE_SCALE))
        ypos_temp = spline(range(ntime), ypos_temp,
                        linspace(0, ntime, ntime*INTERPOLATE_SCALE))
        ntime *= INTERPOLATE_SCALE
        # arange: (start, end, step)
        time = arange(0, ntime*TSTEP, TSTEP)
        endtime = time[-1]
        # Subtract the linear drift and multiply with depth.
        if shift_type == 'linear':
            driftx = linspace(0, fpos[0], ntime)
            drifty = linspace(0, fpos[1], ntime)
        elif shift_type == 'quad':
            # We assume shift is of the form y = a*x^2 + b*x
            a = -fpos[0]/(endtime**2)
            b = -2*a*endtime
            driftx = a*time**2 + b*time
            
            a = -fpos[1]/(endtime**2)
            b = -2*a*endtime
            drifty = a*time**2 + b*time
            
        else:
            raise ValueError("shift_type has to be linear or quad")
        
        xpos = depth*(xpos_temp - driftx)
        ypos = depth*(ypos_temp - drifty)
        
        xdim = max(abs(xpos))*4.0/3.0
        ydim = max(abs(ypos))
        
        kernel = zeros((2*xdim+1, 2*ydim+1), dtype=uint8)
        
        for i in range(len(xpos)):
            kernel[xdim+xpos[i]*4.0/3.0, ydim-ypos[i]] += 1
            
        return kernel
        
    def calculate_position(self, linear_drift=True,
                              final_pos=None, depth=1.0):
        """ Calculate the position from the given acceleration data.
        Note the following assumptions:
        1. The average acceleration has to be zero.
        2. The drift is assumed to be linear. Hence, we do a least
           square fitting of the position and subtract.
        """

        global G, TSTEP
        # Subtract the mean.
        nwindow = 10
        window = ones(nwindow)/(nwindow*1.0)
        ntime = len(self.xaccel)
        avgx = convolve(array(self.xaccel), window)[:ntime]
        avgy = convolve(array(self.yaccel), window)[:ntime]
        avgz = convolve(array(self.zaccel), window)[:ntime]
        xaccel = array(self.xaccel) - avgx
        yaccel = array(self.yaccel) - avgy
        zaccel = array(self.zaccel) - avgz

        # Integrate twice to get the position.
        xpos = cumsum(cumsum(xaccel))*G*TSTEP*TSTEP
        ypos = cumsum(cumsum(yaccel))*G*TSTEP*TSTEP
        zpos = cumsum(cumsum(zaccel))*G*TSTEP*TSTEP

        # Find the slope of the line 'drift = m*time'. If final_pos is provided,
        # Use drift = (final_pos/time_end)*time
        if final_pos == None:
            norm_time = arange(0, TSTEP*self.ndata,
             TSTEP)[:self.ndata].reshape((self.ndata,1))
            self.mx, res, rank, sing = lstsq(norm_time, xpos.T)
            self.my, res, rank, sing = lstsq(norm_time, ypos.T)
            self.mz, res, rank, sing = lstsq(norm_time, zpos.T)
        else:
            final_time = norm_time[-1]
            self.mx = final_pos[0] / final_time
            self.my = final_pos[1] / final_time
            self.mz = final_pos[2] / final_time            

        if linear_drift:
            # Subtract the drift
            self.ypos = ypos - self.my*norm_time.reshape((self.ndata))
            self.xpos = xpos - self.mx*norm_time.reshape((self.ndata))
            self.zpos = zpos - self.mz*norm_time.reshape((self.ndata))
        else:
            self.ypos = ypos
            self.xpos = xpos
            self.zpos = zpos

    def plot_position(self):
        """ Plot the positions in X, Y, Z direction. Note that the
        method calculate_position has to be called first"""

        global TSTEP
        global IM_HEIGHT, IM_WIDTH
        global WORLD_HEIGHT, WORLD_WIDTH
        fig = figure()
        time = arange(0, TSTEP*self.ndata, TSTEP)[:self.ndata]

        scale_x = IM_HEIGHT/WORLD_HEIGHT
        scale_y = scale_x
        
        subplot(3,1,1)
        plot(time, self.xpos*scale_x)
        #plot(time, time*self.mx*scale_x, 'r')
        #plot(time, (self.xpos+time*self.mx)*scale_x, 'y.')
        title('X position')

        subplot(3,1,2)
        plot(time, self.ypos*scale_y)
        #plot(time, time*self.my*scale_y, 'r')
        #plot(time, (self.ypos+time*self.my)*scale_y, 'y.')
        title('Y position')

        subplot(3,1,3)
        plot(time, self.zpos)
        #plot(time, time*self.mz, 'r')
        #plot(time, self.zpos+time*self.mz, 'y.')
        title('Z position')

        show()

def tcp_listen():
    """ Listen to the TCP socket and get data"""
    # Wait till you get a socket client

    global HOST, PORT
    global STRT, STIM, EDIM, STAC, EDAC, ENDT
    global TOKEN_FILE

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print 'Waiting for a client'
    sock.bind((HOST, PORT))
    sock.listen(1)
    conn, addr = sock.accept()
    print 'Accepted client ', addr

    dstring = ''
    tokenfile = open(os.path.join(OUTPUT_DIR, TOKEN_FILE), 'w')
    # Start getting data
    while 1:
        data = conn.recv(BUFFER_SIZE)
        dstring += data
        tokenfile.write(data)

        # Check if we need to start reception
        if STRT in data:
            print 'Started reception'
        if STIM in data:
            print 'Starting image reception'
        if STAC in data:
            print 'Starting acceleration data reception'
        if EDAC in data:
            print 'Acceleration reception done'
        if EDIM in data:
            print 'Image reception done'
        if STIP in data:
            print 'Started receiving preview image'
        if EDIP in data:
            print 'Preview image reception complete'
        if ENDT in dstring:
            print 'Reception complete. Closing connection'
            conn.close()
            return dstring

def get_tcp_data():
    """Get the images and accelerometer data from the mobile"""
    dstring = tcp_listen();
    save_data(dstring);
    return TCPDataHandle(dstring)

def continuous_recv(start_token, end_token, frame_token, save_path):
    '''Print a message everytime a new token arrives'''
    # Wait till you get a socket client
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print 'Waiting for a client'
    s.bind((HOST, PORT))
    s.listen(1)
    conn, addr = s.accept()

    print 'Connection address:', addr
    time_old = time.clock()
    data_string = ''
    while 1:
        data = conn.recv(BUFFER_SIZE)
        data_string += data
        time_delay = time.clock() - time_old
        time_old = time.clock()
        if start_token in data:
            print 'Started logging'
        if frame_token in data:
            print 'got a new frame at ', time_old
        if end_token in data:
            print 'Stopped logging data'
            conn.close()
            break
    # Save the data
    f = open(save_path, 'w')
    f.write(data_string)
    f.close()

def extract_images(load_path, nimages, fstart, fend, save_path='.'):
    ''' Extract images from saved data'''
    data = open(load_path).read()

    # Extract nimages
    for i in range(1,nimages):
        print 'Extracting %d image'%i
        stidx_raw = fstart+str(i)+'\n'; stidx = ''
        edidx_raw = fend+str(i)+'\n'; edidx = ''
        for p in stidx_raw:
            stidx += p + '\x00'
        for p in edidx_raw:
            edidx += p + '\x00'
        start = data.index(stidx)
        end = data.index(edidx)
        imdat = array([ord(k) for k in data[start+len(stidx):end]])
        im = imdat.reshape((480,640))
        Image.fromarray(im).convert('L').save(
            os.path.join(save_path, 'im%d.pgm'%i))

def live_sensors():
    """This function should be called for plotting the sensor data dynamically.
    """
    
    # Activate live plotting
    dummy = [0]*WINDOW_SIZE
    ion()
    # Create the plot lines
    subplot(3,1,1)
    linex, = plot(dummy)
    xlabel('Time ( ms)')
    ylabel('m/s^2')
    title('X acceleration')
    ylim([-2,2])

    subplot(3,1,2)
    liney, = plot(dummy)
    xlabel('Time ( ms)')
    ylabel('m/s^2')
    title('Y acceleration')
    ylim([-2,2])

    subplot(3,1,3)
    linez, = plot(dummy)
    xlabel('Time ( ms)')
    ylabel('m/s^2')
    title('Z acceleration')
    ylim([-2,2])

    # Wait till you get a socket client
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print 'Waiting for a client'
    s.bind((HOST, PORT))
    s.listen(1)
    conn, addr = s.accept()

    print 'Connection address:', addr

    start_plot = False
    enough_data = False
    xaccel = []
    yaccel = []
    zaccel = []
    time = []
    current_time = 0

    gx = 0; gy = 0; gz = 0;
    alpha = 0.99
    while 1:
        data = conn.recv(BUFFER_SIZE).replace('\x00', '')
        if 'EDLG' in data:
            break
        if 'STLG' in data:
            start_plot = True
            print 'Starting sensor log.'
        if start_plot:
            ac_tokens = data.split(';;')
            for i in ac_tokens[:-1]:
                try:
                    acx, acy, acz = i.split(';')
                    gx = alpha*gx + (1-alpha)*float(acx)
                    gy = alpha*gy + (1-alpha)*float(acy)
                    gz = alpha*gz + (1-alpha)*float(acz)
                    xaccel.append(float(acx))
                    yaccel.append(float(acy))
                    zaccel.append(float(acz))
                    time.append(current_time)
                    current_time += TSTEP
                except ValueError:
                    print i
                    print 'Not valid data. Skipping'
            if len(xaccel) > WINDOW_SIZE:
                enough_data = True
                    
        if enough_data:
            linex.set_xdata(range(WINDOW_SIZE))
            liney.set_xdata(range(WINDOW_SIZE))
            linez.set_xdata(range(WINDOW_SIZE))

            linex.set_ydata(cumsum(array(xaccel[-WINDOW_SIZE:]) - gx))
            liney.set_ydata(cumsum(array(yaccel[-WINDOW_SIZE:]) - gy))
            linez.set_ydata(cumsum(array(zaccel[-WINDOW_SIZE:]) - gz))
            draw()
        
    conn.close()
    print 'Closed connection'    
    return 0
