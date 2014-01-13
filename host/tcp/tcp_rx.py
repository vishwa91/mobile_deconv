#!/usr/bin/python

#from __future__ import print_function
import os, sys
import commands
import socket
from StringIO import StringIO

from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D

from scipy import *
from scipy.ndimage import *
from scipy.linalg import *
from numpy import fft
from scipy.signal import convolve2d
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

IM_WIDTH = 640
IM_HEIGHT = 640

INTERPOLATE_SCALE = 40

# Output files and directory
OUTPUT_DIR = '../output/cam'
IMAGE_NAME = 'saved_im.bmp'
ACCEL_FILE = 'saved_ac.dat'
TOKEN_FILE = 'tokens.dat'
TMP_DIR = '../tmp/cam'

# Delimiting tokens
STRT = '\x00S\x00T\x00R\x00T\x00\n'
STIM = '\x00S\x00T\x00I\x00M\x00\n'
EDIM = '\x00E\x00D\x00I\x00M\x00\n'
STAC = '\x00S\x00T\x00A\x00C\x00\n'
EDAC = '\x00E\x00D\x00A\x00C\x00\n'
ENDT = '\x00E\x00N\x00D\x00T\x00\n'

def _deblur(kernel, im, nsr):
    ''' Deblur a single channel image'''
    kernel = (1.0*kernel)/kernel.sum()
    diffx = array([[1,  1],
                   [-1,-1]])
    diffy = array([[-1, 1],
                   [-1, 1]])
    x, y = im.shape
    #x *= 2; y *= 2
    DX = fft.fft2(diffx, s=(x,y))
    DY = fft.fft2(diffy, s=(x,y))
    REG = abs(DX)**2 + abs(DY)**2
    #x2, y2 = kernel.shape
    #x = x1+x2; y = y1+y2
    F = fft.fft2(kernel, s=(x,y))
    IM = fft.fft2(im, s=(x,y))
    IMOUT = conj(F)*IM/(abs(F)**2 + nsr*REG)
    imout = real(fft.ifft2(IMOUT))
    
    return imout.astype(float)

def deblur(kernel, im, nsr):
    ''' Deblur image using Wiener filter method'''
    s = im.shape
    
    if len(s) == 2:
        return _deblur(kernel, im, nsr)
    else:
        x,y,_ = im.shape
        imout = zeros((x, y, 3), dtype=float)
        imout[:,:,0] = _deblur(kernel, im[:,:,0], nsr)
        imout[:,:,1] = _deblur(kernel, im[:,:,1], nsr)
        imout[:,:,2] = _deblur(kernel, im[:,:,2], nsr)
        
        return imout.astype(uint8)
        
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

    # Save the image and the data
    imstring = dstring[stim_index+len(STIM)+1:edim_index-1]
    im = Image.open(StringIO(imstring[:-2]))
    im.save(os.path.join(OUTPUT_DIR, IMAGE_NAME))
    afile.close()

class DataHandle(object):
    """ Class to get the polished data handle. This includes the 
    image, the acceleration, position and velocity. In future when
    the gyroscope will be incorporated, the handle will have rotation
    data also"""

    def __init__(self, dstring, acname=None, imname=None):
        """ Extract the data and store the image and other values"""
        global STIM, EDIM, STAC, EDAC

        if acname == None:
            # Get the index of the image delimiters
            stim_index = dstring.index(STIM)
            edim_index = dstring.index(EDIM)

            # Get the index of the acceleration data delimiters
            stac_index = dstring.index(STAC)
            edac_index = dstring.index(EDAC)

            acstring = dstring[stac_index+len(STAC)+1:edac_index-1]
            acstring = acstring.replace('\x00', '')
            self.xaccel, self.yaccel, self.zaccel = [], [], []

            acvals = acstring.split(';;')
            for ac in acvals:
                try:
                    ax, ay, az = [float(i) for i in ac.split(';')]
                    self.xaccel.append(ax)
                    self.yaccel.append(ay)
                    self.zaccel.append(az)
                except ValueError:
                    print 'Invalid acceleration value. Skipping'
        else:
            acdat = loadtxt(acname)
            self.xaccel, self.yaccel, self.zaccel = [acdat[:, 0],
                                                     acdat[:, 1],
                                                     acdat[:, 2]]

        self.ndata = len(self.xaccel)

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
        avgx = mean(array(self.xaccel[start:end]))
        avgy = mean(array(self.yaccel[start:end]))
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
        
        xdim = max(abs(xpos))
        ydim = max(abs(ypos))
        
        kernel = zeros((2*xdim+1, 2*ydim+1), dtype=uint8)
        
        for i in range(len(xpos)):
            kernel[xdim+xpos[i], ydim-ypos[i]] += 1
            
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

    def deblurr_kernel(self):
        """ Construct the deblurr kernel from the position information
        """

        global IM_HEIGHT, IM_WIDTH
        global WORLD_HEIGHT, WORLD_WIDTH

        scale_x = IM_HEIGHT/WORLD_HEIGHT
        scale_y = IM_WIDTH/WORLD_WIDTH

        # Find the shift in terms of pixels.
        xpos_pixel = (self.xpos*scale_x).astype(int)
        ypos_pixel = (self.ypos*scale_y).astype(int)

        # Smoothen the position to 10 times the points.
        xlen = len(xpos_pixel)
        ylen = len(ypos_pixel)
        
        xpos_pixel = spline(range(xlen), xpos_pixel,
                            linspace(0, xlen, xlen*INTERPOLATE_SCALE))
        ypos_pixel = spline(range(ylen), ypos_pixel,
                            linspace(0, ylen, ylen*INTERPOLATE_SCALE))
        
        xdim, ydim = abs(xpos_pixel).max(), abs(ypos_pixel).max()
        blurr_kernel = zeros(((xdim+1)*2, (ydim+1)*2))

        for i in range(len(xpos_pixel)):
            blurr_kernel[xdim+xpos_pixel[i], ydim+ypos_pixel[i]]+=1
            #blurr_kernel[ydim+ypos_pixel[i], xdim+xpos_pixel[i]]+=1
        return blurr_kernel

def _junk_deblur(im, kernel, nsr=0.01):
    """ Weiner deconvolution for deblurring"""
    # Attempt a dumb deconvolution
    if len(im.shape) == 2:
        kx, ky = kernel.shape
        xdim, ydim = im.shape
        nchan = 1
    else:
        kx, ky = kernel.shape
        xdim, ydim, nchan = im.shape
    if nchan == 3:  
        imr = im[:,:,0]
        img = im[:,:,1]
        imb = im[:,:,2]

        imdim = (xdim, ydim)
        IMR = fft2(imr, s=imdim)
        IMG = fft2(img, s=imdim)
        IMB = fft2(imb, s=imdim)

        H = fft2(kernel/sum(kernel), s=imdim)

        IMOUTR = IMR*conj(H)/(H*conj(H) + nsr)
        IMOUTG = IMG*conj(H)/(H*conj(H) + nsr)
        IMOUTB = IMB*conj(H)/(H*conj(H) + nsr)

        imoutr = ifft2(IMOUTR)
        imoutg = ifft2(IMOUTG)
        imoutb = ifft2(IMOUTB)

        imout = zeros((xdim, ydim, 3))
        imout[:,:,0] = imoutr
        imout[:,:,1] = imoutg
        imout[:,:,2] = imoutb

    elif nchan == 1:
        IM = fft2(im, s=(xdim, ydim))
        H = fft2(kernel/sum(kernel), s=(xdim, ydim))
        IMOUT = IM*conj(H)/(H*conj(H) + nsr)
        imout = ifft2(IMOUT)

    return (imout*255.0/imout.max()).astype(uint8)


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
        if ENDT in dstring:
            print 'Reception complete. Closing connection'
            conn.close()
            return dstring

if __name__  == '__main__':
    # Start listening to TCP socket
    dstring = tcp_listen();save_data(dstring);dhandle = DataHandle(dstring)
    #dhandle = DataHandle(None, os.path.join(OUTPUT_DIR, ACCEL_FILE),os.path.join(OUTPUT_DIR, IMAGE_NAME))
    dhandle.calculate_position(linear_drift = False)
    #dhandle.plot_position()
    print len(dhandle.xaccel)
    
    count = 0
    dhandle.im.save(os.path.join(TMP_DIR, 'imtest.bmp'))
    # The right position values start from t=43 to t=43+20
    maxshift = 1e-2
    shiftstep = 1e-3
    tstart = 41
    for xfinal in arange(-maxshift, maxshift, shiftstep):
        for yfinal in arange(-maxshift, maxshift, shiftstep):
            for depth in [5000]:
                # Compute the kernel
                print 'Computing new latent image with x=%f,y=%f,d=%f'%(
                xfinal,yfinal, depth),
                print '\r'
                kernel = dhandle.compute_kernel((xfinal, yfinal),
                            depth, 'linear', tstart, tstart+20)
                #imout = deblur(kernel, array(dhandle.im)[:,:,0], 0.01)
                #Image.fromarray(imout).convert('RGB').save(
                #os.path.join(TMP_DIR, 'im/im%d.bmp'%count))
                kernel *= 200.0/kernel.max()
                Image.fromarray(kernel).convert('RGB').save(
                os.path.join(TMP_DIR, 'kernel/kernel_%f_%f.bmp'%(
                                    xfinal, yfinal)))
                
                #out = commands.getoutput('../output/cam/robust_deconv.'
                #                         'exe ../tmp/cam/imtest.bmp'
                #                         ' ../tmp/cam/kernel/kernel%d'
                #                         '.bmp ../tmp/cam/im/im%d.bmp'
                #                         '0 0.1 1'%(count,count))
                #print out
                
                count += 1
    
