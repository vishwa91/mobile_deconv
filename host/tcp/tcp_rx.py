#!/usr/bin/python

import os, sys
import shutil
import commands
import socket
from StringIO import StringIO

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

IM_WIDTH = 640
IM_HEIGHT = 640

INTERPOLATE_SCALE = 40

# Output files and directory
OUTPUT_DIR = '../output/cam'
IMAGE_NAME = 'saved_im.bmp'
PREVIEW_IMAGE_NAME = 'preview_im.bmp'
ACCEL_FILE = 'saved_ac.dat'
TOKEN_FILE = 'tokens.dat'
TMP_DIR = '../tmp/cam'

# Delimiting tokens
STRT = '\x00S\x00T\x00R\x00T\x00\n'
STIP = '\x00S\x00T\x00I\x00P\x00\n'
EDIP = '\x00E\x00D\x00I\x00P\x00\n'
STIM = '\x00S\x00T\x00I\x00M\x00\n'
EDIM = '\x00E\x00D\x00I\x00M\x00\n'
STAC = '\x00S\x00T\x00A\x00C\x00\n'
EDAC = '\x00E\x00D\x00A\x00C\x00\n'
ENDT = '\x00E\x00N\x00D\x00T\x00\n'

def _deblur(kernel, im, nsr, mode):
    ''' Deblur a single channel image'''
    if mode in ['wien', 'reg']:
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
        if mode == 'reg':
            nsr *= REG
        IMOUT = conj(F)*IM/(abs(F)**2 + nsr)
        imout = real(fft.ifft2(IMOUT))
    elif mode == 'rl':
        flipped_kernel = flipud(fliplr(kernel))
        imout = ones_like(im)
        for i in range(20):
            temp = im/(nsr+fftconvolve(imout, kernel, mode='same'))
            imout = imout*fftconvolve(temp, flipped_kernel, mode='same')
    else:
        raise('mode needs to be wien, reg or rl. Type help(deblur) for help')
    #return (imout*255.0/imout.max()).astype(float)
    return imout.astype(float)

def deblur(kernel, im, nsr, mode='rl'):
    ''' Deblur an image. Currently, we have 3 options:
        1. Wiener with mode = wien
        2. Regularized with mode = reg
        3. Richardson lucy, with mode = rl
    '''
    s = im.shape
    
    if len(s) == 2:
        return _deblur(kernel, im, nsr, mode)
    else:
        x,y,_ = im.shape
        imout = zeros((x, y, 3), dtype=float)
        imout[:,:,0] = _deblur(kernel, im[:,:,0], nsr, mode)
        imout[:,:,1] = _deblur(kernel, im[:,:,1], nsr, mode)
        imout[:,:,2] = _deblur(kernel, im[:,:,2], nsr, mode)
        
        return imout.astype(uint8)

def pgauss(val, mean, mvar):
    ''' Return the probability of val'''
    den = sqrt(2*pi*mvar)
    mexp = ((val-mean)**2)/(2.0*mvar)
    return exp(-mexp)/den
    
def estimate_g(data, niters=10):
    ''' We estimate graviational value using iterative method.'''
    g = mean(data)
    gvar = variance(data)
    for i in range(niters):
        g=sum(data*pgauss(data, g, gvar))/sum(pgauss(data, g, gvar))
    return g
        
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
    Image.fromarray(preview_im[:,:,:3]).convert('RGB').save(
        os.path.join(OUTPUT_DIR, PREVIEW_IMAGE_NAME))
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

def my_main():
    # Start listening to TCP socket
    dstring = tcp_listen();save_data(dstring);dhandle = DataHandle(dstring)
    #dhandle = DataHandle(None, os.path.join(OUTPUT_DIR, ACCEL_FILE),os.path.join(OUTPUT_DIR, IMAGE_NAME))
    dhandle.calculate_position(linear_drift = False)
    #dhandle.plot_position()
    print len(dhandle.xaccel)
    
    # Clear up the im and the kernel directory
    shutil.rmtree(os.path.join(TMP_DIR, 'kernel'))
    shutil.rmtree(os.path.join(TMP_DIR, 'im'))
    os.mkdir(os.path.join(TMP_DIR, 'kernel'))
    os.mkdir(os.path.join(TMP_DIR, 'im'))
    
    count = 0
    dhandle.im.save(os.path.join(TMP_DIR, 'imtest.bmp'))
    
    # The right position values start from t=43 to t=43+20
    maxshift = 10e-3
    shiftstep = 2e-3
    tstart = 41
    
    for xfinal in arange(-maxshift, maxshift, shiftstep):
        for yfinal in arange(-maxshift, maxshift, shiftstep):
            for depth in range(900, 1500, 20):
                # Compute the kernel
                print 'Computing new latent image with x=%f,y=%f,d=%f'%(
                xfinal,yfinal, depth)
                kernel = dhandle.compute_kernel((xfinal, yfinal),
                            depth, 'quad', tstart, tstart+21)
                imout = deblur(kernel, array(dhandle.im)[:,:,0], 0.001, mode='reg')
                Image.fromarray(imout).convert('RGB').save(
                os.path.join(TMP_DIR, 'im/im%d.jpg'%count))
                kernel *= 255.0/kernel.max()
                Image.fromarray(kernel).convert('RGB').save(
                os.path.join(TMP_DIR, 'kernel/kernel%d.bmp'%count))
                #os.path.join(TMP_DIR, 'kernel/kernel_%f_%f_%d.bmp'%(
                #                    xfinal, yfinal, depth)))
                
                #out = commands.getoutput('../output/cam/robust_deconv.exe ../tmp/cam/imtest.bmp ../tmp/cam/kernel/kernel%d.bmp ../tmp/cam/im/im%d.bmp 0 0.1 1'%(count,count))
                #print out
                
                count += 1

if __name__ == '__main__':
    my_main()
