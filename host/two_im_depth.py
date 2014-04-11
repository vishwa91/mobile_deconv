#!/usr/bin/env python

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

# Custom modules
from modules.deconv import *
from modules.kernel import *
from modules.methods import *
from modules.depth import *

accel_data_file = '../output/cam/saved_ac.dat'
T = 10e-3
G = 9.8

if __name__ == '__main__':
    try:
        os.mkdir('../tmp/steer')
    except OSError:
        pass
    for idx in [1]:#range(1,7):
        main_dir = 'output/cam'
        impure = imread(os.path.join(main_dir, 'preview_im.bmp'), flatten=True)
        imblur = imread(os.path.join(main_dir, 'saved_im.bmp'), flatten=True)


        # Load the acceleration data.
        data = loadtxt(os.path.join(main_dir, 'saved_ac.dat'))
        start = 5
        end = 30
        x, y, z = estimate_simple_pos(data, start, end)
        x -= mean(x); y -= mean(y)

        #y = range(-4, 5) + [1]*10
        #x = linspace(0,1,len(y))

        niters = 10
        window = 4

        impure = register(impure, imblur)
        shifts = range(-5,5,1)
        for xshift in shifts:
            for yshift in shifts:
                print 'Estimating depth for a (%d,%d) shift'%(xshift, yshift)
                imdepth, save_data = iterative_depth(
                    shift(impure, [xshift, yshift]), imblur, x, y)
                imdepth *= 255.0/imdepth.max()
                Image.fromarray(imdepth).convert('RGB').save(
                    'tmp/steer/%d/depth_%d_%d.bmp'%(idx,xshift,yshift))
                #save('../tmp/diff_data', save_data)
        
    imdepth = 255*imdepth/imdepth.max()
    Image.fromarray(imdepth).show()
    Image.fromarray(imdepth).convert('RGB').save(
        'tmp/steer/imdepth.bmp')
