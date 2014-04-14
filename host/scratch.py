#!/usr/bin/env python

from modules.methods import *
from modules.tcp import *
from modules.kernel import *
from modules.depth import *

if __name__ == '__main__':
    strt_token = 'S\x00T\x00F\x00S\x00'
    end_token = 'E\x00D\x00F\x00S\x00'
    frame_token = 'S\x00T\x00F\x00R\x00'
    fstart = 'STFR'
    fend = 'EDFR'
    #continuous_recv(strt_token, end_token, frame_token, 'tmp/focus/tokens.dat')
    #extract_images('tmp/focus/tokens.dat', 100, fstart, fend, 'tmp/focus/src')
    imdepth, imfocus = sml_focus_depth('tmp/focus/src', 'tmp/focus/sml', 3, 100)
    Image.fromarray(imdepth).convert('L').save('tmp/focus/imdepth.pgm')
    Image.fromarray(imfocus).convert('L').save('tmp/focus/imfocus.pgm')
    '''
    im1 = imread('tmp/focus/src/im3.pgm')
    im2 = imread('tmp/focus/src/im99.pgm')
    xdim, ydim = im1.shape
    tx = xdim / 2; ty = ydim / 2
    scale_array = linspace(1., 1.1, 101)[:-1] 
    for i in range(1,100):
        print 'New scale of %f'%scale_array[i]
        im2 = imread('tmp/focus/src/im%d.pgm'%i)
        imtemp = zoom(im2, scale_array[i])
        x, y = imtemp.shape; cx = x // 2; cy = y // 2
        imtemp = imtemp[cx-tx:cx+tx, cy-ty:cy+ty]
        imsave = hstack((imtemp, im1))
        #imsave = abs(imtemp-im1)
        Image.fromarray(imsave).convert('L').save('tmp/focus/scale/im%d.pgm'%i)
    '''


    
