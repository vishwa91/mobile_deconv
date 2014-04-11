#!/usr/bin/env python

from modules.methods import *
from modules.tcp import *
from modules.kernel import *

if __name__ == '__main__':
    start_token = 'S\x00T\x00F\x00S\x00'
    end_token = 'E\x00D\x00F\x00S\x00'
    frame_token = 'S\x00T\x00F\x00R\x00'
    fstart = 'STFR'
    fend = 'EDFR'
    #dummy_recv(start_token, end_token, frame_token, 'tmp/focus/tokens.dat')
    #extract_images('tmp/focus/tokens.dat', 100, fstart, fend, 'tmp/focus/src')

    imdepth = zeros((480, 640))
    imdepth[:,:] = float('inf')
    im_max = zeros_like(imdepth)
    window = 31
    kx = array([[0,1,0],
                [0,-2,0],
                [0,1,0]])
    ky = array([[0,0,0],
                [1,-2,1],
                [0,0,0]])
    for i in range(3, 100):
        print 'Reading image %d'%i
        im = imread('tmp/focus/src/im%d.bmp'%i, flatten=True)
        gxx = linear_convolve(im, kx)
        gyy = linear_convolve(im, ky)
        imlap = abs(gxx) + abs(gyy)
        mfilter = ones((window, window), dtype=float)/(8.0*8.0)
        imlap = linear_convolve(imlap, mfilter)
        x, y = where(imlap > im_max)
        im_max[x, y] = imlap[x, y]
        imdepth[x, y] = i
        Image.fromarray((imlap*255.0/imlap.max()).astype(uint8)).save(
            'tmp/focus/sml/im%d.bmp'%i)
    #imdepth = pow(1.1, imdepth)
    Image.fromarray(imdepth*255.0/imdepth.max()).convert('RGB').save(
            'tmp/focus/imdepth.bmp')
