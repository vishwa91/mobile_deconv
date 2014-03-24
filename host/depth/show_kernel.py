#!/usr/bin/env python

# All includes from this file
from two_im_depth import *

def estimate_pos(accel, start, end):
    ''' Simple calculation of position using just integration'''
    if end == len(accel[:,0]):
        end = -1
    # Convert acceleration into m/s^2
    accel *= G
    xaccel = accel[:,0]; yaccel = accel[:,1]; zaccel = accel[:,2]
    raw_xpos = cumsum(cumsum(xaccel[start:end]))*T*T
    raw_ypos = cumsum(cumsum(yaccel[start:end]))*T*T
    raw_zpos = cumsum(cumsum(zaccel[start:end]))*T*T

    return raw_xpos, raw_ypos, raw_zpos

if __name__ == '__main__':
    im = imread('../output/cam/saved_im.bmp', flatten=True)
    data = loadtxt(accel_data_file)
    ntime = 20
    idx_start = 41
    idx_end = idx_start + ntime + 2
    xpos, ypos, zpos = estimate_pos(data, idx_start, idx_end)
    dmax = hypot(xpos, ypos).max()
    kernel = construct_kernel(xpos, ypos, 100/dmax, 10)
    Image.fromarray(im).show()
    Image.fromarray(kernel*255.0/kernel.max()).show()
       
