#!/usr/bin/python

from scipy import *
from scipy.ndimage import *
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D
acdat = loadtxt('output/test_ac.dat')
xaccel, yaccel, zaccel = acdat[:,0], acdat[:,1], acdat[:,2]

g = 9.8
t = 20e-3

time = arange(0, len(xaccel)*t, t)[:len(xaccel)]

xaccel -= mean(xaccel)
yaccel -= mean(yaccel)
zaccel -= mean(zaccel)

xpos = cumsum(cumsum(xaccel))*g*t*t
ypos = cumsum(cumsum(yaccel))*g*t*t
zpos = cumsum(cumsum(zaccel))*g*t*t

# Position plots
xposfig = figure()
plot(time, xpos)
xlabel('Time (s)')
ylabel('Position (m)')
title('X position')

yposfig = figure()
plot(time, ypos)
xlabel('Time (s)')
ylabel('Position (m)')
title('Y position')

zposfig = figure()
plot(time, zpos)
xlabel('Time (s)')
ylabel('Position (m)')
title('Z position')

show()
