#!/usr/bin/python

from scipy import *
from scipy.ndimage import *
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D
acdat = loadtxt('output/test_ac.dat')
xaccel, yaccel, zaccel = acdat[:,0], acdat[:,1], acdat[:,2]

# Assume xaccel has -1 due to gravity
xaccel -= mean(xaccel)
yaccel -= mean(yaccel)
g = 9.8
t = 10e-3
xpos = cumsum(cumsum(xaccel))*g*t*t*640/(3.2e-3)
ypos = cumsum(cumsum(yaccel))*g*t*t*480/(2.4e-3)
ac_fig = figure()
plot(xaccel, yaccel)
pos_fig = figure()
plot(xpos, ypos)
ypos_fig = figure()
plot(range(len(yaccel)), xpos)
show()
