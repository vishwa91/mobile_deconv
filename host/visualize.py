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
axa = ac_fig.gca(projection='3d')
axa.plot(xaccel, yaccel, range(len(xpos)))
title('Acceleration plot')
pos_fig = figure()
axp = pos_fig.gca(projection='3d')
axp.plot(xpos, ypos, range(len(xpos)))
title('Position plot')
vpos_fig = figure()
axv = vpos_fig.gca(projection='3d')
axv.plot(cumsum(xaccel), cumsum(yaccel), range(len(yaccel)))
title('Velocity plot')
show()
