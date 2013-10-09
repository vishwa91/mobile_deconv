#!/usr/env python

from scipy import *
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D
data = open('sensor_log.dat').read()

timestep = 20e-3
g = 9.8
windowsize = 65

tokens = data.split('\n')
sensor_string = tokens[tokens.index('STLG')+1]
xaccel, yaccel, zaccel = [], [], []
sensor_tokens = sensor_string.split(';;')

for token in sensor_tokens[:-1]:
    fields = token.split(';')
    if fields != '':
        xaccel.append(round(float(fields[0]), 2))
        yaccel.append(round(float(fields[1]), 2))
        zaccel.append(round(float(fields[2]), 2))
figure(-1)
plot(xaccel-mean(xaccel))

# Subtract the average.
mx, my, mz = mean(xaccel), mean(yaccel), mean(zaccel)
time = arange(0, timestep*len(xaccel), timestep)
snaptime = arange(0, timestep*windowsize, timestep)
# plot the overall curve
figure(0)
plot(time, cumsum(cumsum(xaccel-mx))*g*timestep*timestep, 'r', label='xpos')
plot(time, cumsum(cumsum(yaccel-my))*g*timestep*timestep, 'k', label='ypos')
plot(time, cumsum(cumsum(zaccel-mz))*g*timestep*timestep, 'b', label='zpos')
legend()

# We wish to check if we can predict the drift over a time of windowsize samples
xaccel = array(xaccel[:-(len(xaccel)%windowsize)])
yaccel = array(yaccel[:-(len(yaccel)%windowsize)])
zaccel = array(zaccel[:-(len(zaccel)%windowsize)])

snapx = xaccel.reshape((len(xaccel)//windowsize, windowsize))
snapy = yaccel.reshape((len(yaccel)//windowsize, windowsize))
snapz = zaccel.reshape((len(zaccel)//windowsize, windowsize))

for i in range(snapx.shape[0]):
    figure(1)
    plot(snaptime, cumsum(cumsum(snapx[i]-mean(snapx[i])))*g*timestep*timestep)
    #plot(snaptime, snapx[i]*g)
    title('X position')
    figure(2)
    plot(snaptime, cumsum(cumsum(snapy[i]-mean(snapy[i])))*g*timestep*timestep)
    #plot(snaptime, snapy[i]*g)
    title('Y position')
    figure(3)
    plot(snaptime, cumsum(cumsum(snapz[i]-mean(snapz[i])))*g*timestep*timestep)
    #plot(snaptime, snapz[i]*g)
    title('Z position')
    
posfig = figure()
ax = posfig.gca(projection = '3d')
ax.plot(cumsum(cumsum(xaccel)), cumsum(cumsum(yaccel)),
        range(len(xaccel)))
show()
