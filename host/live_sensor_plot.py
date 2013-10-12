#!/usr/bin/env python

import socket
from scipy import *
from matplotlib.pyplot import *

# Global constants
TCP_IP = '192.168.151.1'
TCP_PORT = 1991
BUFFER_SIZE = 1024
WINDOW_SIZE = 100
TIME_STEP = 20e-3
G = 9.8

# Activate live plotting
dummy = [0]*WINDOW_SIZE
ion()
# Create the plot lines
subplot(3,1,1)
linex, = plot(dummy)
xlabel('Time ( ms)')
ylabel('m/s^2')
title('X acceleration')
ylim([-2,2])

subplot(3,1,2)
liney, = plot(dummy)
xlabel('Time ( ms)')
ylabel('m/s^2')
title('Y acceleration')
ylim([-2,2])

subplot(3,1,3)
linez, = plot(dummy)
xlabel('Time ( ms)')
ylabel('m/s^2')
title('Z acceleration')
ylim([-2,2])

# Wait till you get a socket client
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print 'Waiting for a client'
s.bind((TCP_IP, TCP_PORT))
s.listen(1)
conn, addr = s.accept()

print 'Connection address:', addr

start_plot = False
enough_data = False
xaccel = []
yaccel = []
zaccel = []
time = []
current_time = 0

while 1:
    data = conn.recv(BUFFER_SIZE).replace('\x00', '')
    if 'EDLG' in data:
        break
    if 'STLG' in data:
        start_plot = True
        print 'Starting sensor log.'
    if start_plot:
        ac_tokens = data.split(';;')
        for i in ac_tokens[:-1]:
            try:
                acx, acy, acz = i.split(';')
                xaccel.append(float(acx))
                yaccel.append(float(acy))
                zaccel.append(float(acz))
                time.append(current_time)
                current_time += TIME_STEP
            except ValueError:
                print i
                print 'Not valid data. Skipping'
        if len(xaccel) > WINDOW_SIZE:
            enough_data = True
                
    if enough_data:
        linex.set_xdata(range(WINDOW_SIZE))
        liney.set_xdata(range(WINDOW_SIZE))
        linez.set_xdata(range(WINDOW_SIZE))

        linex.set_ydata(xaccel[-WINDOW_SIZE:])
        liney.set_ydata(yaccel[-WINDOW_SIZE:])
        linez.set_ydata(zaccel[-WINDOW_SIZE:])
        draw()
    
conn.close()
print 'Closed connection'
