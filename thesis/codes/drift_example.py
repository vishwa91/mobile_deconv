#!/usr/bin/python

from scipy import *
from scipy.special import *
from scipy.integrate import *
import numpy as np

from matplotlib.pyplot import *
G =9.8; T=10e-3;
if __name__ == '__main__':
	# Create position by low pass filtering random data.
	theta = linspace(0,10*pi, 50)
	time = linspace(0, 1, 50)
	random_data = random.random(52)
	posmax = abs(random_data).max()
	accel = diff(diff(random_data))

	subplot(2,1,1); plot(cumtrapz(cumtrapz(accel))*G*T*T)
	ylabel('meters'); xlabel('samples')
	title('Actual trajectory')
	# Create drifts
	for drift in linspace(-0.01*posmax, 0.01*posmax, 9):
		temp_accel = accel + drift
		subplot(2,1,2)
		plot(cumtrapz(cumtrapz(temp_accel))*G*T*T, label='Drift=%f'%drift)
	subplot(2,1,2); title('Trajectories with drif in acceleration')
	ylabel('meters'); xlabel('samples')
	legend(loc='upper left', prop={'size':8})
	savefig('drift_image.png', dpi=250)
	show()