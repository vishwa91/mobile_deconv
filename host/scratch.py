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
	dummy_recv(start_token, end_token, frame_token, 'tmp/focus/tokens.dat')
	extract_images('tmp/focus/tokens.dat', 100, fstart, fend, 'tmp/focus')