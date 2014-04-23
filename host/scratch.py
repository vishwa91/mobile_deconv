#!/usr/bin/env python

from modules.methods import *
from modules.tcp import *
from modules.kernel import *
from modules.depth import *

if __name__ == '__main__':
    strt_token = 'S\x00T\x00I\x00G\x00'
    end_token = 'E\x00D\x00I\x00G\x00'
    frame_token = 'S\x00T\x00I\x00M\x00'
    fstart = 'STIM'
    fend = 'EDIM'
    continuous_recv(strt_token, end_token, frame_token, 'tmp/burst/tokens.dat')
    extract_images('tmp/burst/tokens.dat', 100, fstart, fend, 'tmp/burst/src')