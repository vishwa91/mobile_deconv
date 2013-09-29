#!/usr/bin/python

from scipy import *
from scipy.ndimage import *
from StringIO import StringIO
import Image

stim = '\x00S\x00T\x00I\x00M\x00\n'
edim = '\x00E\x00D\x00I\x00M\x00\n'

dfile = open('tokens.dat')
dstring = dfile.read()

stim_index = dstring.index(stim)
edim_index = dstring.index(edim)

imstring = dstring[stim_index+len(stim)+1:edim_index-1]
print len(imstring)
im = Image.open(StringIO(imstring[:-2]))
im.save('test_im.bmp')
