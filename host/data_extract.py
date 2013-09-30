#!/usr/bin/python

from scipy import *
from scipy.ndimage import *
from StringIO import StringIO
import Image
import os
import sys

im_shake_file = 'test_im_shake.bmp'
im_noshake_file = 'test_im_hoshake.bmp'
acfile = 'test_ac.dat'
outputdir = 'output'

try:
    os.mkdir(outputdir)
except OSError:
    pass

stim = '\x00S\x00T\x00I\x00M\x00\n'
edim = '\x00E\x00D\x00I\x00M\x00\n'
stac = '\x00S\x00T\x00A\x00C'
edac = '\x00E\x00D\x00A\x00C'

dfile = open('tokens.dat')
dstring = dfile.read()

stim_index = dstring.index(stim)
edim_index = dstring.index(edim)

stac_index = dstring.index(stac)
edac_index = dstring.index(edac)

acstring = dstring[stac_index+len(stac)+1:edac_index-1]
acstring = acstring.replace('\x00', '')

acvals = acstring.split(';;')
afile = open(os.path.join(outputdir, acfile), 'w')
for ac in acvals:
    afile.write(ac.replace(';', ' ')+'\n')

imstring = dstring[stim_index+len(stim)+1:edim_index-1]
print len(imstring)
im = Image.open(StringIO(imstring[:-2]))
if sys.argv[1] == 'shake':
    im.save(os.path.join(outputdir, im_shake_file))
elif sys.argv[1] == 'noshake':
    im.save(os.path.join(outputdir, im_noshake_file))
dfile.close()
afile.close()

