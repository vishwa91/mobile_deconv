#!/bin/python

# Code for creating a TCP server on the PC

import SocketServer
import Image
from StringIO import StringIO
imdat = ''
tokenfile = open('tokens.dat', 'w')
class MyTCPHandler(SocketServer.BaseRequestHandler):
    """
        TCP service handler.
    """
    def handle(self):
        global imsize, imdat, tokenfile, imh, imw
        nextimage = False
        bufsize = 64
        imdat = ''
        receive_size = 0
        chunk_number = 0
        nchunks = 0
        while 1:
            self.data = self.request.recv(bufsize)
            tokenstring = self.data.replace('\x00','')
            print tokenstring
            tokenfile.write(tokenstring)
            imdat += tokenstring
            if 'STRT' in tokenstring:
                print 'Started transmission.'
            if 'ENDT' in tokenstring:
                print 'Transmission complete'
                print 'Will start image processing now.'
                process()
                return 0
            
def process():
    """ Routine to process the incoming data."""
    global imdat
    tokens = imdat.split('\n')
    imstart = tokens.index('STIM')
    imtoken = tokens[imstart+1]
    imtokens = imtoken.split(';')
    imtokens = [chr(int(i)) for i in imtokens[:-1]]
    imstring = ''
    for i in imtokens:
        imstring += i
    im = Image.open(StringIO(imstring))
    im.save('test_im.bmp')
    acstart = tokens.index('STAC')
    actoken = tokens[acstart+1]
    actokens = actoken[:-1].replace(';;', '\n').replace(';', ',')
    acfile = open('test_ac.dat', 'w')
    acfile.write(actokens)
    acfile.close()
    print 'Competed processing'
if __name__ == '__main__':
    #HOST = '10.22.43.192'
    HOST = '192.168.151.1'
    PORT = 1992
    server = SocketServer.TCPServer((HOST, PORT), MyTCPHandler)
    print 'TCP socket listening at host %s and port %d'%(HOST, PORT)
    server.serve_forever()
    server.close()
