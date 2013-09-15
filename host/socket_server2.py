#!/bin/python

# Code for creating a TCP server on the PC

import SocketServer
import Image
global_data = []
imsize = 0
imdat = ''
tokenfile = open('tokens.dat', 'w')
imh, imw = (0,0)
class MyTCPHandler(SocketServer.BaseRequestHandler):
    """
        TCP service handler.
    """
    def handle(self):
        global imsize, imdat, tokenfile, imh, imw
        nextimage = False
        bufsize = 1024
        imdat = ''
        receive_size = 0
        chunk_number = 0
        nchunks = 0
        while 1:
            self.data = self.request.recv(bufsize)
            tokenstring = self.data.replace('\x00','').replace('null', chr(0))
            tokenfile.write(tokenstring)
            tokens = tokenstring.split('\n')
            imdat += tokenstring
            if 'ENDT' in tokens:
                print 'Transmission complete'
                return 0            

if __name__ == '__main__':
    HOST = '10.22.43.192'
    PORT = 1991
    server = SocketServer.TCPServer((HOST, PORT), MyTCPHandler)
    print 'TCP socket listening at host %s and port %d'%(HOST, PORT)
    server.serve_forever()
    server.close()
