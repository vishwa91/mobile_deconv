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
        Base service handler. Just a simple echo service.
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
            token = self.data.replace('\x00','')
            tokenfile.write(token)
            if 'STRT' in token:
                print 'Established connection'
            elif 'ACKR' in token:
                print 'Acknowledged'
            elif 'ENDT' in token:
                print 'Done transmission'
                imdat += token.replace('ENDT', '')
                tokenfile.write('done\n')
                tokenfile.write(token)
                tokenfile.close()
                break
            elif 'SIZE' in token:
                imsize = int(token.split(':')[-1])
                bufsize = 1024
                print 'getting an image next'
                nchunks = imsize/bufsize
                continue
            elif 'HGHT' in token:
                imh = int(token.split(':')[-1])
            elif 'WDTH' in token:
                imw = int(token.split(':')[-1])
            elif 'STIM' in token:
                nextimage = True
                continue
            elif nextimage:
                imdat += token
                print 'Got chunk number %d. Expecting %d chunks'%(chunk_number,
                                                                  nchunks)
                chunk_number += 1
                receive_size += len(token)
            else:
                print token
                
            global_data.append(token)
            self.request.sendall(token)

if __name__ == '__main__':
    HOST = '10.22.43.192'
    PORT = 1991
    server = SocketServer.TCPServer((HOST, PORT), MyTCPHandler)
    print 'TCP socket listening at host %s and port %d'%(HOST, PORT)
    server.serve_forever()
    server.close()
