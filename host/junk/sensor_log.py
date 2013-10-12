#!/bin/python

# Code for creating a TCP server on the PC

import SocketServer
import Image
global_data = []
imsize = 0
imdat = ''
tokenfile = open('sensor_log.dat', 'w')
imh, imw = (0,0)
class MyTCPHandler(SocketServer.BaseRequestHandler):
    """
        Base service handler. Logs data
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
            print token
            if 'STLG' in token:
                print 'Startin sensor log'
            if 'EDLG' in token:
                print 'Sensor log complete'
                tokenfile.close()
                self.finish()
                return            

if __name__ == '__main__':
    HOST = '192.168.151.1'
    PORT = 1991
    server = SocketServer.TCPServer((HOST, PORT), MyTCPHandler)
    print 'TCP socket listening at host %s and port %d'%(HOST, PORT)
    server.serve_forever()
    server.close()
