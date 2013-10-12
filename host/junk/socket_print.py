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
        while 1:
            try:
                self.data = self.request.recv(256)
                tokenfile.write(self.data)
                print self.data
                if 'A\x00C\x00K\x00R' in self.data:
                    print 'Starting transmission'
                if 'E\x00N\x00D\x00T' in self.data:
                    print 'Transmission complete. Breaking TCP'
                    tokenfile.close()
                    return
            except KeyboardInterrupt:
                tokenfile.close()
                print 'Transmission halted'
                return

if __name__ == '__main__':
    HOST = '192.168.151.1'
    PORT = 1991
    server = SocketServer.TCPServer((HOST, PORT), MyTCPHandler)
    print 'TCP socket listening at host %s and port %d'%(HOST, PORT)
    server.serve_forever()
    server.close()
