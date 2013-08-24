#!/bin/python

# Code for creating a TCP server on the PC

import SocketServer

class MyTCPHandler(SocketServer.BaseRequestHandler):
    """
        Base service handler. Just a simple echo service.
    """
    def handle(self):
        self.data = self.request.recv(1024).strip()
        print self.data
        self.request.sendall(self.data)

if __name__ == '__main__':
    HOST = 'localhost'
    PORT = 23
    server = SocketServer.TCPServer((HOST, PORT), MyTCPHandler)
    print 'Got server. Starting service.'
    server.serve_forever()
