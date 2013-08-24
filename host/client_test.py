#!/bin/python

# Code for testing a client connection.

import socket
import sys

HOST, PORT = 'localhost', 23

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    sock.connect((HOST, PORT))
    sock.sendall(' '.join(sys.argv[1:]))
    received = sock.recv(1024)
finally:
    sock.close()

print received
