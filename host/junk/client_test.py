#!/bin/python

# Code for testing a client connection.

import socket
import sys

HOST, PORT = '10.21.3.191', 1991

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    sock.connect((HOST, PORT))
    sock.sendall(' '.join(sys.argv[1:]))
    received = sock.recv(1024)
finally:
    sock.close()

print received