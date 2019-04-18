#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: murzinv
"""

import zmq
from time import sleep

port = "6400"
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:%s" % port)
socket.setsockopt(zmq.LINGER, 1000)

n=5
while True:
    n = n%7 + 1
    with open("./CuSO4_01M_flow_000%d.dat"%n, "r") as f:
        socket.send("test")
        sleep(0.5)
        socket.send("clear")
        print f.name
        sleep(0.5)
        socket.send(f.name)
        f.readlines(15)
        line = f.readline()
        while line:
            splitted = line.split()
            message = " ".join([splitted[0], splitted[5], splitted[6], splitted[7], splitted[8]])
            socket.send(message)
            line = f.readline()
            sleep(0.05)
