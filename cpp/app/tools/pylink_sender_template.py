#!/usr/bin/env python3
# pylink_sender_template.py
# (template that forwards EyeLink samples through pylink, untested, clearly marked)
# UNTESTED TEMPLATE for forwarding pylink samples to EyeLive via UDP v1

import socket
import struct
import time
import sys

# import pylink # Uncomment this if you have the EyeLink SDK

MAGIC = 0x31455945
HOST = "127.0.0.1"
PORT = 5005

def main():
    print("This is an untested template for forwarding EyeLink data.")
    print("Do not use unless pylink is installed and configured.")
    sys.exit(0)

    # el = pylink.EyeLink("100.1.1.1")
    # el.openDataFile("test.edf")
    # el.startRecording(1,1,1,1)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # while True:
    #     dt = el.getNextData()
    #     if dt == pylink.SAMPLE_TYPE:
    #         sample = el.getFloatData()
    #         t = sample.getTime() / 1000.0
    #         if sample.isRightSample():
    #             x, y = sample.getRightEye().getGaze()
    #             flags = 1
    #             # You can add blink check based on logic, e.g., flags = 2 for blink
    #             data = struct.pack("<IIdffI", MAGIC, 1, t, float(x), float(y), flags)
    #             sock.sendto(data, (HOST, PORT))

if __name__ == "__main__":
    main()
