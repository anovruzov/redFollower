#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import socket, os, struct
import numpy as np
import cv2

parser = argparse.ArgumentParser(description='Connect to AI-deck JPEG streamer example')
parser.add_argument("-n",  default="192.168.4.1", metavar="ip", help="AI-deck IP")
parser.add_argument("-p", type=int, default='5000', metavar="port", help="AI-deck port")
parser.add_argument('--save', action='store_true', help="Save streamed images")
args = parser.parse_args()

deck_port = args.p
deck_ip = args.n

print("Connecting to socket on {}:{}...".format(deck_ip, deck_port))
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((deck_ip, deck_port))
print("Socket connected")

def rx_bytes(size):
    data = bytearray()
    while len(data) < size:
        data.extend(client_socket.recv(size - len(data)))
    return data

def detect_red_ball(image):
    # Convert image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define red color range in HSV
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 50])
    upper_red2 = np.array([179, 255, 255])

    # Create masks for red
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 500:  # adjust threshold as needed
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, "Red Ball", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return image

start = time.time()
count = 0

while True:
    packetInfoRaw = rx_bytes(4)
    [length, routing, function] = struct.unpack('<HBB', packetInfoRaw)
    imgHeader = rx_bytes(length - 2)
    [magic, width, height, depth, format, size] = struct.unpack('<BHHBBI', imgHeader)

    if magic == 0xBC:
        imgStream = bytearray()

        while len(imgStream) < size:
            packetInfoRaw = rx_bytes(4)
            [length, dst, src] = struct.unpack('<HBB', packetInfoRaw)
            chunk = rx_bytes(length - 2)
            imgStream.extend(chunk)

        count += 1
        meanTimePerImage = (time.time() - start) / count
        print("{:.2f} FPS".format(1 / meanTimePerImage))

        if format == 0:
            bayer_img = np.frombuffer(imgStream, dtype=np.uint8)
            bayer_img.shape = (244, 324)
            color_img = cv2.cvtColor(bayer_img, cv2.COLOR_BayerBG2BGR)
            color_img = detect_red_ball(color_img)
            cv2.imshow('Color', color_img)
            if args.save:
                cv2.imwrite(f"stream_out/raw/img_{count:06d}.png", bayer_img)
                cv2.imwrite(f"stream_out/debayer/img_{count:06d}.png", color_img)
            cv2.waitKey(1)
        else:
            nparr = np.frombuffer(imgStream, np.uint8)
            decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            decoded = detect_red_ball(decoded)
            cv2.imshow('JPEG', decoded)
            cv2.waitKey(1)
