#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import numpy as np
import cv2

parser = argparse.ArgumentParser(description='Red ball tracking using Mac camera')
parser.add_argument('--save', action='store_true', help="Save captured images")
args = parser.parse_args()

# Initialize the video capture for the default camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

def detect_red_ball(image):
    # Convert image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define red color range in HSV
    lower_red1 = np.array([0, 150, 150])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 150, 150])
    upper_red2 = np.array([180, 255, 255])


    # Create masks for red
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    direction = None
    # Get the center of the image
    image_center = (image.shape[1] // 2, image.shape[0] // 2)

    for contour in contours:
        if cv2.contourArea(contour) > 500:  # adjust threshold as needed
            x, y, w, h = cv2.boundingRect(contour)
            ball_center = (x + w // 2, y + h // 2)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, "Red Ball", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Compute the difference between the ball center and the image center
            dx = ball_center[0] - image_center[0]
            dy = ball_center[1] - image_center[1]

            # Determine the most significant direction to move:
            # If horizontal difference is larger, move left or right; otherwise up or down.
            if abs(dx) > abs(dy):
                direction = "left" if dx < 0 else "right"
            else:
                direction = "up" if dy < 0 else "down"
            break

    return image, direction

start = time.time()
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    count += 1
    meanTimePerImage = (time.time() - start) / count
    fps = 1 / meanTimePerImage
    print("{:.2f} FPS".format(fps))

    # Process the frame to detect the red ball and determine the direction
    processed_frame, direction = detect_red_ball(frame)
    if direction:
        print("Direction:", direction)

    cv2.imshow('Red Ball Tracking', processed_frame)

    # Optionally save the frame if requested
    if args.save:
        cv2.imwrite(f"img_{count:06d}.png", processed_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
