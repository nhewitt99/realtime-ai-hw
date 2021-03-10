#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECGR 4090: Real-Time AI
Author: Nathan Hewitt
Homework 1, part 3

Description: This file contains code to get frames from a USB camera connected to the Nano,
and then processing them with yolov5s. The only argument specifies batch size.

I decided to write this from scratch since there was a lot of unneeded functionality in
yolo's detect.py demo.
"""

import sys
import torch
from time import time
from jetcam.usb_camera import USBCamera


def main():
    # Get batch size from args
    batchSize = int(sys.argv[1])

    # Set up USB camera (assuming device 0)
    cam = USBCamera(width=1920, height=1080, capture_device=0)

    # Set up model for yolov5s
    yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    device = torch.device('cuda')
    yolo.to(device)

    # Basic analytics
    imageCount = 0
    modelTime = 0

    # Continue until interrupt
    try:
        while True:
            # Grab as many frames as batch size
            imgs = []
            for i in range(batchSize):
                imgs.append(cam.read()[:, :, ::-1]) # Convert BGR to RGB

            # Process with yolo
            t = time()
            results = yolo(imgs)
            modelTime += time() - t

            # Save files, janky method to change filenames and avoid overwrite
            for i,f in enumerate(results.files):
                newNumber = sum(map(int, filter(str.isdigit, f)), imageCount)
                newStr = str(newNumber).zfill(4) + '.jpg'
                results.files[i] = newStr
            results.save()
            imageCount += batchSize

    except KeyboardInterrupt:
        print(f'Ending. Processed {imageCount} images in {modelTime}s, average FPS of {imageCount/modelTime}')

if __name__=='__main__':
    main()
