#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECGR 4090: Real-Time AI
Author: Nathan Hewitt
Homework 3, part 3

Description: This file reads images from a webcam, uses YOLOv5 to identify
people in the image, and then passes the people into a ResNet18 TensorRT
model to perform pose estimation on the people. This pipeline is adapted
from my work for Homework 1, part 3.
"""

import sys
import torch
from time import time
from jetcam.usb_camera import USBCamera
from jetcam.utils import bgr8_to_jpeg

import json
import trt_pose.coco, trt_pose.models
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
from torch2trt import torch2trt, TRTModule

import cv2
from torchvision import transforms
from PIL import Image

def processFrame(frame, yolo, resnet):
    detections = runYolo(frame, yolo)

    bboxes = getPeople(detections)

    for b in bboxes:
        # Grab pixels for bbox
        personImg = ...

        img = runResNet(personImg, resnet)
        combineFrame(frame, b, img)

def runResNet(image, resnet):
    # Transform input image into ResNet size
    # Run ResNet on image
    # Draw joints
    # Transform output image back to input size
    # Return image
    return image

def combineFrame(frame, bbox, image):
    return

def runYolo(frame, yolo):
    # Run yolo and return List of detections
    ...

def getPeople(detections):
    # Pull the people out of the detections from YOLO
    # Return a List of their bboxes
    ...

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
