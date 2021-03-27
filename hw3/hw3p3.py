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

def processFrame(frame, yolo, resnet, parseObjects, drawObjects):
    detections = runYolo(frame, yolo)
    frame = detections.render()[-1].copy()

    bboxes = getPeople(detections)

    if bboxes:
        print(f'Found {len(bboxes)} people')
        for b in bboxes:
            # Grab pixels for bbox
            personImg = frame[b[0]:b[2], b[1]:b[3]]

            img = runResNet(cv2.UMat(personImg), resnet, parseObjects, drawObjects)
            img.detach().cpu()
            frame = combineFrame(frame, b, img)
    return frame, (bboxes is None)


def runResNet(image, resnet, parseObjects, drawObjects):
    # Transform input image into ResNet size
    # Run ResNet on image
    # Draw joints
    # Transform output image back to input size
    # Return image
    try:
        shape = [len(image), len(image[0])]
        print(shape)
    except Exception as e:
        print(e)
    image = preprocessResNet(image)
    #cv2.imwrite('imgs/preproc.jpg', image.numpy().transpose(1, 2, 0))

    cmap, paf = resnet(image)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parseObjects(cmap, paf)
    drawObjects(image, counts, objects, peaks)

    try:
        image = transforms.functional.resize(image, shape)
    except Exception as e:
        print(e)
    #cv2.imwrite('imgs/resnet.jpg', image.numpy().transpose(1, 2, 0))

    return image


# Image preprocessing function from trt_pose demo notebook
def preprocessResNet(image):
    mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
    std = torch.Tensor([0.299, 0.224, 0.225]).cuda()
    device = torch.device('cuda')

    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:,None,None]).div_(std[:,None,None])
    image = transforms.functional.resize(image, [224, 224])
    return image[None, ...]


def combineFrame(frame, bbox, image):
    frame[bbox[0]:bbox[2], bbox[1]:bbox[3]] = image
    return frame


def runYolo(frame, yolo):
    # Run yolo and return detections
    return yolo([frame])


def getPeople(results):
    # Pull the people out of the detections from YOLO
    # Return a List of their bboxes
    predictions = results.pred[-1]

    people = [p for p in predictions if p[-1] == 0] # 0 is label for Person
    bboxes = [p[0:4].tolist() for p in people]

    # Round everything to int
    bboxes = [[round(i) for i in box] for box in bboxes]
    return bboxes


def main():
    # Set up USB camera (assuming device 0)
    cam = USBCamera(width=1920, height=1080, capture_device=0)

    # Set up model for yolov5s
    yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    device = torch.device('cuda')
    yolo.to(device)

    # Set up topology, model, and classes for ResNet
    with open('human_pose.json', 'r') as f:
        human_pose = json.load(f)
    topology = trt_pose.coco.coco_category_to_topology(human_pose)

    resnet = TRTModule()
    resnet.load_state_dict(torch.load('densenet_trt.pth'))

    parseObjects = ParseObjects(topology)
    drawObjects = DrawObjects(topology)

    # Basic analytics
    imageCount = 0
    t = time()

    # Continue until interrupt
    try:
        while True:
            # Grab as many frames as batch size
            img = cam.read()[:, :, ::-1] # Convert BGR to RGB
            print('got frame')
            # Process with yolo
            result, empty = processFrame(img, yolo, resnet, parseObjects, drawObjects)

            # Save file
            cv2.imwrite(f'imgs/{imageCount:04}.jpg', result)

            imageCount += 1

    except KeyboardInterrupt:
        print('Keyboard interrupt!')
    finally:
        t = time() - t
        print(f'Ending. Processed {imageCount} images in {t}s, average FPS of {imageCount/t}')

if __name__=='__main__':
    main()
