#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECGR 4090: Real-Time AI
Author: Nathan Hewitt
Homework 3, part 3

Description: This file reads images from a webcam, uses YOLOv5 to identify
people in the image, and then passes the people into a ResNet18 TensorRT
model to perform pose estimation on the people. The main() fn is adapted
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
    # Get detections from YOLO
    detections = runYolo(frame, yolo)
    
    # Save the YOLO output frame
    frame = detections.render()[-1].copy()

    # Find bboxes of people by checking the Detection class' label
    bboxes = getPeople(detections)

    if bboxes:
        print(f'Found {len(bboxes)} people.')
        for b in bboxes:
            # Grab pixels from YOLO image for bbox
            personImg = frame[b[1]:b[3], b[0]:b[2]]

            # Render person's joints with ResNet and stack onto YOLO frame
            img = runResNet(personImg, resnet, parseObjects, drawObjects)
            frame = combineFrame(frame, b, img)
            
    # Convert back to BGR on return so cv.imwrite will work
    return frame[:, :, ::-1], (bboxes is None)


def runResNet(image, resnet, parseObjects, drawObjects):
    # Preprocess according to demo
    data = preprocessResNet(image)

    # Run ResNet on image
    cmap, paf = resnet(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parseObjects(cmap, paf)
    
    # Draw joints
    drawObjects(image, counts, objects, peaks)

    return image


# Image preprocessing function from trt_pose demo notebook
def preprocessResNet(image):
    mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
    std = torch.Tensor([0.299, 0.224, 0.225]).cuda()
    device = torch.device('cuda')

    image = Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    #image.sub_(mean[:,None,None]).div_(std[:,None,None])
    image = transforms.functional.resize(image, [224, 224])
    return image[None, ...]


def combineFrame(frame, bbox, image):
    # Just replace pixels in frame of bounding box with image
    frame[bbox[1]:bbox[3], bbox[0]:bbox[2]] = image
    return frame


def runYolo(frame, yolo):
    # Run yolo and return detections. In hindsight, this didn't need a function.
    return yolo([frame])


def getPeople(results):
    # Pull the people out of the detections from YOLO
    predictions = results.pred[-1]

    # List comprehension to grab people, then those people's boxes
    people = [p for p in predictions if p[-1] == 0] # 0 is label for Person
    bboxes = [p[0:4].tolist() for p in people]

    # List comp to round everything to int
    bboxes = [[round(i) for i in box] for box in bboxes]
    return bboxes


def main():
    liveDemo = False
    
    if liveDemo:
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
    resnet.load_state_dict(torch.load('resnet_trt.pth'))

    parseObjects = ParseObjects(topology)
    drawObjects = DrawObjects(topology)

    # Basic analytics
    imageCount = 0
    t = time()

    # Live demo on webcam
    if liveDemo:
        # Continue until interrupt
        try:
            while True:
                # Grab a frame
                img = cam.read()[:, :, ::-1] # Convert BGR to RGB
                print(f'got frame {imageCount}')

                # Process with yolo and resnet
                result, empty = processFrame(img, yolo, resnet, parseObjects, drawObjects)

                # Save file
                cv2.imwrite(f'imgs/{imageCount:04}.jpg', result)

                imageCount += 1

        except KeyboardInterrupt:
            print('Keyboard interrupt!')
        finally:
            t = time() - t
    # Recorded video
    else:
        cap = cv2.VideoCapture('example_video.mpg')
        
        # Grab a frame
        ret, frame = cap.read()
        
        # Continue until video is done
        while ret:
            # Process and save image
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result, empty = processFrame(img, yolo, resnet, parseObjects, drawObjects)
            cv2.imwrite(f'imgs/{imageCount:04}.jpg', result)
            
            # Try to grab next frame
            ret, frame = cap.read()
            imageCount += 1
            
        t = time() - t
        cap.release()
        
    print(f'Ending. Processed {imageCount} images in {t}s, average FPS of {imageCount/t}')
    # 249 in 116.69, average fps of 2.13
    
    # Note that unlike in part 2, this program just outputs frames rather than a video.
    # This is because for the live demo, I need to manually match the framerate to the
    # posterior-calculated framerate, which isn't possible when initializing an openCV
    # VideoWriter. Instead, I combine them into a video with ffmpeg.
            
if __name__=='__main__':
    main()
