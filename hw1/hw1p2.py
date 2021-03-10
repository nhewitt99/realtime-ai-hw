#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECGR 4090: Real-Time AI
Author: Nathan Hewitt
Homework 1, part 2

Description: This file processes various batch sizes over each of the
yolov5 model sizes, outputting performance information (time).

NOTE: for some reason, this wouldn't work??? I could not track down
the issue in time, but it was inexplicably taking 30s for batch size 1
on small, then crashing. For the numbers in the report, I manually
ran tests in the python command line.
"""

import glob
import torch
import numpy
from PIL import Image
from time import time

def runBatches(imgs, model):
    # Batch size 1, then 8, 16
    t = time()
    model(imgs[0])
    t = time() - t
    print(f'Batch size 1 took {t}')

    t = time()
    model(imgs[0:8])
    t = time() - t
    print(f'Batch size 8 took {t}')

    t = time()
    model(imgs)
    t = time() - t
    print(f'Batch size 16 took {t}')


def main():
    # Get images
    imgs = []
    for filename in glob.glob('img/*'):
        img = Image.open(filename)
        imgs.append(numpy.array(img))

    # Set up model for yolov5s
    yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    device = torch.device('cuda')
    yolo.to(device) # Move to GPU

    print('Small')
    runBatches(imgs, yolo)

    # Set up m, then l, then x
    yolo = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
    yolo.to(device)
    print('Medium')
    runBatches(imgs, yolo)
    
    yolo = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
    yolo.to(device)
    print('Large')
    runBatches(imgs, yolo)
    
    yolo = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
    yolo.to(device)
    print('Extra')
    runBatches(imgs, yolo)


if __name__=='__main__':
    main()
