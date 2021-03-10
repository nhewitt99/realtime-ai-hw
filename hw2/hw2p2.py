#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECGR 4090: Real-Time AI
Author: Nathan Hewitt
Homework 2, part 2

Description: Some simple image operations with tensors
"""

import torch
from PIL import Image
import glob
from torchvision import transforms

def main():
    # Import images
    imgs = []
    for filename in glob.glob('img/*'):
        img = Image.open(filename)

        t = transforms.ToTensor()(img)

        mean = t.mean()
        red = t[0,:,:].mean()
        grn = t[1,:,:].mean()
        blu = t[2,:,:].mean()

        color = 'Red'
        if grn > red and grn > blu:
            color = 'Green'
        if blu > red and blu > grn:
            color = 'Blue'

        print(f'Image {filename} processed.')
        print(f'    Brightness: {mean}')
        print(f'    Red Mean:   {red}')
        print(f'    Green Mean: {grn}')
        print(f'    Blue Mean:  {blu}')
        print(f'    This image is predicted to be {color}.')

if __name__=='__main__':
    main()

