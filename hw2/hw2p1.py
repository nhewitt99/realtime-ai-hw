#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECGR 4090: Real-Time AI
Author: Nathan Hewitt
Homework 2, part 1

Description: Some simple tensor operations
"""

import torch

def main():
    temp = torch.tensor(list(range(9)))
    print(temp)

    # Try to apply cos and sqrt to temp
    try:
        temp.cos()
        temp.sqrt()
    except Exception as e:
        print(f'When trying to apply cos and sqrt, an error occurred:\n    {e}')

    # Change temp's datatype to allow cos and sqrt
    temp = temp.float()

    cos = temp.cos()
    sqrt = temp.sqrt()
    print(f'Element-wise cos on temp yields:\n    {cos}')
    print(f'Element-wise sqrt on temp yields:\n    {sqrt}')

    # Demonstrate in-place operations
    temp.sqrt_()
    temp.cos_()
    print(f'After in-place sqrt and cos, temp is:\n    {temp}')

if __name__=='__main__':
    main()

