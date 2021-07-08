#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 14:53:15 2021

@author: yan
"""



import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import struct
import math
from os.path import join

data_dir = 'block-insertion-saved'
folder = 'reward'

file = '000000-1'

with open(join(data_dir, folder, file + '.pkl'), 'rb') as f:
    data = pickle.load(f)
