#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 10:13:22 2021

@author: yan
"""

import pickle
import cv2
import matplotlib.pyplot as plt

path = 'block-insertion-test/'
folder = path + 'color/'
pcl_file = '000000-1'


with open(folder + pcl_file + '.pkl', 'rb') as f:
    data = pickle.load(f)
#
#for i in range(2):
#    for j in range(3):
#        plt.figure(str(i)+str(j))
#        plt.imshow(data[i,j,:,:,:])
