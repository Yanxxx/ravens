#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 10:13:22 2021

@author: yan
"""

import pickle
import cv2
import matplotlib.pyplot as plt
from matplotlib import animation
from time import sleep

path = 'block-insertion-test/'
folder = path + 'ee/'
#folder = path + 'object_pos/'
#folder = path + 'targ_pos/'
pcl_file = '000000-1'


with open(folder + pcl_file + '.pkl', 'rb') as f:
    data = pickle.load(f)
    
#plt.figure()
#im = plt.imshow(data[0,0, :,:])

#for i in range(data.shape[0]):
##    plt.imshow(data[i,0,:,:,:])
##    plt.show()
##    sleep(0.05)
#    im.set_data(data[i,0,:,:,:])
#    sleep(0.005)


#
#for i in range(2):
#    for j in range(3):
#        plt.figure(str(i)+str(j))
#        plt.imshow(data[i,j,:,:,:])
