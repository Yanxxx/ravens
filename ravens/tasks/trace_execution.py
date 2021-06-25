#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 11:08:30 2021

@author: Yan Li
"""

# coding=utf-8
# Copyright 2021 The Yan Li, UTK, Knoxville, TN, 37996. 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trace following motion."""

import numpy as np
from ravens.utils import utils

class Trace():
  """Trace tracking method."""
    
  def __init__(self, height=0.32, speed=0.01):
    self.height, self.speed = height, speed

  def __call__(self, movej, movep, ee, action):
    """Execute trace tracking.

    Args:
      movej: function to move robot joints.
      movep: function to move robot end effector pose.
      ee: robot end effector.
      action: SE(3) pose.

    Returns:
      timeout: robot movement timed out if True.
    """
    pose = action['pose']
    grasp = action['grasp']
    
    delta = (np.float32([0, 0, -0.001]),
             utils.eulerXYZ_to_quatXYZW((0, 0, 0)))    
    timeout = movep(pose)
    targ_pose = pose
    
    if grasp[0] == 1:
        while not ee.detect_contact():  # and target_pose[2] > 0:
          targ_pose = utils.multiply(targ_pose, delta)
          timeout |= movep(targ_pose)
          if timeout:
            return True, [0, 0, 0]
        if grasp[1] == 1:
            ee.activate()
        else:
            ee.release()
    
    return timeout, targ_pose
    
    