#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 00:12:10 2021

@author: yan
"""

# coding=utf-8
# Copyright 2021, Yan Li, UTK, Knoxville, TN, 37996.
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

"""Sweeping task."""

import numpy as np
from ravens.tasks import primitives
from ravens.tasks.grippers import Bucket
from ravens.tasks.task import Task
from ravens.utils import utils

import pybullet as p

class AutoExcavation(Task):
  """Sweeping task."""

  def __init__(self):
    super().__init__()
    self.ee = Bucket
    self.max_steps = 20
    self.primitive = primitives.push

  def reset(self, env):
    super().reset(env)

    # Add goal zone.
    blocks = {}
    zone_size = (0.12, 0.12, 0)
#    tray_size = (0.5,0,-0.65)
    tray_pose = ([0.5 , 0 , 0], [0,0,0,1])#self.get_random_pose(env, zone_size)
    env.add_object("tray/traybox.urdf", tray_pose, 'fixed',scale=0.25)
#    env.add_object("container/container-template.urdf", tray_pose, 'fixed')

    # Add pile of small blocks.
    obj_pts = {}
    obj_ids = []
    obj_num = 60
    rc = 5
    for count in range(obj_num):
#      rx = self.bounds[0, 0] + 0.15 + np.random.rand() * 0.2
#      ry = self.bounds[1, 0] + 0.40 + np.random.rand() * 0.2
#      xyz = (rx, ry, 0.02)
      
      rx = self.bounds[0, 0] + 0.2 + (count % rc) * 0.01
      ry = self.bounds[1, 0] + 0.5 + ((count % rc**2) // rc) * 0.01
      rz = 0.02 * (count // (rc ** 2))
      xyz = (rx, ry, rz)
      theta = np.random.rand() * 2 * np.pi
      xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
      obj_id = env.add_object('sphere/sphere.urdf', (xyz, xyzw)) #block/small.urdf
#      obj_id = env.add_object('block/small.urdf', (xyz, xyzw)) #block/small.urdf
#      obj_id = env.add_object('box/box-template.urdf', (xyz, xyzw)) #block/small.urdf
      obj_pts[obj_id] = self.get_object_points(obj_id)
      blocks[obj_id] = [xyz, xyzw]
      obj_ids.append((obj_id, (0, None)))

    # Goal: all small blocks must be in zone.
    # goal = Goal(list(obj_pts.keys()), [0] * len(obj_pts), [zone_pose])
    # metric = Metric('zone', (obj_pts, [(zone_pose, zone_size)]), 1.)
    # self.goals.append((goal, metric))
    self.goals.append((obj_ids, np.ones((obj_num, 1)), [tray_pose], True, False,
                       'zone', (obj_pts, [(tray_pose, zone_size)]), 1))
    
    for i in range(10):
#      print(i)
      p.stepSimulation()
    print('auto excavation finished.')
    return blocks, tray_pose
