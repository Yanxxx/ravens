#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 11:40:32 2021

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

import numpy as np
from ravens.utils import utils

class TraceGenerator():
    """"""
    
    def __init__(self):
        self.init_pose = ([0.487, 0.109, 0.347], [0, 0, 0, 1])
#        self.init_pose = ([0.46562498807907104, -0.375, 0.3599780201911926], [0, 0, 0, 1])
#        self.prepick_to_pick = ([0, 0, 0.05], (0, 0, 0, 1))
#        self.postpick_to_pick = ([0, 0, 0.05], (0, 0, 0, 1))
#        self.preplace_to_place = ([0, 0, 0.05], (0, 0, 0, 1))
#        self.postplace_to_place = ([0, 0, 0.35], (0, 0, 0, 1))
        self.prepick_to_pick = ([0, 0, 0.35], (0, 0, 0, 1))
        self.postpick_to_pick = ([0, 0, 0.35], (0, 0, 0, 1))
        self.preplace_to_place = ([0, 0, 0.35], (0, 0, 0, 1))
        self.postplace_to_place = ([0, 0, 0.35], (0, 0, 0, 1))
        self.pick = ([0, 0, 0.32], (0, 0, 0, 1))
        self.place = ([0, 0, 0.32], (0, 0, 0, 1))
        self.trace_key = []
        self.trace_key.append({'pose': self.init_pose, 'grasp': [0, 0]})
        self.trace_key.append({'pose': self.prepick_to_pick, 'grasp': [0, 0]})
        self.trace_key.append({'pose': self.pick, 'grasp': [1, 1]})
        self.trace_key.append({'pose': self.postpick_to_pick, 'grasp': [0, 0]})
        self.trace_key.append({'pose': self.preplace_to_place, 'grasp': [0, 0]})
        self.trace_key.append({'pose': self.place, 'grasp': [1, 0]})
        self.trace_key.append({'pose': self.postplace_to_place, 'grasp': [0, 0]})
        self.step_resolution = 0.02
        
    def optim(self, action):
        init_pose = ([0.487, 0.109, 0.347], [0, 0, 0, 1])
        prepick_to_pick = ([0, 0, 0.05], (0, 0, 0, 1))
        postpick_to_pick = ([0, 0, 0.05], (0, 0, 0, 1))
        preplace_to_place = ([0, 0, 0.05], (0, 0, 0, 1))
        postplace_to_place = ([0, 0, 0.35], (0, 0, 0, 1))
        pick = ([0, 0, 0.32], (0, 0, 0, 1))
        place = ([0, 0, 0.32], (0, 0, 0, 1))
        trace_key = []
        trace_key.append({'pose': init_pose, 'grasp': [0, 0]})
        trace_key.append({'pose': prepick_to_pick, 'grasp': [0, 0]})
        trace_key.append({'pose': pick, 'grasp': [1, 1]})
        trace_key.append({'pose': postpick_to_pick, 'grasp': [0, 0]})
        trace_key.append({'pose': preplace_to_place, 'grasp': [0, 0]})
        trace_key.append({'pose': place, 'grasp': [1, 0]})
        trace_key.append({'pose': postplace_to_place, 'grasp': [0, 0]})
        step_resolution = 0.05
        result = []
        q = [0,0,0,1]
        if action:
            pick_pose = action['pose0']
            place_pose = action['pose1']
        
            
            prepick_pose = utils.multiply(pick_pose, prepick_to_pick)
            postpick_pose = utils.multiply(pick_pose, postpick_to_pick)
            preplace_pose = utils.multiply(place_pose, preplace_to_place)
            postplace_pose = utils.multiply(place_pose, postplace_to_place)
            trace_key[2]['pose'] = pick_pose
            trace_key[5]['pose'] = place_pose
            trace_key[1]['pose'] = prepick_pose
            trace_key[3]['pose'] = postpick_pose
            trace_key[4]['pose'] = preplace_pose
            trace_key[6]['pose'] = postplace_pose
            
            for i in range(len(trace_key) - 1):                
              result.append(trace_key[i])
              p1 = trace_key[i]['pose'][0]
              p2 = trace_key[i+1]['pose'][0]
#              print(p1,p2)
        #      q1 = act_tmp[i][1]
        #      q2 = act_tmp[i+1][1]
              d = np.linalg.norm(np.array(p2) - np.array(p1))
              step = int(np.round(d / step_resolution))
#              print('step=:', step)
              position = (np.array(p2) - np.array(p1))/step
              for j in range(step - 1):
        #          result.append((p1 + position * (j + 1), q1 + pose * (j + 1)))
                  q = trace_key[i + 1]['pose'][1]
                  result.append({'pose': (p1 + position * (j + 1), q), 'grasp': [0, 0]})
                  
        return result
        
    def trace(self, action = None):
        result = []
        q = [0,0,0,1]
        if action:
            pick_pose = action['pose0']
            place_pose = action['pose1']
            
            prepick_pose = utils.multiply(pick_pose, self.prepick_to_pick)
            postpick_pose = utils.multiply(pick_pose, self.postpick_to_pick)
            preplace_pose = utils.multiply(place_pose, self.preplace_to_place)
            postplace_pose = utils.multiply(place_pose, self.postplace_to_place)
            self.trace_key[2]['pose'] = pick_pose
            self.trace_key[5]['pose'] = place_pose
            self.trace_key[1]['pose'] = prepick_pose
            self.trace_key[3]['pose'] = postpick_pose
            self.trace_key[4]['pose'] = preplace_pose
            self.trace_key[6]['pose'] = postplace_pose
            
            for i in range(len(self.trace_key) - 1):                
              result.append(self.trace_key[i])
              p1 = self.trace_key[i]['pose'][0]
              p2 = self.trace_key[i+1]['pose'][0]
#              print(p1,p2)
        #      q1 = act_tmp[i][1]
        #      q2 = act_tmp[i+1][1]
              d = np.linalg.norm(np.array(p2) - np.array(p1))
              step = int(np.round(d / self.step_resolution))
#              print('step=:', step)
              position = (np.array(p2) - np.array(p1))/step
              for j in range(step - 1):
        #          result.append((p1 + position * (j + 1), q1 + pose * (j + 1)))
                  q = self.trace_key[i + 1]['pose'][1]
                  result.append({'pose': (p1 + position * (j + 1), q), 'grasp': [0, 0]})
                  
        return result
#  prepick_to_pick = ((0, 0, 0.32), (0, 0, 0, 1))
#  postpick_to_pick = ((0, 0, 0.32), (0, 0, 0, 1))
#  prepick_pose = utils.multiply(pick_pose, prepick_to_pick)
#  postpick_pose = utils.multiply(pick_pose, postpick_to_pick)
#  
#  delta = (np.float32([0, 0, -0.001]),
#           utils.eulerXYZ_to_quatXYZW((0, 0, 0)))
#  
#  preplace_to_place = ((0, 0, 0.32), (0, 0, 0, 1))
#  postplace_to_place = ((0, 0, 0.32), (0, 0, 0, 1))
#  preplace_pose = utils.multiply(place_pose, preplace_to_place)
#  postplace_pose = utils.multiply(place_pose, postplace_to_place)