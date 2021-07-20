#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 07 23:25:32 2021

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

import os

from absl import app
from absl import flags

import numpy as np

from ravens import tasks
from ravens.dataset import Dataset
from ravens.environments.environment import Environment
from ravens.utils import utils

import matplotlib.pyplot as plt
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
import random

assets_root = "/home/yan/git/ravens/ravens/environments/assets/"
dataset_root = "/data/ravens_demo/"
#task_name = "place-red-in-green"
task_name = "block-insertion-nofixture"
mode = "train"
n = 10

class RandomTrajectory:
    """"""
    
    def __init__(self):
        self.init_pose = ([0.487, 0.109, 0.347], [0, 0, 0, 1])
#        self.init_pose = ([0.46562498807907104, -0.375, 0.3599780201911926], [0, 0, 0, 1])
        self.prepick_to_pick = ([0, 0, 0.05], (0, 0, 0, 1))
        self.postpick_to_pick = ([0, 0, 0.05], (0, 0, 0, 1))
        self.preplace_to_place = ([0, 0, 0.05], (0, 0, 0, 1))
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
        self.step_resolution = 0.1
        self.step_interpolate_distance = 0.02
        
    def __call__(self, actions=None):
        result = []
        if not actions:
            return result
        q = [0,0,0,1]
    
        pick_pose = actions['pose0']
        place_pose = actions['pose1']
        
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
          steps = int(np.round(d / self.step_resolution))
          print('step=:', steps)          
          position = (np.array(p2) - np.array(p1))/steps
          for j in range(steps - 1):
    #          result.append((p1 + position * (j + 1), q1 + pose * (j + 1)))
              q = self.trace_key[i + 1]['pose'][1]
              result.append({'pose': (p1 + position * (j + 1), q), 'grasp': [0, 0]})
                  
        return result
    
    def load(self, blocks, tar_pose):
        
        print(blocks)
        pos, pose = blocks[5]
        
        pick_pose = blocks[5]
        place_pose = tar_pose
        init_pose = np.array([0.4, 0.15, 0.36])
        prepick_pose = utils.multiply(pick_pose, self.prepick_to_pick)
        postpick_pose = utils.multiply(pick_pose, self.postpick_to_pick)
        preplace_pose = utils.multiply(place_pose, self.preplace_to_place)
        postplace_pose = utils.multiply(place_pose, self.postplace_to_place)
        
        gposes = []
        gposes.append(init_pose)
        gposes.append(prepick_pose[0])
        gposes.append(pick_pose[0])
        gposes.append(postpick_pose[0])
        gposes.append(preplace_pose[0])
        gposes.append(place_pose[0])
        gposes.append(postplace_pose[0])
        
        positions = []
        positions.append(init_pose)
        
        trajectory = []
        trajectory.append(init_pose)
        
        fig2 = plt.figure(2)
        ax3d = fig2.add_subplot(111, projection='3d')
        factor = 1.5
        
        trs_x = []
        trs_y = []
        trs_z = []
        
        for i in range(len(gposes) - 1):
          p1 = gposes[i]
          p2 = gposes[i+1]
          d = np.linalg.norm(np.array(p2) - np.array(p1))
          step = int(np.round(d / self.step_resolution))
          print('step=:', step)
          step_distance = (np.array(p2) - np.array(p1))/step
          for j in range(step - 1):
#              q = self.trace_key[i + 1]['pose'][1]
              p = p1 + step_distance * (j+1) + [random.uniform(-0.1,0.1) * factor, 
                                  random.uniform(-0.1, 0.1) * factor, 
                                  random.uniform(0, 0.1) * factor] 
              positions.append(p)
          positions.append(np.array(p2))
          tmp = np.array(positions)
          if len(positions) < 3:
              positions = []
              continue
          positions = []
          steps = int(np.round(d / self.step_interpolate_distance))
          print(steps)
          X = tmp[:,0]
          Y = tmp[:,1]
          Z = tmp[:,2]
#          print('xyz', X, Y, Z)
          tck, u = interpolate.splprep([X, Y, Z], s=2)
          x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
          u_fine = np.linspace(0,1,steps)
          x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
          trs_x.append(x_fine)
          trs_y.append(y_fine)
          trs_z.append(z_fine)
          ax3d.plot(x_knots, y_knots, z_knots, 'go')
          ax3d.plot(X, Y, Z, 'r*')
          ax3d.plot(x_fine, y_fine, z_fine, 'g')
          del tck, u, x_knots, y_knots, z_knots, x_fine, y_fine, z_fine, X, Y, Z

        gtptx = [pick_pose[0][0], place_pose[0][0], prepick_pose[0][0], preplace_pose[0][0]]
        gtpty = [pick_pose[0][1], place_pose[0][1], prepick_pose[0][1], preplace_pose[0][1]]
        gtptz = [pick_pose[0][2], place_pose[0][2], prepick_pose[0][2], preplace_pose[0][2]]
        
        ax3d.plot(gtptx, gtpty, gtptz, 'bx')
        print(len(trs_x))
        
        t1 = np.array([trs_x[0], trs_x[1], trs_x[2]])
        x = t1[-1,:]
        
        
            
#        print(len(positions), positions)
#        positions = np.array(positions)
#        print(positions.shape)
#        X = positions[:,0]
#        Y = positions[:,1]
#        Z = positions[:,2]
#        tck, u = interpolate.splprep([X, Y, Z], s=1)
#        
#        print('tck', tck)
#        print('u ', u)
#        
#        x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
#        u_fine = np.linspace(0,1,200)
#        x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
#
##        ax3d.plot(x_true, y_true, z_true, 'b')
##        ax3d.plot(x_sample, y_sample, z_sample, 'r*')
#        ax3d.plot(x_knots, y_knots, z_knots, 'go')
#        ax3d.plot(X, Y, Z, 'r-')
#        ax3d.plot(x_fine, y_fine, z_fine, 'g')
        fig2.show()
        plt.show()

def expriment_setup(env, task):
  size = (0.1, 0.1, 0.04)
  env.set_task(task)
  urdf = 'insertion/ell.urdf'
  pose = env.task.get_random_pose(env, size)
  urdf = 'insertion/fixture.urdf'
  target_pose = env.task.get_random_pose(env, size)
  env.add_object(urdf, pose, 'fixed')
  return pose, target_pose

def main():
    
  env = Environment(
      assets_root,
      disp=True,
      shared_memory=False,
      hz=480)
  
  task = tasks.names[task_name]()
  
#  pose, tar_pose = expriment_setup(env, task)
#  print(pose, tar_pose)
  
  
  
  
  
  
  
  
  
  
  
  
  task.mode = 'test'
  agent = task.oracle(env)
  dataset = Dataset(os.path.join('.', f'{task}-{task.mode}'))

  # Train seeds are even and test seeds are odd.
  seed = dataset.max_seed
  if seed < 0:
    seed = -1 if (task.mode == 'test') else -2

  rt = RandomTrajectory()
  # Collect training data from oracle demonstrations.
  while dataset.n_episodes < 10:
    print(f'Oracle demonstration: {dataset.n_episodes + 1}/{n}')
    episode, total_reward = [], 0
    seed += 2
    
    np.random.seed(seed)
    env.set_task(task)
    obs, blocks, pose = env.reset()

    
#    rt.load(blocks, pose)
     
    info = None
    reward = 0
    for _ in range(task.max_steps):
      act = agent.act(obs, info)      
      if not act:
          continue
      
      rt.load(blocks, pose)
      return
#      actions = trace_generator(act)
#      print(actions)
      episode.append((obs, act, reward, info))
      for a in actions:
#          atmp = {}
#          obs, reward, done, info = env.step_simple(act)
          obs, reward, done, info = env.step_move(a)
          episode.append((obs, a, reward, info))
          print(reward)
#          total_reward += reward
#          
#      reward, info = self.task.reward() if action is not None else (0, {})
#      done = self.task.done()
#
#      if self.ee.check_grasp() == True:
#        print("grasp succeed! Total steps in current episodes{:d}".format(self.episode_steps))
#        done = True
##      reward = 1
##      self.reset()
#  # Add ground truth robot state into info.
#      info.update(self.info)
#
#      obs = self._get_obs()
      #obs = None
      print(f'Total Reward: {total_reward} Done: {done}')
      if done:
        break
    episode.append((obs, None, reward, info))
    total_reward = 0

    # Only save completed demonstrations.
    # TODO(andyzeng): add back deformable logic.
    if total_reward > 0.99:
      dataset.add(seed, episode)

if __name__ == '__main__':
    main()
#  app.run(main)
