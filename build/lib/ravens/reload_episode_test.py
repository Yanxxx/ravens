#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 14:31:03 2021

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

"""Data collection script."""

import os

from absl import app
from absl import flags

import numpy as np

from ravens import tasks
from ravens.dataset import Dataset
from ravens.environments.environment import Environment

import pickle
from os.path import join

#flags.DEFINE_string('assets_root', '.', '')
#flags.DEFINE_string('data_dir', '.', '')
#flags.DEFINE_bool('disp', False, '')
#flags.DEFINE_bool('shared_memory', False, '')
#flags.DEFINE_string('task', 'towers-of-hanoi', '')
#flags.DEFINE_string('mode', 'train', '')
#flags.DEFINE_integer('n', 1000, '')

#FLAGS = flags.FLAGS

asset_root = './environments/assets'
n = 100
mode = 'test'
#task_choice = 'block-insertion'
task_choice = 'auto-excavation'
display = True
data_dir = '.'

def expriment_setup(env, task):
  size = (0.1, 0.1, 0.04)
  env.set_task(task)
  urdf = 'insertion/ell.urdf'
  pose = env.task.get_random_pose(env, size)
  urdf = 'insertion/fixture.urdf'
  target_pose = env.task.get_random_pose(env, size)
  env.add_object(urdf, pose, 'fixed')
  return pose, target_pose
  

def main(unused_argv):

  # Initialize environment and task.
#  env = Environment(
#      FLAGS.assets_root,
#      disp=FLAGS.disp,
#      shared_memory=FLAGS.shared_memory,
#      hz=480)
  env = Environment(
      asset_root,
      disp=display,
      shared_memory=False,
      hz=480)
#  print(tasks.names['excavation'])
  task = tasks.names[task_choice]()
  task.mode = mode

  # Initialize scripted oracle agent and dataset.
  agent = task.oracle(env)
  dataset = Dataset(os.path.join('.', f'{task_choice}-{mode}'))

  # Train seeds are even and test seeds are odd.
  seed = dataset.max_seed
  if seed < 0:
    seed = -1 if (task.mode == 'test') else -2
#  if seed < 0:
#    seed = -1 if (task.mode == 'test') else -2
#
#  episode, total_reward = [], 0
#  seed += 2
#  np.random.seed(seed)
#  env.set_task(task)
#  obs = env.reset()
#  info = None
#  reward = 0
#  for _ in range(task.max_steps):
#      act = agent.act(obs, info)
#      episode.append((obs, act, reward, info))
#      obs, reward, done, info = env.step(act)
#      total_reward += reward
#      print(f'Total Reward: {total_reward} Done: {done}')
#      if done:
#          break
#  episode.append((obs, None, reward, info))
#  print(f'Total Reward: {total_reward} Done: {done}')
#  dataset.add(seed, episode)

  # Collect training data from oracle demonstrations.
  

#  data_dir = 'block-insertion-saved'
#  folder = 'info'
#  file = '000000-1'
#  
#  with open(join(data_dir, folder, file + '.pkl'), 'rb') as f:
#    infos = pickle.load(f)
  
  block_pose, fixture_pose = expriment_setup(env, task)

    
  while dataset.n_episodes < n:
    print(f'Oracle demonstration: {dataset.n_episodes + 1}/{n}')
    episode, total_reward = [], 0
    seed += 2
    np.random.seed(seed)
    env.set_task(task)
#    obs = env.reset()
    obs = env.load_env(block_pose, fixture_pose)
    info = None
    reward = 0
    for _ in range(task.max_steps):
      act = agent.act(obs, info)
      episode.append((obs, act, reward, info))
      obs, reward, done, info = env.step(act)
      total_reward += reward
      print(f'Total Reward: {total_reward} Done: {done}')
      if done:
        break
    episode.append((obs, None, reward, info))

    # Only save completed demonstrations.
    # TODO(andyzeng): add back deformable logic.
    if total_reward > 0.99:
      dataset.add(seed, episode)
      
#    obs = env.load_env(block_pose, fixture_pose)
    

if __name__ == '__main__':
  app.run(main)
