#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 15:57:46 2021

@author: yan
"""



# coding=utf-8
# Copyright 2021 Yan Li, UTK, Knoxville, TN, 37996.
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

flags.DEFINE_string('assets_root', '.', '')
flags.DEFINE_string('data_dir', '.', '')
flags.DEFINE_bool('disp', False, '')
flags.DEFINE_bool('shared_memory', False, '')
flags.DEFINE_string('task', 'towers-of-hanoi', '')
flags.DEFINE_string('mode', 'train', '')
flags.DEFINE_integer('n', 1000, '')

FLAGS = flags.FLAGS


homej = [-1, -0.5, 0.5, -0.5, -0.5, 0]
def val2robot(val):
    targ_j = None
    if len(val) == 6:
        targ_j = np.array(homej) + np.array(val)
        targ_j *= np.pi
    if len(val) == 7:
        targ_pos = val[:3]
        targ_pose = val[3:]
        targ_j = (targ_pos, targ_pose)
    return targ_j

def main(unused_argv):

  # Initialize environment and task.
  env = Environment(
      FLAGS.assets_root,
      disp=FLAGS.disp,
      shared_memory=FLAGS.shared_memory,
      hz=480)
#  print(tasks.names['excavation'])
  task = tasks.names[FLAGS.task]()
  task.mode = FLAGS.mode

  # Initialize scripted oracle agent and dataset.
#  agent = task.oracle(env)
  dataset = Dataset(os.path.join(FLAGS.data_dir, f'{FLAGS.task}-{task.mode}'))

  # Train seeds are even and test seeds are odd.
  seed = dataset.max_seed
  if seed < 0:
    seed = -1 if (task.mode == 'test') else -2
    


#  episode, total_reward = [], 0
  seed += 2
  np.random.seed(seed)
  env.set_task(task)
  obs, blocks, targ_pose = env.reset()
#  obs = env.reset()
  act = {}
  while True:
      print('wait for the end of current setup')
      x = input().split(',')
      if len(x) < 6:
          if x[0] == 'b':
              break
          continue
      val = [float(x[i]) for i in range(len(x))]
      targ = val2robot(val)
      if len(x) == 6:
          env.movej(targ)
      if len(x) == 7:
          act['pose'] = targ
          act['grasp'] = [0, 0]
          env.step_move(act)
  
#homej = np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi

if __name__ == '__main__':
  app.run(main)
