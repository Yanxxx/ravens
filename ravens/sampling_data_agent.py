# coding=utf-8
# Copyright 2021 The Yan Li, UTK, Knoxville, TN.
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


def interpolateAct(act):
  if not act:
      return []
  step_resolution = 0.01
  init_pose = ([0.46562498807907104, -0.375, 0.3599780201911926], [0, 0, 0, 1])
  act_tmp = []
#  act_tmp['pose0'] = init_pose
#  act_tmp['pose1'] = act['pose0']
#  act_tmp['pose2'] = act['pose1']
  act_tmp.append(init_pose)
  act_tmp.append(act['pose0'])
  act_tmp.append(act['pose1'])
  
#  print(type(act), act)
#  act.insert(0, init_pose)
  result = []
  q = [0, 0, 0, 1]
  for i in range(len(act_tmp) - 1):
      result.append(act_tmp[i])
      p1 = act_tmp[i][0]
      p2 = act_tmp[i+1][0]
#      q1 = act_tmp[i][1]
#      q2 = act_tmp[i+1][1]
      d = np.linalg.norm(p2 - p1)
      step = int(np.round(d / step_resolution))
      print('step=:', step)
      position = (p2-p1)/step
#      pose = (q2 - q1)/step
      for j in range(step - 1):
#          result.append((p1 + position * (j + 1), q1 + pose * (j + 1)))
          result.append((p1 + position * (j + 1), q))
  return result


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
  agent = task.oracle(env)
  dataset = Dataset(os.path.join(FLAGS.data_dir, f'{FLAGS.task}-{task.mode}'))

  # Train seeds are even and test seeds are odd.
  seed = dataset.max_seed
  if seed < 0:
    seed = -1 if (task.mode == 'test') else -2

  # Collect training data from oracle demonstrations.
  while dataset.n_episodes < FLAGS.n:
    print(f'Oracle demonstration: {dataset.n_episodes + 1}/{FLAGS.n}')
    episode, total_reward = [], 0
    seed += 2
    np.random.seed(seed)
    env.set_task(task)
    obs = env.reset()
    info = None
    reward = 0
    for _ in range(task.max_steps):
      act = agent.act(obs, info)      
      if not act:
          continue
      actions = interpolateAct(act)
#      print(actions)
      episode.append((obs, act, reward, info))
      for a in actions:
          atmp = {}
          episode.append((obs, a, reward, info))
#          obs, reward, done, info = env.step_simple(act)
          atmp['pose'] = a
          atmp['grasp'] = 0
          obs, reward, done, info = env.step_single(atmp)
          total_reward += reward
      print(f'Total Reward: {total_reward} Done: {done}')
      if done:
        break
    episode.append((obs, None, reward, info))

    # Only save completed demonstrations.
    # TODO(andyzeng): add back deformable logic.
    if total_reward > 0.99:
      dataset.add(seed, episode)

if __name__ == '__main__':
  app.run(main)
