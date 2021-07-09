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
from ravens.utils import utils
from trace_generator import TraceGenerator

flags.DEFINE_string('assets_root', '.', '')
flags.DEFINE_string('data_dir', '.', '')
flags.DEFINE_bool('disp', False, '')
flags.DEFINE_bool('shared_memory', False, '')
flags.DEFINE_string('task', 'towers-of-hanoi', '')
flags.DEFINE_string('mode', 'train', '')
flags.DEFINE_integer('n', 1000, '')

FLAGS = flags.FLAGS


assets_root = "/home/yan/git/ravens/ravens/environments/assets/"
dataset_root = "/data/ravens_demo/"
#task_name = "place-red-in-green"
task_name = "block-insertion-nofixture"
mode = "train"
FLAGS = flags.FLAGS


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

  trace_generator = TraceGenerator()

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
      actions = trace_generator(act)
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
  app.run(main)
