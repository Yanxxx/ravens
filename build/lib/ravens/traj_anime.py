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
from ravens.agents.naive_mdp import NaiveMDP

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

def getline(actions):
    r = []
    if len(actions) < 1:
        return r
    for pt in actions:
#        print(pt)
        pos = pt['pose'][0]
        r.append(list(pos))
    return r

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
#  agent = NaiveMDP()
  key_agent = NaiveMDP()
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
    obs, blocks, pose = env.reset()
    info = None
    reward = 0
    for _ in range(task.max_steps):
      act = agent.act(obs, info)      
      if not act:
          continue
      actions = trace_generator.trace(act)
      line = getline(actions)
      key_agent.loadExperience(actions)
#      input("Press Enter to continue...")
      env.load_env(blocks, pose)
      
      env.add_line(line,[210/255.0,105/255.0,30/255.0])
      env.add_line_points(line, [0,100/255.0,0], 9)
      for a in actions:
          obs, r, done, info = env.step_move(a)
          
      input("Press Enter to continue...")
      env.load_env(blocks, pose)
      env.add_line(line,[210/255.0,105/255.0,30/255.0])
      env.add_line_points(line, [0,100/255.0,0], 9)
#      input("Press Enter to continue...")
      flag = True
      kp1 = []
      kp2 = []
      kp3 = []
      kp4 = []
      kp5 = []
      jump = 0
      preced = []
      preced, cur, subtraces = key_agent.step()
      preced = preced + cur
#          env.add_line_points(line, [0,1,0], 9)
      if flag:
          kp1 = preced[28]
          kp2 = preced[29]
          kp3 = preced[73]
          kp4 = preced[77]
          kp5 = preced[93]
          flag = False
      jump += 1
      pl = getline(preced)
      env.add_line_points(pl, [124/255.0,252/255.0,0], 16, 0.3, 2)
      
      k1 = getline([kp1])
      k2 = getline([kp2])
      k3 = getline([kp3])
      k4 = getline([kp4])
      k5 = getline([kp5])
      env.add_line_points(k1, [1,1,0], 60, 0, 2)
      env.add_line_points(k2, [1,1,0], 60, 0, 2)
      env.add_line_points(k3, [1,1,0], 60, 0, 2)
      env.add_line_points(k4, [1,1,0], 60, 0, 2)
      env.add_line_points(k5, [1,1,0], 60, 0, 2)
      opa = [kp1, kp2, kp3, kp4, kp5]
      for a in opa:
        obs, r, done, info = env.step_move(a)
#          cl = getline(cur)
#          env.add_line_points(cl, [1,0,0], 40, 2, 2)
      input("Press Enter to continue...")
#      while not key_agent.done():
#          preced = []
#          preced, cur, subtraces = key_agent.step()
#          preced = preced + cur
##          env.add_line_points(line, [0,1,0], 9)
#          if flag:
#              kp1 = preced[28]
#              kp2 = preced[29]
#              kp3 = preced[73]
#              kp4 = preced[77]
#              kp5 = preced[93]
#              flag = False
#          jump += 1
#          pl = getline(preced)
#          env.add_line_points(pl, [124/255.0,252/255.0,0], 16, 0.3, 2)
#          
#          k1 = getline([kp1])
#          k2 = getline([kp2])
#          k3 = getline([kp3])
#          k4 = getline([kp4])
#          k5 = getline([kp5])
#          env.add_line_points(k1, [1,1,0], 60, 0, 2)
#          env.add_line_points(k2, [1,1,0], 60, 0, 2)
#          env.add_line_points(k3, [1,1,0], 60, 0, 2)
#          env.add_line_points(k4, [1,1,0], 60, 0, 2)
#          env.add_line_points(k5, [1,1,0], 60, 0, 2)
#          opa = [kp1, kp2, kp3, kp4, kp5]
#          for a in opa:
#            obs, r, done, info = env.step_move(a)
##          cl = getline(cur)
##          env.add_line_points(cl, [1,0,0], 40, 2, 2)
#          input("Press Enter to continue...")
#          if jump % 20 != 0:
#              continue
#          for count, trace in enumerate(subtraces):
#              if count > 1:
#                  break
#              preced = preced + trace
##              print(type(trace), type(cur))
#              pl = getline(preced)
#              env.add_line_points(pl, [124/255.0,252/255.0,0], 16, 0.3, 2)
#              cl = getline(cur)
#              env.add_line_points(cl, [1,0,0], 40, 2, 2)
#          input("Press Enter to continue...")
#          env.load_env(blocks, pose)
#          env.add_line(line)
#          env.add_line_points(line, [0,1,0], 9)
#      
#      k1 = getline([kp1])
#      k2 = getline([kp2])
#      k3 = getline([kp3])
#      k4 = getline([kp4])
#      k5 = getline([kp5])
#      env.add_line_points(k1, [1,1,0], 60, 0, 2)
#      env.add_line_points(k2, [1,1,0], 60, 0, 2)
#      env.add_line_points(k3, [1,1,0], 60, 0, 2)
#      env.add_line_points(k4, [1,1,0], 60, 0, 2)
#      env.add_line_points(k5, [1,1,0], 60, 0, 2)
#      opa = [kp1, kp2, kp3, kp4, kp5]
#      for a in opa:
#        obs, r, done, info = env.step_move(a)
    episode.append((obs, None, reward, info))
    total_reward = 0

    # Only save completed demonstrations.
    # TODO(andyzeng): add back deformable logic.
    if total_reward > 0.99:
      dataset.add(seed, episode)

if __name__ == '__main__':
  app.run(main)
