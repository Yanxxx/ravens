# coding=utf-8
# Copyright 2021 Yan Li, UTK, Knoxville, TN, 37909. 
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

" Naive MDP "

import numpy as np
import matplotlib.pyplot as plt
import copy

class NaiveMDP:
    """ """
    
    def __init__(self):
        self.states = []
        self.action = []
        self.value = []
        self.policy = []
        self.length = 0
        self.trace = []
        self.current_state = 0
        self.decay = 0.9
        self.action.append([])
        
        
    def loadExperience(self, trace):
        self.length = 0
        length = len(trace)
        if length < 1:
            print('please input a valid trace, the input trace does not contain any valid points')
            return 
        self.length = length
        self.current_state = length - 2
        self.trace = trace
        self.states = list(range(length))
        self.value = []
        v = np.zeros(length)
        v[-1] = 0.1
        self.value.append(v)
        self.policy = []
        p = np.zeros(self.length)
        p[-1] = 1
        self.policy.append(p)
#        self.policy = np.ones(length) * 1.0 / length

    def act(self, state):
        if self.length == 0:
            return
        steps = self.length - state
        precedence = self.trace[:state - 1]
        current_state = self.trace[state - 1]
        sub_traces = []
        for i in range(1, steps):
#            tmp = copy.deepcopy(precedence)
            sub_trace = []
            sub_trace.append(self.trace[state + i])
            ids = self.tail(state + i)
            if not ids:
                sub_traces.append(sub_trace)
                continue
            for idx in ids:
                sub_trace.append(self.trace[idx])
            sub_traces.append(sub_trace)
        return precedence, current_state, sub_traces

    # dynamic programming
    def tail(self, state):
        pos = self.length - state - 1
        if len(self.policy) < pos:
            return
        print(len(self.policy), self.length, state, pos)
        p = self.policy[pos]
        idx = np.argmax(p)
        a = copy.deepcopy(self.action[self.length - 1 -idx])
        a = a.insert(0, idx)
        self.action.append(a)
        return a
    
    def done(self):
        return self.current_state < 0
    
    def step(self):
        if self.length == 0:
            return
        if self.current_state < 0:
            return 
        print('step, current state: ', self.current_state)
        traces = self.act(self.current_state)
        self.current_state -= 1
        return traces
    
    def update(self, rewards):
        if len(rewards) < 1:
            return
        if self.length == 0:
            return
        dominator = sum(np.exp(self.value))
        immediate_reward = rewards[0]
        v = np.zeros(self.length)
        v[self.current_state] = immediate_reward
#        self.value[self.current_state] = immediate_reward
        
#            self.value[self.current_state] += self.decay * rewards[i]
        e = sum(rewards) - rewards[0]
        self.value[self.current_state] += self.decay * e
        policy = np.zeros(self.length)
        for i in range(self.current_state, self.length):
            policy[i] = np.exp(self.value[i]) / dominator
        print(policy)
        self.policy.append(policy)
        
    def updatePolicy(self):
        if self.length == 0:
            return
        self.policy = self.policy[-1::-1]
        p = np.array(self.policy)
#        print(p)
        plt.figure
        plt.imshow(p)
        plt.show()
    
    def keyframe(self): # greedy
        if self.length == 0:
            return
        action = []
        
        for p in self.policy:
            a = p.index(max(p))
            action.append(a)
        
        return action
            
            
    def getMetric(self):
        return self.value, self.policy
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            