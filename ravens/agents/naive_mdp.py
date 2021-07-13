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
        self.decay = 0.99
#        self.action.append([])
        
        
    def loadExperience(self, trace):
        self.length = 0
        length = len(trace)
        if length < 1:
            print('please input a valid trace, the input trace does not contain any valid points')
            return 
        self.length = length
        self.current_state = length - 1
        self.trace = trace
        self.states = list(range(length))
        self.value = []
        self.value = np.zeros((length, length))
        self.value[-1, -1] = 0.1
        
        self.policy = np.ones((length, length)) * 1.0 / length
        
        for i in range(length):
            self.action.append([])
#        self.policy = np.ones(length) * 1.0 / length

    def act(self, state):
        if self.length == 0:
            return
        steps = self.length - state
        precedence = list(range(state))
        current_state = state
        sub_traces = []
        for i in range(1, steps):
#            tmp = copy.deepcopy(precedence)
            sub_trace = []
#            sub_trace.append(state + i)
            
            idx = self.tailNaive(state + i)
            
#            ids = self.tail(state + i)
            print('tails ', idx)
            sub_trace = np.concatenate((sub_trace, idx))
#            if not ids:
#                sub_traces.append(sub_trace)
#                continue
#            for idx in ids:
#                sub_trace.append(self.trace[idx])
            sub_traces.append(sub_trace)
            
        return precedence, current_state, sub_traces
    
    def getSubtraces(self, idx):
        if not idx:
            return []
        if type(idx) != list:
#            print('current state: *********************** ', idx)
            print(self.trace[int(idx)])
            return [self.trace[int(idx)]]
        r = []
        for i in idx:
            r.append(self.trace[int(i)])
        return r

    # dynamic programming
    def tail(self, state):
        
#        print(len(self.policy), self.length, state, pos)
#        p = self.policy[state, :]
#        v = self.value[state, :]
#        idx = np.argmax(p)
#        idx = np.argmax(v)
        idx = state
        a = []
        while idx != self.length - 1:
            v = self.value[idx]
            idx = np.argmax(v)
            a = self.action[idx]
            a.insert(0, idx)
        print(idx, a)
        self.action.append(a)
        return a
    
    def tailNaive(self, state):
        return list(range(state, self.length))
    
    def done(self):
        return self.current_state < 1
    
    def step(self):
        if self.length == 0:
            return
        if self.current_state < 1:
            return 
        self.current_state -= 1
        print('step, current state: ', self.current_state)
        p, c, s = self.act(self.current_state)
        precedence = self.getSubtraces(p)
        current = self.getSubtraces(c)
        subtraces = self.getSubtraces(s)
        return precedence, current, subtraces
    
    
    def update(self, rewards):
        if len(rewards) < 1:
            return
        if self.length == 0:
            return
#        comb = len(rewards)    
#        check = self.length - self.current_state - 2
#        if comb != check:
#            print('debug error, the rewards number is not match the state', comb, check)
#            return
        immediate_reward = rewards[0][0]

        self.value[self.current_state, self.current_state] = immediate_reward
        for count, r in enumerate(rewards):
            vp = 0
#            for j in range(1, len(r)):
            for j in range(len(r)):
                vp = r[j] + vp * self.decay
                print(self.current_state, j)
            print('rewards: ', r)
            print('v ', vp)
            self.value[self.current_state, self.current_state + count] = vp

        dominator = sum(np.exp(self.value[self.current_state]))
        self.policy[self.current_state] = np.exp(self.value[self.current_state]) / dominator

#        print(self.policy)

        
    def updatePolicy(self):
        if self.length == 0:
            return
#        self.policy = self.policy[-1::-1]
#        print(self.policy)
#        p = np.array(self.policy)
#        print('*************************************************', self.policy.shape)
        print(self.value)
        plt.figure
        plt.imshow(self.value)
        plt.show()
    
    def keyframe(self): # greedy
        if self.length == 0:
            return
        action = []
        
        for p in self.policy:
            a = np.argmax(p)
            action.append(a)
        
        return action
            
            
    def getMetric(self):
        return self.value, self.policy
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            