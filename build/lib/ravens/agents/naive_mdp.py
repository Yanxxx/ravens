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
import pickle


class NaiveMDP:
    """ """
    
    def __init__(self, naive=True):
        self.states = []
        self.action = []
        self.value = []
        self.policy = []
        self.length = 0
        self.trace = []
        self.current_state = 0
        self.decay = 0.99
        self.naive = naive
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
            self.action.append([length - 1])
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
            
            if self.naive:
                idx = self.tailNaive(state + i)
            else:
                idx = self.tailDP(state + i)
            
#            ids = self.tail(state + i)
            print('tails ', idx)
            sub_trace = np.concatenate((sub_trace, idx))
#            if not ids:
#                sub_traces.append(sub_trace)
#                continue
#            for idx in ids:
#                sub_trace.append(self.trace[idx])
            sub_traces.append(list(sub_trace))
            
        return precedence, current_state, sub_traces
    
    def getSubtraces(self, idx):
        r = []
#        print('trace indeces: ', idx)
        if not idx:
            return r        
        for l in idx:
            sub = []
            for i in l:
                sub.append(self.trace[int(i)])
            r.append(sub)
        return r

    def getHead(self, idx):
        r = []
#        print('condition porblem ', idx, (idx is None))
        if idx is None:
            return []
#        print('*********************8 ', idx)
        if type(idx) != list:
#            print('current state: ', idx)
#            print('current state: *********************** ', idx)
#            print(self.trace[int(idx)])
            r.append(self.trace[int(idx)])
#            print(r)
            return r
        for i in idx:
            r.append(self.trace[int(i)])
        return r
        

    # dynamic programming
    def tailDP(self, state):
        
#        print(len(self.policy), self.length, state, pos)
#        p = self.policy[state, :]
#        v = self.value[state, :]
#        idx = np.argmax(p)
#        idx = np.argmax(v)
        idx = state
        a = []
        while idx < self.length:
            if idx not in a:
                a.append(idx)
            v = self.value[idx]
#            print(self.value)
#            print(idx, v)
            idx = np.argmax(v)
#            print(idx, v)
#            input('wait')
            if idx not in a:
                a.append(idx)
            idx += 1
#            a = self.action[idx]
        print(idx, a)
        self.action.append(a)
        return a
    
    def tailNaive(self, state):
        return list(range(state, self.length))
    
    def done(self):
        return self.current_state < 0
    
    def step(self):
        if self.length == 0:
            return
        if self.current_state < 0:
            return 
        print('step, current state: ', self.current_state)
#        print('trace: ',self.trace[0])
        p, c, s = self.act(self.current_state)
#        print('s ', s)
#        print('current ******************8 ', c)
        precedence = self.getHead(p)
        current = self.getHead(c)
#        print('current position ', current)
        subtraces = self.getSubtraces(s)
        self.current_state -= 1
        return precedence, current, subtraces
    
    
    def update(self, rewards):
        print('update value function ', rewards)
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
        cs = self.current_state + 1
#        print('cs ', cs, immediate_reward)

#        print(self.value)
        self.value[cs, cs] = immediate_reward
        for count, r in enumerate(rewards):
            if len(r) < 2:
                continue
            vp = 0
#            for j in range(1, len(r)):
            for j in range(len(r)):
                vp = r[j] + vp * self.decay
#                print(self.current_state, j)
            print('rewards: ', r)
            print('v ', vp)
            print('pos ', cs, cs + count)
            self.value[cs, cs + count + 1] = vp

#        print(self.value)
        dominator = sum(np.exp(self.value[cs]))
        self.policy[cs] = np.exp(self.value[cs]) / dominator
#        print(self.value)
        with open('./value.txt', 'wb') as f:
            pickle.dump(self.value, f)

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
        plt.colorbar()
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
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            