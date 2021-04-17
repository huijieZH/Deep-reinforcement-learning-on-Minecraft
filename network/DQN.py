import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
import numpy as np
import random
import os

from collections import deque
import copy
import time

## my own module
import sys
sys.path.append("./")
sys.path.append("../dataloader")
from .Arch import MineCraftRL
try:
    from dataloader import MineCraftRLDataLoader
except ImportError:    
    from dataloader.dataloader import MineCraftRLDataLoader

import time

from pfrl.replay_buffers.replay_buffer import ReplayBuffer
from pfrl.collections.prioritized import PrioritizedBuffer
from pfrl.replay_buffers.prioritized import PrioritizedReplayBuffer

class DQN_QNet(nn.Module):
    def __init__(self, args):
        super(DQN_QNet, self).__init__()
        self.actionNum = args.actionNum
        self.initialDim = args.dim_DQN_Qnet
        self.layers = nn.Sequential(
            ## cat 1 RBG images together
            # -1*3*64*64 -> -1*64*32*23
            nn.Conv2d(args.CONTINUOUS_FRAME * 3, self.initialDim, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(self.initialDim, self.initialDim, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, padding = 1, stride = 2),

            # -1*64*32*32 -> -1*128*16*16
            nn.Conv2d(self.initialDim, self.initialDim * 2, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(self.initialDim * 2, self.initialDim * 2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, padding = 1, stride = 2),

            # -1*128*16*16 -> -1*256*8*8
            nn.Conv2d(self.initialDim * 2, self.initialDim * 4, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(self.initialDim * 4, self.initialDim * 4, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, padding = 1, stride = 2),

            # -1*256*8*8 -> -1*512*4*4
            nn.Conv2d(self.initialDim * 4, self.initialDim * 8, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(self.initialDim * 8, self.initialDim * 8, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(3, padding = 1, stride = 2),

            # -1*(512*4*4) -> 8192
            nn.Flatten(),
            # -1*8192 -> 512
            nn.Linear(self.initialDim * 8 * 4 * 4, self.initialDim * 8),
            nn.ReLU(),
            nn.Linear(self.initialDim * 8, self.actionNum),
        )
    
    def forward(self, x):
        return self.layers(x)

class DQN(MineCraftRL):
    def __init__(self, args, actionspace, env):
        ## load model or create a new model
        self.arch = "DQN"
        self.args = args
        ## this is the current network
        self.Qnet = DQN_QNet(self.args)
        self.Qnet.to(torch.device(self.args.device))
        summary(self.Qnet, (3,64, 64))
        ## this is the target network
        self.target_Qnet = copy.deepcopy(self.Qnet)
        self.iter_update = 0

        self.actionspace = actionspace

        self.env = env
        # self.replaybuffer = deque()
        self.replaybuffer = ReplayBuffer(capacity=selfargs.REPLAY_MEMORY)
        ## 
        self.S = torch.zeros((self.args.CONTINUOUS_FRAME, 3, 64, 64))
        self.totalReward = 0


        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.Qnet.parameters(), 2.5e-4, weight_decay=1e-5)

        self.dataLoader = MineCraftRLDataLoader(self.args, self.replaybuffer)

        self.losses = []
        self.totalRewards = []

        self.initialstep = 0
        self.initialepsilon = self.args.INITIAL_EPSILON
        self.initialR = self.args.INITIAL_R

    def train(self):
        epsilon = self.initialepsilon
        step = self.initialstep
        r = self.initialR
        while True:
            step += 1
            if step < self.args.OBSERVE:
                done = self.step_observe(step, epsilon)
            elif step < self.args.OBSERVE + self.args.EXPLORE:
                epsilon -= (self.args.INITIAL_EPSILON - self.args.FINAL_EPSILON) / self.args.EXPLORE
                r -= (self.args.INITIAL_R - self.args.FINAL_R) / self.args.EXPLORE
                done = self.step_train(step, epsilon, r)
            else:
                done = self.step_train(step, epsilon, r)
            ## refresh every 10000 steps or done
            if done:
                self.totalReward = 0
                self.env.reset()

    def step_observe(self, step, epsilon):
        action_index = np.random.randint(self.args.actionNum)
        reward, done = self.step_interact(action_index)

        self.log(step, "OBSERVE", epsilon, action_index, reward, self.totalReward)

        return done

    def step_train(self, step, epsilon, r):
        input = self.S.reshape((1, -1, 64, 64)).to(torch.device(self.args.device))
        output = self.Qnet(input)
        # choose an action epsilon greedily
        if torch.rand(1) < epsilon:
            print("----------Random Action----------")
            action_index = np.random.randint(self.args.actionNum)
        else:
            action_index = int(torch.argmax(output))
        
        reward, done = self.step_interact(action_index)

        #given minibatch
        X_minibatch, Y_minibatch = self.get_minibatch(r)

        self.optimizer.zero_grad()
        loss = self.criterion(X_minibatch, Y_minibatch)
        loss.backward()
        self.optimizer.step()
    
        self.iter_update += 1
        if self.iter_update >= self.args.UPDATE_INTERVAL:
            self.iter_update = 0
            self.target_Qnet = copy.deepcopy(self.Qnet)

        self.losses.append(loss.item())
        self.totalRewards.append(self.totalReward)

        ## log
        if step > self.args.OBSERVE + self.args.EXPLORE:
            self.log(step, "TRAIN", epsilon, action_index, reward, self.totalReward)
        else:    
            self.log(step, "EXPLORE", epsilon, action_index, reward, self.totalReward)

        ## save model
        if step % self.args.saveStep == 0:
            self.savemodel(step)

        
        return done
    
    def step_interact(self, action_index):
        action_take = {
            'vector': self.actionspace.cluster_centers_[action_index]
        }
        
        
        obs, reward, done, info = self.env.step(action_take)
        state = torch.tensor(obs["pov"]).to(torch.float32)/255.0
        state = state.permute(2, 0, 1)

        S_new = torch.cat((self.S[1:, :, :, :], state_normalize.reshape((1, 3, 64, 64))), axis = 0)

        action_one_hot = F.one_hot(torch.tensor([action_index]), self.args.actionNum).squeeze()
        self.replaybuffer.append((self.S.reshape((-1, 64, 64)), 
                                  action_one_hot, 
                                  torch.tensor([reward]), 
                                  S_new.reshape((-1, 64, 64)),
                                  torch.tensor([not done]),
                                  ))
        self.S = S_new
        if len(self.replaybuffer) > self.args.REPLAY_MEMORY:
            self.replaybuffer.popleft()

        self.totalReward += reward

        return reward, done
    
    def get_minibatch(self, r):

        minibatch_memory, minibatch_memory_demo = self.dataLoader.getbatch(r)
        S_batch, A_batch, R_batch, St1_batch, Done_batch = minibatch_memory
        S_batch_demo, A_batch_demo, R_batch_demo, St1_batch_demo, Done_batch_demo = minibatch_memory_demo

        S_batch_cat = torch.cat((S_batch, S_batch_demo), axis = 0)
        A_batch_cat = torch.cat((A_batch, A_batch_demo), axis = 0)
        R_batch_cat = torch.cat((R_batch, R_batch_demo), axis = 0)
        St1_batch_cat = torch.cat((St1_batch, St1_batch_demo), axis = 0)
        Done_batch_cat = torch.cat((Done_batch, Done_batch_demo), axis = 0)


        Y_minibatch = R_batch_cat.reshape((-1, 1)) + self.args.gamma * torch.max(self.target_Qnet(St1_batch_cat), axis = 1)[0].reshape((-1, 1)) * Done_batch_cat.reshape((-1, 1))
        X_minibatch = torch.sum(A_batch_cat * self.Qnet(S_batch_cat), axis = 1, keepdim = True)

        return X_minibatch, Y_minibatch

    def new_reward(self, reward):
        ## more than on the previous reward
        return reward

    def log(self, step, state, epsilon, action_index, reward, totalReward):
        print("/ SArchitecture", self.arch, "/ STIMESTEP", step, "/ STATE", state, \
        "/ EPSILON", np.round(epsilon, 4), "/ ACTION", action_index, \
        "/ REWARD", reward, "/ TOTALREWARD", self.totalReward)

    def savemodel(self, step):
        torch.save({
            'model_state_dict': self.Qnet.state_dict(),
            'args':self.args,
            }, "./saved_network/" + self.arch + "_" + self.args.env + "_" + "0"*(7 - len(str(step))) + str(step)+".pt")

class DoubleDQN(DQN):
    def __init__(self, args, actionspace, env):
        super().__init__(args, actionspace, env)
        self.arch = "DoubleDQN"
    
    def get_minibatch(self, r):

        minibatch_memory, minibatch_memory_demo = self.dataLoader.getbatch(r)
        S_batch, A_batch, R_batch, St1_batch, Done_batch = minibatch_memory
        S_batch_demo, A_batch_demo, R_batch_demo, St1_batch_demo, Done_batch_demo = minibatch_memory_demo

        S_batch_cat = torch.cat((S_batch, S_batch_demo), axis = 0)
        A_batch_cat = torch.cat((A_batch, A_batch_demo), axis = 0)
        R_batch_cat = torch.cat((R_batch, R_batch_demo), axis = 0)
        St1_batch_cat = torch.cat((St1_batch, St1_batch_demo), axis = 0)
        Done_batch_cat = torch.cat((Done_batch, Done_batch_demo), axis = 0)

        
        
        A_Y_minibatch_cat = F.one_hot(torch.argmax(self.Qnet(St1_batch_cat), axis = 1), self.args.actionNum)
        Y_minibatch = R_batch_cat.reshape((-1, 1)) + self.args.gamma * \
            torch.sum(A_Y_minibatch_cat * self.target_Qnet(St1_batch_cat), axis = 1, keepdim = True) * Done_batch_cat.reshape((-1, 1))
        X_minibatch = torch.sum(A_batch_cat * self.Qnet(S_batch_cat), axis = 1, keepdim = True)

        return X_minibatch, Y_minibatch
    