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
from .Arch import MineCraftRL

class DQN_QNet(nn.Module):
    def __init__(self, args):
        super(DQN_QNet, self).__init__()
        self.actionNum = args.actionNum
        self.initialDim = args.dim_DQN_Qnet
        self.layers = nn.Sequential(
            ## cat 4 RBG images together
            # -1*12*64*64 -> -1*64*16*16
            nn.Conv2d(args.CONTINUOUS_FRAME * 3, self.initialDim, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(self.initialDim, self.initialDim, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, padding = 1, stride = 4),

            # -1*64*16*16 -> -1*128*4*4
            nn.Conv2d(self.initialDim, self.initialDim * 2, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(self.initialDim * 2, self.initialDim * 2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, padding = 1, stride = 4),

            # -1*128*4*4 -> -1*256*1*1
            nn.Conv2d(self.initialDim * 2, self.initialDim * 4, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(self.initialDim * 4, self.initialDim * 4, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, padding = 1, stride = 4),

            # -1*(256*1*1) -> 256
            nn.Flatten(),
            # -1*256 -> 256
            nn.Linear(self.initialDim * 4, self.initialDim * 4),
            nn.Linear(self.initialDim * 4, self.actionNum),
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

        ## this is the target network
        self.target_Qnet = copy.deepcopy(self.Qnet)
        self.iter_update = 0

        self.actionspace = actionspace

        self.env = env
        self.action = self.env.action_space.noop()
        self.replaybuffer = deque()
        ## 
        self.S = torch.zeros((self.args.CONTINUOUS_FRAME, 3, 64, 64))
        self.totalReward = 0

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.Qnet.parameters(), 5e-4, weight_decay=1e-4)

        self.losses = []
        self.totalRewards = []

        self.initialstep = 0
        self.initialepsilon = self.args.INITIAL_EPSILON


    def train(self):
        epsilon = self.initialepsilon
        step = self.initialstep
        while True:
            step += 1
            if step < self.args.OBSERVE:
                done = self.step_observe(step, epsilon)
            elif step < self.args.OBSERVE + self.args.EXPLORE:
                epsilon -= (self.args.INITIAL_EPSILON - self.args.FINAL_EPSILON) / self.args.EXPLORE
                done = self.step_train(step, epsilon)
            else:
                done = self.step_train(step, epsilon)
            ## refresh every 10000 steps or done
            if done:
                self.totalReward = 0
                self.env.reset()

    def step_observe(self, step, epsilon):
        action_index = np.random.randint(self.args.actionNum)
        reward, done = self.step_interact(action_index)

        self.log(step, "OBSERVE", epsilon, action_index, reward, self.totalReward)

        return done

    def step_train(self, step, epsilon):
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
        X_minibatch, Y_minibatch = self.get_minibatch()
        
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
        action_take = self.actionspace[str(action_index)]
        for a in action_take:
            if a == "camera":
                self.action[a] = action_take[a]
            else:
                ## add some uncertainty
                self.action[a] = 1 if np.random.rand() < action_take[a] else 0
        
        obs, reward, done, info = self.env.step(self.action)
        state = torch.tensor(obs["pov"]).to(torch.float32)
        state_normalize = (state - torch.mean(state.reshape((3, -1)), axis = 1))/torch.std(state.reshape((3, -1)), axis = 1)
        state_normalize = state_normalize.permute(2, 0, 1)

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
    
    def get_minibatch(self):
        miniBatch = random.sample(self.replaybuffer, self.args.MINIBATCH)

        Y_minibatch = torch.zeros((self.args.MINIBATCH,)).to(torch.device(self.args.device))
        X_minibatch = torch.zeros((self.args.MINIBATCH,)).to(torch.device(self.args.device))

        ## batchsize * 12 * 64 * 64
        S_batch = torch.stack([b[0] for b in miniBatch]).to(torch.device(self.args.device))
        ## batchsize * 10
        A_batch = torch.stack([b[1] for b in miniBatch]).to(torch.device(self.args.device))
        ## batchsize 
        R_batch = torch.stack([b[2] for b in miniBatch]).to(torch.device(self.args.device))
        St1_batch = torch.stack([b[3] for b in miniBatch]).to(torch.device(self.args.device))
        Done_batch = torch.stack([b[4] for b in miniBatch]).to(dtype = torch.float32, device = torch.device(self.args.device))

        
        # DQN
        Y_minibatch = R_batch.reshape((-1, 1)) + self.args.gamma * torch.max(self.target_Qnet(St1_batch), axis = 1)[0].reshape((-1, 1)) * Done_batch.reshape((-1, 1))
        X_minibatch = torch.sum(A_batch * self.Qnet(S_batch), axis = 1, keepdim = True)

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
            'action': self.action,
            'args':self.args,
            }, "./saved_network/" + self.arch + "_" + self.args.env + "_" + "0"*(7 - len(str(step))) + str(step)+".pt")

class DoubleDQN(DQN):
    def __init__(self, args, actionspace, env):
        super().__init__(args, actionspace, env)
        self.arch = "DoubleDQN"
    
    def get_minibatch(self):
        miniBatch = random.sample(self.replaybuffer, self.args.MINIBATCH)

        Y_minibatch = torch.zeros((self.args.MINIBATCH,)).to(torch.device(self.args.device))
        X_minibatch = torch.zeros((self.args.MINIBATCH,)).to(torch.device(self.args.device))

        ## batchsize * 12 * 64 * 64
        S_batch = torch.stack([b[0] for b in miniBatch]).to(torch.device(self.args.device))
        ## batchsize * 10
        A_batch = torch.stack([b[1] for b in miniBatch]).to(torch.device(self.args.device))
        ## batchsize 
        R_batch = torch.stack([b[2] for b in miniBatch]).to(torch.device(self.args.device))
        St1_batch = torch.stack([b[3] for b in miniBatch]).to(torch.device(self.args.device))
        Done_batch = torch.stack([b[4] for b in miniBatch]).to(dtype = torch.float32, device = torch.device(self.args.device))

        
        # Double DQN
        A_Y_minibatch = F.one_hot(torch.argmax(self.Qnet(St1_batch), axis = 1), self.args.actionNum)
        Y_minibatch = R_batch.reshape((-1, 1)) + self.args.gamma * torch.sum(A_Y_minibatch * self.target_Qnet(St1_batch), axis = 1, keepdim = True) * Done_batch.reshape((-1, 1))

        return X_minibatch, Y_minibatch
    