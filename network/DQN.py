import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
import numpy as np

import os

from collections import deque

## my own module
import sys
sys.path.append("./")
from .Arch import MineCraftRL

class DQN(MineCraftRL):
    def __init__(self, args, actionspace, env):
        self.args = args

        ## load model or create a new model
        if not args.LOADING_MODEL:
            self.Qnet = DQN_QNet(self.args).to(torch.device(self.args.device))
        else:
            # Models_PATH = os.listdir("./saved_newwork")
            pass
        summary(self.Qnet, (self.args.CONTINUOUS_FRAME*3,64,64))

        self.actionspace = actionspace

        self.env = env
        self.action = self.env.action_space.sample()
        self.replaybuffer = deque()
        ## 
        self.S = torch.zeros((self.args.CONTINUOUS_FRAME, 3, 64, 64))
        self.totalReward = 0

        

    def train(self):
        epsilon = self.args.INITIAL_EPSILON
        step = 0
        while step < self.args.OBSERVE:
            step += 1
            self.step_observe(step, epsilon)

            

    def step_observe(self, step, epsilon):
        action_index = np.random.randint(self.args.actionNum)
        action_take = self.actionspace[action_index]
        
        for a in action_take:
            if a == 'camera':
                self.action[a][0] += action_take[a][0]
                self.action[a][1] += action_take[a][1]
            else:
                self.action[a] = action_take[a]
        
        obs, reward, done, info = self.env.step(self.action)
        state = torch.tensor(obs["pov"]).permute(2, 0, 1)  

        S_new = torch.cat((self.S[1:, :, :, :], state.reshape((1, 3, 64, 64))), axis = 0)

        action_one_hot = F.one_hot(torch.tensor([action_index]), self.args.actionNum)
        self.replaybuffer.append((self.S, action_one_hot, torch.tensor([reward]), S_new))
           

        if len(self.replaybuffer) > self.args.REPLAY_MEMORY:
            self.replaybuffer.popleft()

        self.totalReward += reward

        print("TIMESTEP", step, "/ STATE OBSERVE", \
        "/ EPSILON", epsilon, "/ ACTION", action_index, \
        "/ REWARD", reward, "/ TOTALREWARD", self.totalReward)

    def step_train(self):
        pass

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