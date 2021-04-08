import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
import numpy as np
import random
import os

from collections import deque

## my own module
import sys
sys.path.append("./")
from .Arch import MineCraftRL

class DQN(MineCraftRL):
    def __init__(self, args, actionspace, env):
        ## load model or create a new model
        if not args.LOADING_MODEL:
            self.args = args
            self.Qnet = DQN_QNet(self.args)
            summary(self.Qnet.to(torch.device('cuda:0')), (self.args.CONTINUOUS_FRAME*3,64,64))
            self.Qnet.to(torch.device(self.args.device))
            self.actionspace = actionspace

            self.env = env
            self.action = self.env.action_space.sample()
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
        else:
            Models_PATH = os.listdir("./saved_network")
            Models_PATH.sort()
            Models_PATH = Models_PATH[::-1]
            for modelPath in Models_PATH:
                if modelPath[:3] == "DQN":
                    print(modelPath)
                    checkpoint = torch.load(os.path.join("./saved_network", modelPath))
                    break
            self.args = checkpoint['args']
            self.Qnet = DQN_QNet(self.args)
            self.Qnet.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer = optim.Adam(self.Qnet.parameters(), 5e-4, weight_decay=1e-4)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.criterion = nn.MSELoss()

            self.actionspace = actionspace

            self.env = env
            self.action = checkpoint['action']

            self.replaybuffer = checkpoint['replaybuffer']

            self.S = checkpoint['S']

            
            self.losses = checkpoint['losses']
            self.totalRewards = checkpoint['totalRewards']
            self.totalReward = self.totalRewards[-1]
            self.initialstep = checkpoint['step']
            self.initialepsilon = checkpoint['epsilon']


    def train(self):
        epsilon = self.initialepsilon
        step = self.initialstep
        while step < self.args.OBSERVE:
            step += 1
            self.step_observe(step, epsilon)
        
        while step < self.args.OBSERVE + self.args.EXPLORE:
            step += 1
            epsilon -= (self.args.INITIAL_EPSILON - self.args.FINAL_EPSILON) / self.args.EXPLORE
            self.step_train(step, epsilon)

        while True:
            step += 1
            self.step_train(step, epsilon)
            
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

        action_one_hot = F.one_hot(torch.tensor([action_index]), self.args.actionNum).squeeze()
        self.replaybuffer.append((self.S.reshape((-1, 64, 64)), 
                                  action_one_hot, 
                                  torch.tensor([reward]), 
                                  S_new.reshape((-1, 64, 64))))
        self.S = S_new

        if len(self.replaybuffer) > self.args.REPLAY_MEMORY:
            self.replaybuffer.popleft()

        self.totalReward += reward

        print("TIMESTEP", step, "/ STATE OBSERVE", \
        "/ EPSILON", epsilon, "/ ACTION", action_index, \
        "/ REWARD", reward, "/ TOTALREWARD", self.totalReward)

    def step_train(self, step, epsilon):
        input = self.S.reshape((1, -1, 64, 64)).to(torch.device(self.args.device))
        output = self.Qnet(input)

        # choose an action epsilon greedily
        if torch.rand(1) < epsilon:
            print("----------Random Action----------")
            action_index = np.random.randint(self.args.actionNum)
        else:
            action_index = int(torch.argmax(output))
        
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

        action_one_hot = F.one_hot(torch.tensor([action_index]), self.args.actionNum).squeeze()
        self.replaybuffer.append((self.S.reshape((-1, 64, 64)), 
                                  action_one_hot, 
                                  torch.tensor([reward]), 
                                  S_new.reshape((-1, 64, 64))))
        self.S = S_new

        if len(self.replaybuffer) > self.args.REPLAY_MEMORY:
            self.replaybuffer.popleft()

        self.totalReward += reward

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


        Y_minibatch = R_batch.reshape((-1, 1)) + self.args.gamma * torch.max(self.Qnet(St1_batch), axis = 1)[0].reshape((-1, 1))
        X_minibatch = torch.sum(A_batch * self.Qnet(S_batch), axis = 1, keepdim = True)
        
        self.optimizer.zero_grad()
        loss = self.criterion(X_minibatch, Y_minibatch)
        loss.backward()
        self.optimizer.step()

        self.losses.append(loss.item())
        self.totalRewards.append(self.totalReward)

        if step > self.args.OBSERVE + self.args.EXPLORE:
            print("TIMESTEP", step, "/ STATE Train", \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ LOSS", loss.item(), \
            "/ REWARD", reward, "/ TOTALREWARD", self.totalReward)  
        else:    
            print("TIMESTEP", step, "/ STATE Explore", \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ LOSS", loss.item(), \
            "/ REWARD", reward, "/ TOTALREWARD", self.totalReward)  
        
        if step % 100000 == 0:
            torch.save({
            'model_state_dict': self.Qnet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': step,
            'epsilon': epsilon,
            'action': self.action,
            'losses': self.losses,
            'totalRewards': self.totalRewards,
            'args': self.args,
            'replaybuffer': self.replaybuffer,
            'S':self.S
            }, "./saved_network/DQN_" + "0"*(7 - len(str(step))) + str(step)+".pt")

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