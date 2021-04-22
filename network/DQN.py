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
        self.iter_update = 0

        self.actionspace = actionspace

        self.env = env
        # self.replaybuffer = deque()
        self.replaybuffer = PrioritizedReplayBuffer(
                                         capacity=self.args.REPLAY_MEMORY,
                                         alpha=self.args.alpha, beta0=self.args.beta0,
                                         betasteps=self.args.betasteps,
                                         eps=self.args.eps,
                                         normalize_by_max=self.args.normalize_by_max,
                                         error_min=self.args.error_min,
                                         error_max=self.args.error_max,
                                         num_steps=self.args.num_steps)
        ## 
        self.S = torch.zeros((self.args.CONTINUOUS_FRAME, 3, 64, 64))
        self.totalReward = 0

        self.dataLoader = MineCraftRLDataLoader(self.args, self.replaybuffer)

        self.losses = []
        self.totalRewards = []

        self.Qnet = DQN_QNet(self.args)
        self.Qnet.to(torch.device(self.args.device))

        self.action_update = 0
        self.action_index = 0

        self.step_memory = deque()

        if not os.path.exists(self.args.MODEL_SAVE):
            os.mkdir(self.args.MODEL_SAVE)
        models = os.listdir(self.args.MODEL_SAVE)

        if self.args.LOADING_MODEL and len(models) != 0:
            models = os.listdir(self.args.MODEL_SAVE)
            models.sort()
            data = torch.load(os.path.join(self.args.MODEL_SAVE, models[-1]))
            self.Qnet.load_state_dict(data['model_state_dict'])
            self.target_Qnet = copy.deepcopy(self.Qnet)

            self.criterion = nn.MSELoss()
            self.optimizer = optim.Adam(self.Qnet.parameters(), 2e-5, weight_decay=1e-5)
            self.initialstep = data['step']
            self.initialepsilon = data['epsilon']
            # self.initialR = data['r']
            self.initialR = 0
            self.OBSERVE_FROM_MODEL =True
            # self.initialstep = 0
            # self.initialepsilon = self.args.INITIAL_EPSILON
            # self.initialR = self.args.INITIAL_R
        else:
            # summary(self.Qnet, (3,64, 64))
            ## this is the target network
            self.target_Qnet = copy.deepcopy(self.Qnet)
            self.criterion = nn.MSELoss()
            self.optimizer = optim.Adam(self.Qnet.parameters(), 2.5e-4, weight_decay=1e-5)
            self.initialstep = 0
            self.initialepsilon = self.args.INITIAL_EPSILON
            self.initialR = self.args.INITIAL_R
            self.OBSERVE_FROM_MODEL =False
    def train(self):
        self.epsilon = self.initialepsilon
        self.step = self.initialstep
        self.r = self.initialR
        initial = 0 
        while True:

            if min(self.step, initial) < self.args.OBSERVE:
                done = self.step_observe(self.step, self.epsilon)
                initial += 1
            elif self.step < self.args.OBSERVE + self.args.PRETRAIN:
                self.r -= (self.args.INITIAL_R - self.args.FINAL_R) / self.args.PRETRAIN 
                self.epsilon -= (self.args.INITIAL_EPSILON - self.args.FINAL_EPSILON) / (self.args.EXPLORE + self.args.PRETRAIN)

                self.r = max(0.05, self.r)
                self.epsilon = max(0, self.epsilon)
                done = self.step_train(self.step, self.epsilon, self.r)
            
            elif self.step < self.args.OBSERVE + self.args.EXPLORE + self.args.PRETRAIN :
                self.epsilon -= (self.args.INITIAL_EPSILON - self.args.FINAL_EPSILON) / (self.args.EXPLORE + self.args.PRETRAIN)
                self.epsilon = max(0.05, self.epsilon)
                done = self.step_train(self.step, self.epsilon, self.r)

            else:
                done = self.step_train(self.step, self.epsilon, self.r)

            self.action_update += 1
            self.step += 1
            
            ## refresh every VIDEO_FRAME steps or done
            if done or self.step % self.args.VIDEO_FRAME == 0:
                self.totalReward = 0
                self.env.reset()
                self.step_memory = deque()

    def step_observe(self, step, epsilon):
        if not self.OBSERVE_FROM_MODEL:
            if self.action_update % self.args.ACTION_UPDATE_INTERVAL == 0:
                self.action_index = np.random.randint(self.args.actionNum)
                self.action_update = 0
            reward, done = self.step_interact(self.action_index)

        else:
            if self.action_update % self.args.ACTION_UPDATE_INTERVAL == 0:
                input = self.S.reshape((1, -1, 64, 64)).to(torch.device(self.args.device))
                with torch.no_grad():
                    output = self.Qnet(input)      
                self.action_index = int(torch.argmax(output))      
            reward, done = self.step_interact(self.action_index)

        self.log(step, "OBSERVE", epsilon, self.action_index, reward, self.totalReward)

        return done

    def step_train(self, step, epsilon, r):
        if self.action_update % self.args.ACTION_UPDATE_INTERVAL == 0:
            input = self.S.reshape((1, -1, 64, 64)).to(torch.device(self.args.device))
            with torch.no_grad():
                output = self.Qnet(input)
            # choose an action epsilon greedily
            if torch.rand(1) < epsilon:
                print("----------Random Action----------")
                self.action_index = np.random.randint(self.args.actionNum)
            else:
                self.action_index = int(torch.argmax(output))
            self.action_update = 0
        
        reward, done = self.step_interact(self.action_index)

        #given minibatch
        if step % self.args.TRAINING_INTERVAL == 0:
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
            if step > self.args.OBSERVE + self.args.EXPLORE + self.args.PRETRAIN:
                self.log(step, "TRAIN", epsilon, self.action_index, self.totalReward, loss)
            elif step > self.args.OBSERVE + self.args.PRETRAIN:    
                self.log(step, "EXPLORE", epsilon, self.action_index, self.totalReward, loss)
            else: 
                self.log(step, "PRETRAIN", epsilon, self.action_index, self.totalReward, loss)

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
        self.S = state
        # S_new = torch.cat((self.S[1:, :, :, :], state.reshape((1, 3, 64, 64))), axis = 0)

        action_one_hot = F.one_hot(torch.tensor([action_index]), self.args.actionNum).squeeze()
        # self.replaybuffer.append((self.S.reshape((-1, 64, 64)), 
        #                           action_one_hot, 
        #                           torch.tensor([reward]), 
        #                           S_new.reshape((-1, 64, 64)),
        #                           torch.tensor([not done]),
        #                           ))

        self.step_memory.append((state.reshape((-1, 64, 64)), action_one_hot, reward, not done))

        if done:
            ## the frame is terminal
            step_memory_list = list(self.step_memory)
            for data in step_memory_list:
                s, a, r, t = self.step_memory.popleft()
                for i in range(len(self.step_memory)):
                    r += self.step_memory[i][2] * (self.args.gamma**(i + 1))
                if len(self.step_memory) == 0:
                    self.replaybuffer.append(state = s, action = a, reward = r, 
                                            next_state = s, 
                                            is_state_terminal = not t) 
                else:        
                    self.replaybuffer.append(state = s, action = a, reward = r, 
                                            next_state = self.step_memory[0][0], 
                                            is_state_terminal = not t)        
        else:
            ## after accumulate n-step
            if len(self.step_memory) >= self.args.n:
                s, a, r, t = self.step_memory.popleft()
                for i in range(len(self.step_memory)):
                    r += self.step_memory[i][2] * (self.args.gamma**(i + 1))
                self.replaybuffer.append(state = s, action = a, reward = r, 
                                         next_state = self.step_memory[0][0], 
                                         is_state_terminal = not t)

        # self.replaybuffer.append(
        #                         state = self.S.reshape((-1, 64, 64)), 
        #                         action = action_one_hot, 
        #                         reward = torch.tensor([reward]), 
        #                         next_state = S_new.reshape((-1, 64, 64)), 
        #                         is_state_terminal = done)
        # if len(self.replaybuffer) > self.args.REPLAY_MEMORY:
        #     self.replaybuffer.popleft()

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

        error = (Y_minibatch - X_minibatch).to(torch.device("cpu")).detach().numpy()
        self.dataLoader.replaybuffer.update_errors(error[:self.dataLoader.memory_batchsize])
        self.dataLoader.replaybuffer_demonstration.update_errors(error[self.dataLoader.memory_batchsize:])

        return X_minibatch, Y_minibatch

    def new_reward(self, reward):
        ## more than on the previous reward
        return reward

    def log(self, step, state, epsilon, action_index, totalReward, loss):
        print("/ SArchitecture", self.arch, "/ STIMESTEP", step, "/ STATE", state, \
        "/ EPSILON", np.round(epsilon, 4), "/ ACTION", action_index, \
        "/ TOTALREWARD", self.totalReward, "/ Training loss", loss)

    def savemodel(self, step):
        torch.save({
            'model_state_dict': self.Qnet.state_dict(),
            'args':self.args,
            'epsilon':self.epsilon,
            'step':self.step,
            'r':self.r,
            "rewards":self.totalRewards,
            }, os.path.join(self.args.MODEL_SAVE, self.arch + "_" + self.args.env + "_" + "0"*(7 - len(str(step))) + str(step)+".pt"))

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

        error = (Y_minibatch - X_minibatch).to(torch.device("cpu")).detach().numpy()
        self.dataLoader.replaybuffer.update_errors(error[:self.dataLoader.memory_batchsize])
        self.dataLoader.replaybuffer_demonstration.update_errors(error[self.dataLoader.memory_batchsize:])
        return X_minibatch, Y_minibatch

class DQFD(DQN):
    def __init__(self, args, actionspace, env):
        super().__init__(args, actionspace, env)
        self.arch = "DQFD"
    
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

        error = (Y_minibatch - X_minibatch).to(torch.device("cpu")).detach().numpy()
        self.dataLoader.replaybuffer.update_errors(error[:self.dataLoader.memory_batchsize])
        self.dataLoader.replaybuffer_demonstration.update_errors(error[self.dataLoader.memory_batchsize:])

        Q_demo = self.Qnet(S_batch_demo)
        Q_max, Q_argmax = torch.max(Q_demo, axis = 1)
        A_demo = F.one_hot(Q_argmax, self.args.actionNum)
        Q_s_aE = torch.sum(Q_demo * A_batch_demo, axis = 1)

        l_a_aE = torch.sum(A_demo != A_batch_demo)/2

        margin_loss = torch.sum(Q_max - Q_s_aE) + l_a_aE

        return X_minibatch, Y_minibatch, margin_loss

    
    def step_train(self, step, epsilon, r):
        if self.action_update % self.args.ACTION_UPDATE_INTERVAL == 0:
            input = self.S.reshape((1, -1, 64, 64)).to(torch.device(self.args.device))
            with torch.no_grad():
                output = self.Qnet(input)
            # choose an action epsilon greedily
            if torch.rand(1) < epsilon:
                print("----------Random Action----------")
                self.action_index = np.random.randint(self.args.actionNum)
            else:
                self.action_index = int(torch.argmax(output))
            self.action_update = 0
        
        reward, done = self.step_interact(self.action_index)


        if step % self.args.TRAINING_INTERVAL == 0:
        #given minibatch
            X_minibatch, Y_minibatch, margin_loss = self.get_minibatch(r)

            self.optimizer.zero_grad()

            loss = self.criterion(X_minibatch, Y_minibatch) + self.args.loss_coeff_margin * margin_loss/self.args.MINIBATCH
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
                self.log(step, "TRAIN", epsilon, self.action_index, self.totalReward, loss)
            else:    
                self.log(step, "EXPLORE", epsilon, self.action_index, self.totalReward, loss)

        ## save model
        if step % self.args.saveStep == 0:
            self.savemodel(step)
        
        return done