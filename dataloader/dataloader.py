from torch.utils.data import Dataset, DataLoader
import torch
import os 
import numpy as np
import random
import time

from collections import deque

from pfrl.collections.prioritized import PrioritizedBuffer
from pfrl.replay_buffers.prioritized import PrioritizedReplayBuffer

class MineCraftRLDataset(Dataset):

    def __init__(self, args, shuffle = True, seq_len=1):
        self.args = args
        self.dataset_path = os.path.join(args.ROOT, "data", "processdata", args.env + "_preprocess")
        videos = os.listdir(self.dataset_path)
        videos.sort()
        self.ID = {}
        self.id = 0
        self.seq_len = seq_len
        for video in videos:
            video_path = os.path.join(self.dataset_path, video)
            frames = os.listdir(video_path)
            for frame in frames:
                self.ID[self.id] = (video, int(frame.split(".")[0]))
                self.id += 1
        self.datanum = len(self.ID.keys())
        self.shuffle = shuffle
        self.indexs = np.arange(self.datanum)
        ## if shuffle, random change the sequence of Files
        if self.shuffle:
            np.random.shuffle(self.indexs)
        pass

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        index = self.indexs[idx]
        video, frame = self.ID[index]
        file_path_t0 = os.path.join(self.dataset_path, video, "{:04d}.npz".format(frame))
        file_path_t1 = os.path.join(self.dataset_path, video, "{:04d}.npz".format(frame + 1))
        data_t0 = np.load(file_path_t0)
        if os.path.exists(file_path_t1):
            data_t1 = np.load(file_path_t1)
            return data_t0["arr_0"], data_t0["arr_1"], data_t0["arr_2"], data_t0["arr_3"], data_t1["arr_0"]
        else:
            return data_t0["arr_0"], data_t0["arr_1"], data_t0["arr_2"], data_t0["arr_3"], np.zeros((3, 64, 64))


class MineCraftRLDataLoader(object):

    def __init__(self, args, replaybuffer, shuffle = True):
        self.dataset = MineCraftRLDataset(args, shuffle)
        self.replaybuffer = replaybuffer
        self.args = args
        self.shuffle = shuffle
        # self.replaybuffer_demonstration = deque()
        self.replaybuffer_demonstration = PrioritizedReplayBuffer(
                                         capacity=self.args.REPLAY_MEMORY,
                                         alpha=self.args.alpha, beta0=self.args.beta0,
                                         betasteps=self.args.betasteps,
                                         eps=self.args.eps,
                                         normalize_by_max=self.args.normalize_by_max,
                                         error_min=self.args.error_min,
                                         error_max=self.args.error_max,
                                         num_steps=self.args.num_steps)

        prebufferPath = os.path.join(args.ROOT, "data", "processdata", "replaybuffer_demonstration.pkl")
        if os.path.exists(prebufferPath):
            self.replaybuffer_demonstration.load(prebufferPath)
        else:
            for i in range(args.REPLAY_MEMORY):
                if i % 100 == 0:
                    print("load the {0}th data to buffer".format(i))
                self.replaybuffer_demonstration.append(
                    state = self.dataset[i][0],
                    action = self.dataset[i][1],
                    reward = self.dataset[i][2],
                    next_state = self.dataset[i][4],
                    is_state_terminal = not self.dataset[i][3][0],
                    )
            self.replaybuffer_demonstration.save(prebufferPath)
    def getbatch(self, r):

        self.demonstration_batchsize = int(r * self.args.MINIBATCH)
        self.memory_batchsize = self.args.MINIBATCH - self.demonstration_batchsize
        
        #### prepare memory_buffer data
        miniBatch = self.replaybuffer.sample(self.memory_batchsize)

        # ## batchsize * 3 * 64 * 64
        # S_batch = torch.stack([b[0] for b in miniBatch]).to(torch.device(self.args.device))
        # ## batchsize * 32
        # A_batch = torch.stack([b[1] for b in miniBatch]).to(torch.device(self.args.device))
        # ## batchsize 
        # R_batch = torch.stack([b[2] for b in miniBatch]).to(torch.device(self.args.device))
        # St1_batch = torch.stack([b[3] for b in miniBatch]).to(torch.device(self.args.device))
        # Done_batch = torch.stack([b[4] for b in miniBatch]).to(dtype = torch.float32, device = torch.device(self.args.device))

        S_batch = torch.stack([b[0]['state'].clone().detach() for b in miniBatch]).to(torch.device(self.args.device))
        A_batch = torch.stack([b[0]['action'].clone().detach() for b in miniBatch]).to(torch.device(self.args.device))
        R_batch = torch.stack([torch.tensor(b[0]['reward']) for b in miniBatch]).reshape((-1, 1)).to(torch.device(self.args.device))
        Done_batch = torch.stack([torch.tensor([not b[0]['is_state_terminal']]) for b in miniBatch]).to(dtype = torch.float32, device = torch.device(self.args.device))
        St1_batch = torch.stack([b[0]['next_state'].clone().detach() for b in miniBatch]).to(dtype = torch.float32, device = torch.device(self.args.device))

        minibatch_memory = (S_batch, A_batch, R_batch, St1_batch, Done_batch)

        miniBatch = self.replaybuffer_demonstration.sample(self.demonstration_batchsize)
        random_index = np.random.randint(len(self.dataset))
        self.replaybuffer_demonstration.append(
                    state = self.dataset[random_index][0],
                    action = self.dataset[random_index][1],
                    reward = self.dataset[random_index][2],
                    next_state = self.dataset[random_index][4],
                    is_state_terminal = not self.dataset[random_index][3][0],
                    )

        # self.replaybuffer_demonstration.append(self.dataset[np.random.randint(len(self.dataset))])
        # self.replaybuffer_demonstration.popleft()

        # S_batch_demo = torch.stack([torch.tensor(b[0]) for b in miniBatch]).to(torch.device(self.args.device))
        # ## batchsize * 10
        # A_batch_demo = torch.stack([torch.tensor(b[1]) for b in miniBatch]).to(torch.device(self.args.device))
        # ## batchsize 
        # R_batch_demo = torch.stack([torch.tensor(b[2]) for b in miniBatch]).to(torch.device(self.args.device))
        # Done_batch_demo = torch.stack([torch.tensor(b[3]) for b in miniBatch]).to(dtype = torch.float32, device = torch.device(self.args.device))
        # St1_batch_demo = torch.stack([torch.tensor(b[4]) for b in miniBatch]).to(dtype = torch.float32, device = torch.device(self.args.device))
        S_batch_demo = torch.stack([torch.tensor(b[0]['state']) for b in miniBatch]).to(torch.device(self.args.device))
        A_batch_demo = torch.stack([torch.tensor(b[0]['action']) for b in miniBatch]).to(torch.device(self.args.device))
        R_batch_demo = torch.stack([torch.tensor(b[0]['reward']) for b in miniBatch]).to(torch.device(self.args.device))
        Done_batch_demo = torch.stack([torch.tensor([not b[0]['is_state_terminal']]) for b in miniBatch]).to(dtype = torch.float32, device = torch.device(self.args.device))
        St1_batch_demo = torch.stack([torch.tensor(b[0]['next_state']) for b in miniBatch]).to(dtype = torch.float32, device = torch.device(self.args.device))

        minibatch_memory_demo = (S_batch_demo, A_batch_demo, R_batch_demo, St1_batch_demo, Done_batch_demo)

        return minibatch_memory, minibatch_memory_demo
            
        
        
