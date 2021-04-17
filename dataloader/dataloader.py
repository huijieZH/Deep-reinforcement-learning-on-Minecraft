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
        self.replaybuffer_demonstration = deque()
        prebufferPath = os.path.join(args.ROOT, "data", "processdata", "replaybuffer_demonstration.npy")
        if os.path.exists(prebufferPath):
            self.replaybuffer_demonstration = deque(np.load(prebufferPath, allow_pickle=True))
        else:
            for i in range(args.REPLAY_MEMORY):
                if i % 100 == 0:
                    print("load the {0}th data to buffer".format(i))
                self.replaybuffer_demonstration.append(self.dataset[i])
                np.save(prebufferPath, self.replaybuffer_demonstration)

    def getbatch(self, r):


        self.demonstration_batchsize = int(r * self.args.MINIBATCH)
        self.memory_batchsize = self.args.MINIBATCH - self.demonstration_batchsize
        
        #### prepare memory_buffer data
        miniBatch = random.sample(self.replaybuffer, self.memory_batchsize)

        ## batchsize * 3 * 64 * 64
        S_batch = torch.stack([b[0] for b in miniBatch]).to(torch.device(self.args.device))
        ## batchsize * 32
        A_batch = torch.stack([b[1] for b in miniBatch]).to(torch.device(self.args.device))
        ## batchsize 
        R_batch = torch.stack([b[2] for b in miniBatch]).to(torch.device(self.args.device))
        St1_batch = torch.stack([b[3] for b in miniBatch]).to(torch.device(self.args.device))
        Done_batch = torch.stack([b[4] for b in miniBatch]).to(dtype = torch.float32, device = torch.device(self.args.device))

        minibatch_memory = (S_batch, A_batch, R_batch, St1_batch, Done_batch)
        
        miniBatch = random.sample(self.replaybuffer_demonstration, self.demonstration_batchsize)

        self.replaybuffer_demonstration.append(self.dataset[np.random.randint(len(self.dataset))])
        self.replaybuffer_demonstration.popleft()

        S_batch_demo = torch.stack([torch.tensor(b[0]) for b in miniBatch]).to(torch.device(self.args.device))
        ## batchsize * 10
        A_batch_demo = torch.stack([torch.tensor(b[1]) for b in miniBatch]).to(torch.device(self.args.device))
        ## batchsize 
        R_batch_demo = torch.stack([torch.tensor(b[2]) for b in miniBatch]).to(torch.device(self.args.device))
        Done_batch_demo = torch.stack([torch.tensor(b[3]) for b in miniBatch]).to(dtype = torch.float32, device = torch.device(self.args.device))
        St1_batch_demo = torch.stack([torch.tensor(b[4]) for b in miniBatch]).to(dtype = torch.float32, device = torch.device(self.args.device))
        
        minibatch_memory_demo = (S_batch_demo, A_batch_demo, R_batch_demo, St1_batch_demo, Done_batch_demo)

        return minibatch_memory, minibatch_memory_demo
            
        
        
