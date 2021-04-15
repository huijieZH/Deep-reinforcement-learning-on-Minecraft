from torch.utils.data import Dataset
import torch
import os 
import numpy as np
import random

class MineCraftRLDataset(Dataset):

    def __init__(self, args, shuffle = True):
        self.args = args
        self.dataset_path = os.path.join(args.ROOT, "data", "processdata", args.env + "_preprocess")
        self.Files = os.listdir(self.dataset_path)
        self.Files.sort()
        self.currentData = np.load(os.path.join(self.dataset_path, self.Files[0]), allow_pickle=True)
        self.currentFile = 0
        self.filenum = len(self.Files)
        self.datanum = self.filenum * args.DATA_PER_FILE
        self.FilesIndex = np.arange(self.filenum)
        self.shuffle = shuffle
        ## if shuffle, random change the sequence of Files
        if self.shuffle:
            np.random.shuffle(self.FilesIndex)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        idx = idx % self.datanum
        randomfilenum = idx // self.args.DATA_PER_FILE
        index_withinfile = idx % self.args.DATA_PER_FILE
        if self.currentFile != self.FilesIndex[randomfilenum]:
            self.currentFile = self.FilesIndex[randomfilenum]
            self.currentData = np.load(os.path.join(self.dataset_path, self.Files[self.currentFile]), allow_pickle=True)
        if not self.shuffle:
            s_t, a, r, s_t1, t = self.currentData[index_withinfile, :]
            s_t = torch.tensor(s_t)
            s_t1 = torch.tensor(s_t1)
            t = torch.tensor(t)
            a = torch.tensor(a)
            r = torch.tensor(r)
        else:
            s_t, a, r, s_t1, t = self.currentData[np.random.randint(self.currentData.shape[0]), :]
            s_t = torch.tensor(s_t)
            s_t1 = torch.tensor(s_t1)
            t = torch.tensor(t)
            a = torch.tensor(a)
            r = torch.tensor(r)
        return s_t, a, r, s_t1, t


class MineCraftRLDataLoader(object):

    def __init__(self, args, replaybuffer, shuffle = True):
        self.dataset = MineCraftRLDataset(args, shuffle)
        self.replaybuffer = replaybuffer
        self.args = args
        self.accumulate_num = 0
        self.total_num = len(self.dataset)
        self.shuffle = shuffle

    def getbatch(self, r):


        self.demonstration_batchsize = int(r * self.args.MINIBATCH)
        self.memory_batchsize = self.args.MINIBATCH - self.demonstration_batchsize
        
        #### prepare memory_buffer data
        miniBatch = random.sample(self.replaybuffer, self.memory_batchsize)

        ## batchsize * 3 * 64 * 64
        S_batch = torch.stack([b[0] for b in miniBatch]).to(torch.device(self.args.device))
        ## batchsize * 10
        A_batch = torch.stack([b[1] for b in miniBatch]).to(torch.device(self.args.device))
        ## batchsize 
        R_batch = torch.stack([b[2] for b in miniBatch]).to(torch.device(self.args.device))
        St1_batch = torch.stack([b[3] for b in miniBatch]).to(torch.device(self.args.device))
        Done_batch = torch.stack([b[4] for b in miniBatch]).to(dtype = torch.float32, device = torch.device(self.args.device))

        minibatch_memory = (S_batch, A_batch, R_batch, St1_batch, Done_batch)

        #### prepare demonstation data
        if self.accumulate_num > self.total_num:
            self.accumulate_num = 0
            self.dataset = MineCraftRLDataset(self.args, self.shuffle)
        
        demonstration_miniBatch = []
        for i in range(self.demonstration_batchsize):
            demonstration_miniBatch.append(self.dataset[i + self.accumulate_num])
        self.accumulate_num += self.demonstration_batchsize

        S_batch_demo = torch.stack([b[0] for b in demonstration_miniBatch]).to(torch.device(self.args.device))
        ## batchsize * 10
        A_batch_demo = torch.stack([b[1] for b in demonstration_miniBatch]).to(torch.device(self.args.device))
        ## batchsize 
        R_batch_demo = torch.stack([b[2] for b in demonstration_miniBatch]).to(torch.device(self.args.device))
        St1_batch_demo = torch.stack([b[3] for b in demonstration_miniBatch]).to(torch.device(self.args.device))
        Done_batch_demo = torch.stack([b[4] for b in demonstration_miniBatch]).to(dtype = torch.float32, device = torch.device(self.args.device))

        minibatch_memory_demo = (S_batch_demo, A_batch_demo, R_batch_demo, St1_batch_demo, Done_batch_demo)

        return minibatch_memory, minibatch_memory_demo
            
        
        
