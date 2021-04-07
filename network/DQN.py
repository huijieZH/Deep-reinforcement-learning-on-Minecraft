import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, args):
        super(DQN, self).__init__()
        self.actionNum = args.actionNum

        self.layers = nn.Sequential(
            ## cat 4 RBG images together
            # -1*12*64*64 -> -1*64*16*16
            nn.Conv2d(12, 64, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, padding = 1, stride = 4),

            # -1*64*16*16 -> -1*128*4*4
            nn.Conv2d(64, 128, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, padding = 1, stride = 4),

            # -1*128*4*4 -> -1*256*1*1
            nn.Conv2d(128, 256, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, padding = 1, stride = 4),

            # -1*(256*1*1) -> 256
            nn.Flatten(),
            # -1*256 -> 256
            nn.Linear(256, 256),
            nn.Linear(256, self.actionNum),
        )
    
    def forward(self, x):
        return self.layers(x)