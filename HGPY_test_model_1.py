import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super.__init__()

        self.feature_tower_l = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(96, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.feature_tower_r1 = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(96, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.feature_tower_r2 = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(96, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_l = nn.Linear(64*8*8,128)
        self.fc_r1 = nn.Linear(64*8*8,128)
        self.fc_r2 = nn.Linear(64*8*8,128)

    def cnn_l(self,x):
        in_size = x.size(0)
        x = self.feature_tower_l(x)
        x = x.view(in_size,-1)
        x = F.relu(self.fc_l(x))
        return x

    def cnn_r1(self,x):
        in_size = x.size(0)
        x = self.feature_tower_r1(x)
        x = x.view(in_size,-1)
        x = F.relu(self.fc_r1(x))
        return x

    def cnn_r2(self,x):
        in_size = x.size(0)
        x = self.feature_tower_r2(x)
        x = x.view(in_size,-1)
        x = F.relu(self.fc_r2(x))
        return x

    def forward(self,x,y):
        out_l_co = self.cnn_l(x)
        out_r1 = self.cnn_r1(x)