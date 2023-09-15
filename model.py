import torch
import torch.nn as nn

import numpy as np


class resblock(nn.Module):
    def __init__(self, input_channel, filter_num, stride=1):
        super(resblock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(
                input_channel, filter_num, stride=stride, kernel_size=3,padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                filter_num, filter_num, stride=1, kernel_size=3,padding=1
            ),
        )
        if stride != 1:
            self.downsample = nn.Conv2d(
                input_channel, filter_num, kernel_size=1, stride=stride
            )
        else:
            self.downsample = lambda x: x

    def forward(self, x):     
        out1 = self.conv(x)
        out2=self.downsample(x)
        
        return out1+out2


class DQN(nn.Module):
    def __init__(self, action_dim):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(4, 32, kernel_size=(2, 3, 3), stride=(1, 2, 2)),
            nn.ReLU(),
            nn.Conv3d(32, 48, kernel_size=(2, 3, 3), stride=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(48, 64, kernel_size=(1, 3, 3), stride=(1, 1, 1)),
        )
        self.res = nn.Sequential(
            self.build_resblock(64, 64, 2),
            self.build_resblock(64, 96, 2, stride=2),
            self.build_resblock(96, 128, 2, stride=2),
            self.build_resblock(128, 256, 2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(12800, 512), nn.ReLU(), nn.Linear(512, action_dim)
        )

    def build_resblock(self, input_channel, filter_num, blocks, stride=1):
        reslist = []
        reslist.append(
            resblock(input_channel=input_channel, filter_num=filter_num, stride=stride)
        )
        for i in range(1, blocks):
            reslist.append(
                resblock(input_channel=filter_num, filter_num=filter_num, stride=1)
            )

        return nn.Sequential(*reslist)

    def forward(self, x):
        # x.shape=(4,3,270,480)
        out=self.conv(x)
        if len(x.shape)==4:
            out = torch.sum(out, dim=1)
        elif len(x.shape)==5:
            out = torch.sum(out, dim=2)
        else:
            raise Exception("expected 4 or 5 dimension of data, but recieve a data with %d"%(x.shape))
        out = self.res(out)
        out = self.fc(torch.max_pool2d(out,kernel_size=3).view(-1, 12800))
        return out
    