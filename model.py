import torch
import torch.nn as nn

import numpy as np


class resblock(nn.Module):

    def __init__(self, input_channel, filter_num, stride=1):
        super(resblock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channel,
                      filter_num,
                      stride=stride,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(filter_num,
                      filter_num,
                      stride=1,
                      kernel_size=3,
                      padding=1),
        )

        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(input_channel,
                          filter_num,
                          kernel_size=1,
                          stride=stride), )
        else:
            self.downsample = lambda x: x
        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.conv(x)
        out2 = self.downsample(x)

        return self.relu(out1 + out2)


class DQN(nn.Module):

    def __init__(self, action_dim):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 48, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        self.res = nn.Sequential(
            self.build_resblock(64, 64, 2),
            self.build_resblock(64, 96, 2, stride=2),
            self.build_resblock(96, 128, 2, stride=2),
            self.build_resblock(128, 256, 2, stride=2),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.fc = nn.Sequential(nn.Linear(768, 256), nn.ReLU(),
                                nn.Linear(256, action_dim))

    def build_resblock(self, input_channel, filter_num, blocks, stride=1):
        reslist = []
        reslist.append(
            resblock(input_channel=input_channel,
                     filter_num=filter_num,
                     stride=stride))
        for i in range(1, blocks):
            reslist.append(
                resblock(input_channel=filter_num,
                         filter_num=filter_num,
                         stride=1))

        return nn.Sequential(*reslist)

    def forward(self, x):
        # torch.Size([3, 150, 460])
        out = self.conv(x)
        out = self.res(out)
        out = self.maxpool(out)
        out = out.view(-1, 768)
        out = self.fc(out)
        return out
