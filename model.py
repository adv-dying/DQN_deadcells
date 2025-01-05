import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, action_dim):
        super(DQN, self).__init__()

        # Convolutional Layers
        self.net = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Flatten(),
        )

        # Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(12544, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(12544, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        # Dueling Streams
        self.a = nn.Linear(512, action_dim)  # Advantage stream
        self.v = nn.Linear(512, 1)           # Value stream

    def forward(self, x):
        x = self.net(x)
        a = self.fc1(x)
        a = self.a(a)
        print(a.shape)
        v = self.fc2(x)
        v = self.v(v)
        print(v.shape)
        print(a.mean(dim=-1, keepdim=True).shape)
        q = v + (a - a.mean(dim=-1, keepdim=True))  # Combine streams
        return q
