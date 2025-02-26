import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, action_dim):
        super(DQN, self).__init__()

        # Convolutional Layers
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(4608, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        # Dueling Streams
        self.a = nn.Linear(128, action_dim)
        self.v = nn.Linear(128, 1)

    def forward(self, x):
        x = self.net(x)

        x = self.fc(x)
        a = self.a(x)
        v = self.v(x)

        q = v + (a - a.mean(dim=-1, keepdim=True))  # Combine streams
        return q
