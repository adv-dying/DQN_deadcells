import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, action_dim):
        super(DQN, self).__init__()

        # Convolutional Layers
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        # Dueling Streams
        self.a = nn.Linear(128, action_dim)  # Advantage stream
        self.v = nn.Linear(128, 1)           # Value stream

    def forward(self, x):
        x = self.net(x)

        x = self.fc(x)
        a = self.a(x)
        v = self.v(x)

        q = v + (a - a.mean(dim=-1, keepdim=True))  # Combine streams
        return q
