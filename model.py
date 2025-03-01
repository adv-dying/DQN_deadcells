import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, action_dim):
        super(DQN, self).__init__()

        # Convolutional Layers
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=3),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )

        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),

        )
        # Dueling Streams
        self.a = nn.Sequential(nn.Linear(256, 256),
                               nn.ReLU(), nn.Linear(256, action_dim))
        self.v = nn.Sequential(nn.Linear(256, 256),
                               nn.ReLU(), nn.Linear(256, 1))

    def forward(self, x):
        x = self.net(x)

        x = self.fc(x)
        a = self.a(x)
        v = self.v(x)

        q = v + (a - a.mean(dim=-1, keepdim=True))
        return q


if __name__ == '__main__':
    a = torch.rand((1, 4, 84, 84))
    net = DQN(3)
    net(a)
