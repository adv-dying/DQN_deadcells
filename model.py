import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, action_dim):
        super(DQN, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(9216, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.5),
        )

        self.a = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, action_dim)
        )
        self.v = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.net(x)
        x = self.fc(x)
        a = self.a(x)
        v = self.v(x)
        q = v + (a - a.mean(dim=-1, keepdim=True))
        return q
if __name__ == '__main__':
    a = torch.rand((64, 4, 128, 128))
    net = DQN(3)
    net(a)
