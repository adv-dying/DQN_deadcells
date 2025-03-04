import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, action_dim):
        super(DQN, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),

            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(9218, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.a = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        self.v = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x, boss, player):
        x = self.net(x)
        boss=boss.unsqueeze(-1)
        player=player.unsqueeze(-1)
        x = torch.cat((x, boss,player),dim=1)
        x = self.fc(x)
        a = self.a(x)
        v = self.v(x)
        q = v + (a - a.mean(dim=-1, keepdim=True))
        return q


if __name__ == '__main__':
    a = torch.rand((64, 4, 128, 128))
    net = DQN(3)
    boss = [2]*64
    player = [3]*64
    net(a, torch.tensor(boss),torch.tensor(player))
