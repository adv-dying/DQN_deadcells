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
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(420, 512),  # Match the flattened size
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout to prevent overfitting
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)  # Output size matches the action space
        )
    def forward(self, x):
        x = self.net(x)
        print(x.shape)
        return self.fc(x)
