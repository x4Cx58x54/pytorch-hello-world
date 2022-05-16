import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # for CIFAR10
            nn.Conv2d(3, 6, 5),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(5*5*16, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )
    def forward(self, x):
        return self.net(x)
