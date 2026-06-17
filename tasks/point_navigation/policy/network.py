import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, img_size: int = 84):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat_dim = self.net(torch.zeros(1, 3, img_size, img_size)).shape[1]
        self.fc = nn.Sequential(nn.Linear(flat_dim, 256), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.net(x))


class GoalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(2, 32), nn.ELU())

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        return self.fc(g)
