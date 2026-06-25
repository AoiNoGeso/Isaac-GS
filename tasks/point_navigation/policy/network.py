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
        self.out_dim = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.net(x))


class GoalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(2, 32), nn.ELU())
        self.out_dim = 32

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        return self.fc(g)


class PointNavEncoder(nn.Module):
    """input_rgb / input_goal フラグに応じて CNNEncoder / GoalEncoder を組み合わせる統合エンコーダ"""

    def __init__(self, input_rgb: bool = True, input_goal: bool = True, img_size: int = 84):
        super().__init__()
        self.input_rgb  = input_rgb
        self.input_goal = input_goal
        self.out_dim = 0
        if input_rgb:
            self.cnn = CNNEncoder(img_size)
            self.out_dim += self.cnn.out_dim
        if input_goal:
            self.goal_enc = GoalEncoder()
            self.out_dim += self.goal_enc.out_dim

    def forward(self, obs: dict) -> torch.Tensor:
        parts = []
        if self.input_rgb:
            parts.append(self.cnn(obs["rgb"]))
        if self.input_goal:
            parts.append(self.goal_enc(obs["goal"]))
        return torch.cat(parts, dim=-1)
