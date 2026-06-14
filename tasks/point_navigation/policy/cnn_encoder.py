"""
skrl 2.x 用 CNN + GoalEncoder ポリシー（Phase 1a）

観測:
  rgb  (3, 84, 84) → CNN Encoder → (256,)
  goal (2,)        → Goal Encoder → (32,)
  concat → (288,) → Actor / Critic MLP
"""

from __future__ import annotations

import torch
import torch.nn as nn
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin


class CNNEncoder(nn.Module):
    """RGB (3, H, W) → embedding (256,)"""

    def __init__(self, img_size: int = 84):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),   # → (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # → (64,  9,  9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # → (64,  7,  7)
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat_dim = self.net(torch.zeros(1, 3, img_size, img_size)).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.net(x))


class GoalEncoder(nn.Module):
    """goal (2,) → embedding (32,)"""

    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 32),
            nn.ELU(),
        )

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        return self.fc(g)


class PointNavActor(GaussianMixin, Model):
    """
    Gaussian policy（連続行動）

    obs: dict {"rgb": (B,3,H,W), "goal": (B,2)}
    action: (B, 2)  [-1, 1]  [v_x_norm, ω_z_norm]
    """

    def __init__(self, observation_space, action_space, device, img_size: int = 84):
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        self._img_size = img_size
        GaussianMixin.__init__(
            self,
            clip_actions=True,
            clip_log_std=True,
            min_log_std=-20,
            max_log_std=2,
        )

        self.cnn  = CNNEncoder(img_size=img_size)
        self.goal = GoalEncoder()
        self.mlp  = nn.Sequential(
            nn.Linear(256 + 32, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
        )
        self.mean_layer    = nn.Linear(64, self.num_actions)
        self.log_std_param = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs: dict, role: str):
        obs = inputs["observations"]
        if isinstance(obs, dict):
            rgb  = obs["rgb"]
            goal = obs["goal"]
        else:
            # skrl wrap_env() が Dict obs をアルファベット順にフラット化:
            # goal(2,) → rgb(3×84×84) の順
            goal = obs[:, :2]
            rgb  = obs[:, 2:].view(-1, 3, self._img_size, self._img_size)
        feat = torch.cat([self.cnn(rgb), self.goal(goal)], dim=1)
        mean = self.mean_layer(self.mlp(feat))
        log_std = self.log_std_param.expand_as(mean)
        return mean, {"log_std": log_std}


class PointNavCritic(DeterministicMixin, Model):
    """Value function（critic）"""

    def __init__(self, observation_space, action_space, device, img_size: int = 84):
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        self._img_size = img_size
        DeterministicMixin.__init__(self, clip_actions=False)

        self.cnn  = CNNEncoder(img_size=img_size)
        self.goal = GoalEncoder()
        self.mlp  = nn.Sequential(
            nn.Linear(256 + 32, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

    def compute(self, inputs: dict, role: str):
        obs = inputs["observations"]
        if isinstance(obs, dict):
            rgb  = obs["rgb"]
            goal = obs["goal"]
        else:
            goal = obs[:, :2]
            rgb  = obs[:, 2:].view(-1, 3, self._img_size, self._img_size)
        feat  = torch.cat([self.cnn(rgb), self.goal(goal)], dim=1)
        value = self.mlp(feat)
        return value, {}
