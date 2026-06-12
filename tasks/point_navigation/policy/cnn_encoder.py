"""
skrl 2.x 用 CNN + MLP ポリシー

RGB (3, H, W) と Depth (1, H, W) を結合して CNN でエンコードし，
得られた embedding を MLP actor/critic に渡す．

skrl 2.x 変更点:
  Model.__init__ の引数がすべて keyword-only になった．
  compute() の inputs キーが "observations" になった．
"""

from __future__ import annotations

import torch
import torch.nn as nn
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin


class CNNEncoder(nn.Module):
    """
    入力: RGB-D 結合テンソル (B, 4, H, W)
    出力: embedding (B, 256)
    """

    def __init__(self, in_channels: int = 4, img_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),  # → (32,20,20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),           # → (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),           # → (64, 7, 7)
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size, img_size)
            flat_dim = self.net(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.net(x))


class PointNavActor(GaussianMixin, Model):
    """
    Gaussian policy（連続行動）

    obs: dict {"rgb": (B,3,H,W), "depth": (B,1,H,W)}
    action: (B, 2)  [-1, 1]
    """

    def __init__(self, observation_space, action_space, device, img_size: int = 84):
        # skrl 2.x: すべて keyword-only
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=True,
            clip_log_std=True,
            min_log_std=-20,
            max_log_std=2,
        )

        self.encoder = CNNEncoder(in_channels=4, img_size=img_size)
        self.mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
        )
        self.mean_layer    = nn.Linear(64, self.num_actions)
        self.log_std_param = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs: dict, role: str):
        # skrl 2.x: obs は inputs["observations"] に入る（Dict space は dict として渡される）
        obs = inputs["observations"]
        if isinstance(obs, dict):
            rgb   = obs["rgb"]
            depth = obs["depth"]
        else:
            # フラットな場合のフォールバック（通常は来ない）
            rgb   = obs[:, :3 * 84 * 84].view(-1, 3, 84, 84)
            depth = obs[:, 3 * 84 * 84:].view(-1, 1, 84, 84)

        x    = torch.cat([rgb, depth], dim=1)
        feat = self.mlp(self.encoder(x))
        mean = self.mean_layer(feat)
        log_std = self.log_std_param.expand_as(mean)
        # skrl 2.x: (mean_actions, outputs_dict) の2値で返す
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
        DeterministicMixin.__init__(self, clip_actions=False)

        self.encoder = CNNEncoder(in_channels=4, img_size=img_size)
        self.mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

    def compute(self, inputs: dict, role: str):
        obs = inputs["observations"]
        if isinstance(obs, dict):
            rgb   = obs["rgb"]
            depth = obs["depth"]
        else:
            rgb   = obs[:, :3 * 84 * 84].view(-1, 3, 84, 84)
            depth = obs[:, 3 * 84 * 84:].view(-1, 1, 84, 84)

        x     = torch.cat([rgb, depth], dim=1)
        value = self.mlp(self.encoder(x))
        # skrl 2.x: (value, outputs_dict) の2値で返す
        return value, {}
