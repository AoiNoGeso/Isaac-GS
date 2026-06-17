import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from tasks.point_navigation.policy.network import CNNEncoder, GoalEncoder


class PointNavFeaturesExtractor(BaseFeaturesExtractor):
    """RGB (3,H,W) + goal (2,) → 288次元特徴ベクトル"""

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 288):
        super().__init__(observation_space, features_dim)
        img_shape = observation_space["rgb"].shape  # (3, H, W)
        img_size = img_shape[1]
        self.cnn = CNNEncoder(img_size=img_size)
        self.goal = GoalEncoder()

    def forward(self, observations: dict) -> torch.Tensor:
        rgb = observations["rgb"]
        goal = observations["goal"]
        return torch.cat([self.cnn(rgb), self.goal(goal)], dim=1)
