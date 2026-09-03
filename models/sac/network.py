import torch
import torch.nn as nn
from gymnasium import spaces

_KEY_ORDER = ("rgb", "goal")  # 特徴量の連結順。既存チェックポイントとの整合のため変更禁止


class CNNEncoder(nn.Module):
    """RGB画像を特徴ベクトルに変換するCNN"""

    def __init__(self, img_size: int = 84, in_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat_dim = self.net(torch.zeros(1, in_channels, img_size, img_size)).shape[1]
        self.fc = nn.Sequential(nn.Linear(flat_dim, 256), nn.ReLU())
        self.out_dim = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.net(x))


class GoalEncoder(nn.Module):
    """ゴールベクトル[距離, 相対角度]を特徴ベクトルに変換する"""

    def __init__(self, in_dim: int = 2):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, 32), nn.ELU())
        self.out_dim = 32

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        return self.fc(g)


class PointNavEncoder(nn.Module):
    """CNNEncoderとGoalEncoderを観測空間に応じて組み合わせる統合エンコーダ"""

    def __init__(self, observation_space: spaces.Dict):
        super().__init__()
        self.keys = tuple(k for k in _KEY_ORDER if k in observation_space.spaces)
        self.out_dim = 0
        if "rgb" in self.keys:
            c, h, w = observation_space["rgb"].shape
            assert h == w, f"CNNEncoderは正方形画像のみ対応: {(h, w)}"
            self.cnn = CNNEncoder(img_size=h, in_channels=c)
            self.out_dim += self.cnn.out_dim
        if "goal" in self.keys:
            self.goal_enc = GoalEncoder(in_dim=observation_space["goal"].shape[0])
            self.out_dim += self.goal_enc.out_dim

    def forward(self, obs: dict) -> torch.Tensor:
        parts = []
        if "rgb" in self.keys:
            parts.append(self.cnn(obs["rgb"]))
        if "goal" in self.keys:
            parts.append(self.goal_enc(obs["goal"]))
        return torch.cat(parts, dim=-1)


def build_obs_pipeline(env_obs_space: spaces.Dict, device: str) -> tuple[spaces.Dict, None]:
    """モデルが消費する観測空間と、env観測の変換関数(sacでは不要のためNone)を返す"""
    return env_obs_space, None


def make_encoder(observation_space: spaces.Dict) -> PointNavEncoder:
    """train/test/deployで共通のエンコーダ生成関数"""
    return PointNavEncoder(observation_space)
