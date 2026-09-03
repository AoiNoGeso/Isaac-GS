import torch
import torch.nn as nn
from gymnasium import spaces

_KEY_ORDER = ("dino_feat", "goal")  # 特徴量の連結順。既存チェックポイントとの整合のため変更禁止


class DINOHead(nn.Module):
    """DINOv3のパッチグリッド特徴(hidden_size, grid, grid)を特徴ベクトルへ変換する軽量CNN
    (DINOv3本体は凍結・rollout層で1回だけforwardしてReplayBufferへ保存する前提。
    ここが学習対象になる唯一の視覚処理部分)"""

    def __init__(self, hidden_size: int, grid_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(hidden_size, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat_dim = self.net(torch.zeros(1, hidden_size, grid_size, grid_size)).shape[1]
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
    """DINOHeadとGoalEncoderを観測空間に応じて組み合わせる統合エンコーダ"""

    def __init__(self, observation_space: spaces.Dict):
        super().__init__()
        self.keys = tuple(k for k in _KEY_ORDER if k in observation_space.spaces)
        self.out_dim = 0
        if "dino_feat" in self.keys:
            hidden_size, grid_h, grid_w = observation_space["dino_feat"].shape
            assert grid_h == grid_w, f"DINOHeadは正方形グリッドのみ対応: {(grid_h, grid_w)}"
            self.dino_head = DINOHead(hidden_size, grid_h)
            self.out_dim += self.dino_head.out_dim
        if "goal" in self.keys:
            self.goal_enc = GoalEncoder(in_dim=observation_space["goal"].shape[0])
            self.out_dim += self.goal_enc.out_dim

    def forward(self, obs: dict) -> torch.Tensor:
        parts = []
        if "dino_feat" in self.keys:
            parts.append(self.dino_head(obs["dino_feat"]))
        if "goal" in self.keys:
            parts.append(self.goal_enc(obs["goal"]))
        return torch.cat(parts, dim=-1)


def model_observation_space(env_obs_space: spaces.Dict) -> spaces.Dict:
    """envの観測空間からこのモデルが消費する空間へ変換する
    (rgbはdino_featへ抽出済みのため、エンコーダにもReplayBufferにも渡さない)"""
    return spaces.Dict({k: v for k, v in env_obs_space.spaces.items() if k != "rgb"})


def make_encoder(observation_space: spaces.Dict) -> PointNavEncoder:
    """train/test/deployで共通のエンコーダ生成関数"""
    return PointNavEncoder(observation_space)
