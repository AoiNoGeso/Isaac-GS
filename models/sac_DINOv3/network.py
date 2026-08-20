import torch
import torch.nn as nn

from models.sac_DINOv3.config import ModelConfig


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

    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(2, 32), nn.ELU())
        self.out_dim = 32

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        return self.fc(g)


class PointNavEncoder(nn.Module):
    """DINOHeadとGoalEncoderを設定に応じて組み合わせる統合エンコーダ"""

    def __init__(
        self,
        input_rgb: bool = True,
        input_goal: bool = True,
        hidden_size: int = 384,
        grid_size: int = 14,
    ):
        super().__init__()
        self.input_rgb = input_rgb
        self.input_goal = input_goal
        self.out_dim = 0
        if input_rgb:
            self.dino_head = DINOHead(hidden_size, grid_size)
            self.out_dim += self.dino_head.out_dim
        if input_goal:
            self.goal_enc = GoalEncoder()
            self.out_dim += self.goal_enc.out_dim

    def forward(self, obs: dict) -> torch.Tensor:
        parts = []
        if self.input_rgb:
            parts.append(self.dino_head(obs["dino_feat"]))
        if self.input_goal:
            parts.append(self.goal_enc(obs["goal"]))
        return torch.cat(parts, dim=-1)


def make_encoder(model_cfg: ModelConfig, hidden_size: int, grid_size: int) -> PointNavEncoder:
    """train/test/deployで共通のエンコーダ生成関数。hidden_size/grid_sizeはDINOBackboneから渡す"""
    return PointNavEncoder(
        input_rgb=model_cfg.input_rgb,
        input_goal=model_cfg.input_goal,
        hidden_size=hidden_size,
        grid_size=grid_size,
    )
