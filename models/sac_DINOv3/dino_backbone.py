"""DINOv3(facebook/dinov3-vits16-pretrain-lvd1689m)による視覚特徴抽出。完全凍結・単一インスタンス
共有を前提とする(SACAgentのactor_enc/critic_encとは独立に、rollout層で1回だけforwardする)。

出力トークン列は 1 CLS + num_register_tokens レジスタ + パッチトークン の順で並ぶ
入力: 224x224 出力: 1+4+196=201トークン*hidden_size=384)。
"""

from __future__ import annotations

import numpy as np
import torch

MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"

# ImageNet標準正規化(DINOv3のprocessorが使う値と同じ)
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class DINOBackbone:
    """完全凍結のDINOv3バックボーン。ReplayBufferへ保存する特徴量を収集時に1回だけ計算する用途
    (models/sac_DINOv3/train.py・test.pyのrollout層で1インスタンスだけ生成して使い回す)"""

    def __init__(self, device: str):
        from transformers import AutoModel

        self.device = device
        self._model = AutoModel.from_pretrained(MODEL_ID).to(device)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

        cfg = self._model.config
        self.hidden_size: int = cfg.hidden_size
        self.patch_size: int = cfg.patch_size
        self.num_register_tokens: int = cfg.num_register_tokens
        self.image_size: int = cfg.image_size  # 224 EnvConfig.camera_resolutionをこれに合わせる
        self.grid_size: int = self.image_size // self.patch_size  # 14

        self._mean = _MEAN.to(device)
        self._std = _STD.to(device)

    @torch.no_grad()
    def extract(self, rgb: np.ndarray) -> np.ndarray:
        """rgb: (3,H,W) float32 [0,1] の観測1枚から、パッチグリッド特徴(hidden_size, grid, grid)を返す
        (H,WはEnvConfig.camera_resolutionでimage_sizeに合わせておくこと。前処理はここで正規化のみ行う)"""
        x = torch.from_numpy(rgb).to(self.device, dtype=torch.float32).unsqueeze(0)
        x = (x - self._mean) / self._std

        last_hidden = self._model(pixel_values=x).last_hidden_state  # (1, 1+reg+patches, D)
        patch_tokens = last_hidden[:, 1 + self.num_register_tokens :, :]  # (1, patches, D)
        patch_grid = patch_tokens.transpose(1, 2).reshape(
            1, self.hidden_size, self.grid_size, self.grid_size
        )
        return patch_grid[0].cpu().numpy()


class DINOEnvWrapper:
    """envのreset()/step()が返すobsへ`dino_feat`(DINOv3パッチグリッド特徴)を追加するラッパー。
    obs["rgb"]はそのまま残すので、EpisodeRecorder等の動画記録・ログには影響しない。
    utils/rollout.py・utils/recorder.py等の共用コードは無変更で使えるよう、DINO固有の処理は
    ここに閉じ込めてある"""

    def __init__(self, env, backbone: DINOBackbone):
        self._env = env
        self._backbone = backbone
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def _augment(self, obs: dict) -> dict:
        obs = dict(obs)
        obs["dino_feat"] = self._backbone.extract(obs["rgb"])
        return obs

    def reset(self, **kwargs):
        obs, info = self._env.reset(**kwargs)
        return self._augment(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        return self._augment(obs), reward, terminated, truncated, info

    def close(self):
        self._env.close()
