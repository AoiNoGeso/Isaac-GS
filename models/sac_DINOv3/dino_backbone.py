"""DINOv3(facebook/dinov3-vits16-pretrain-lvd1689m)による視覚特徴抽出。完全凍結・単一インスタンス
共有を前提とする(SACAgentのactor_enc/critic_encとは独立に、rollout層で1回だけforwardする)。

出力トークン列は 1 CLS + num_register_tokens レジスタ + パッチトークン の順で並ぶ
入力: 224x224 出力: 1+4+196=201トークン*hidden_size=384)。
"""

from __future__ import annotations

import numpy as np
import torch

from models.sac_DINOv3.config import DINO_IMAGE_SIZE

MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"

# ImageNet標準正規化(DINOv3のprocessorが使う値と同じ)
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class DINOBackbone:
    """完全凍結のDINOv3バックボーン。ReplayBufferへ保存する特徴量を収集時に1回だけ計算する用途
    (scripts/train.py・test.pyのrollout層で1インスタンスだけ生成して使い回す)"""

    def __init__(self, device: str):
        from transformers import AutoModel

        self.device = device
        self._model = AutoModel.from_pretrained(MODEL_ID).to(device)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

        cfg = self._model.config
        assert cfg.image_size == DINO_IMAGE_SIZE, (
            f"HuggingFace設定のimage_size({cfg.image_size})がconfig.DINO_IMAGE_SIZE"
            f"({DINO_IMAGE_SIZE})と不一致"
        )
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


_BACKBONE: DINOBackbone | None = None


def get_backbone(device: str = "cuda") -> DINOBackbone:
    """プロセス内で唯一のDINOv3インスタンスを返す(train/evalで二重ロードしないため)"""
    global _BACKBONE
    if _BACKBONE is None:
        _BACKBONE = DINOBackbone(device)
    return _BACKBONE
