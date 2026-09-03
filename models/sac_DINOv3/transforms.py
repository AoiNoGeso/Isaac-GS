"""env観測をモデル観測へ変換するObservationTransform群(Habitat-LabのObservationTransformer相当)"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from models.sac_DINOv3.dino_backbone import get_backbone


class DINOTransform:
    """env観測のrgbをDINOv3パッチグリッド特徴へ変換する観測変換。
    rollout時に1回だけ適用し、ReplayBufferにはdino_featを格納する"""

    def __init__(self, device: str = "cuda"):
        self._backbone = get_backbone(device)

    def transform_observation_space(self, obs_space: spaces.Dict) -> spaces.Dict:
        """rgbをdino_featへ置き換えた観測空間を返す"""
        dino_feat_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._backbone.hidden_size, self._backbone.grid_size, self._backbone.grid_size),
            dtype=np.float32,
        )
        spaces_dict = {k: v for k, v in obs_space.spaces.items() if k != "rgb"}
        spaces_dict["dino_feat"] = dino_feat_space
        return spaces.Dict(spaces_dict)

    def __call__(self, obs: dict) -> dict:
        """rgbをdino_featへ置き換えた観測dictを返す(numpy in / numpy out)"""
        obs = dict(obs)
        obs["dino_feat"] = self._backbone.extract(obs["rgb"])
        del obs["rgb"]
        return obs
