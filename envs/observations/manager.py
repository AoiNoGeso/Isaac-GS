"""観測空間の構築(Isaac Sim不要)と、実行時の観測項目管理"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from gymnasium import spaces

from envs.observations.base import ObsTermCfg
from envs.observations.registry import build_term


def observation_space_from_cfg(terms_cfg: dict[str, ObsTermCfg]) -> spaces.Dict:
    """stack_sizeを適用した最終的な観測空間を返す(Isaac Sim不要)。
    env側の観測スキーマの唯一の情報源"""
    spaces_dict: dict[str, spaces.Box] = {}
    for name, cfg in terms_cfg.items():
        frame_space = cfg.frame_space()
        n = cfg.stack_size
        if n == 1:
            spaces_dict[name] = frame_space
            continue

        low = np.concatenate([frame_space.low] * n, axis=0)
        high = np.concatenate([frame_space.high] * n, axis=0)
        spaces_dict[name] = spaces.Box(low=low, high=high, dtype=frame_space.dtype)
    return spaces.Dict(spaces_dict)


class ObservationManager:
    """有効な観測項目の生成・履歴スタック・観測dictの組み立てを担う"""

    def __init__(self, terms_cfg: dict[str, ObsTermCfg], env: Any):
        self._terms_cfg = terms_cfg
        self._terms = {name: build_term(cfg, env) for name, cfg in terms_cfg.items()}
        self._history: dict[str, deque[np.ndarray]] = {
            name: deque(maxlen=cfg.stack_size) for name, cfg in terms_cfg.items()
        }

    def _push(self, name: str, frame: np.ndarray) -> np.ndarray:
        """最新フレームを履歴末尾へ積み、不足分は同フレームの繰り返しで埋めて連結する"""
        hist = self._history[name]
        hist.append(frame)
        while len(hist) < hist.maxlen:
            hist.append(frame)
        return np.concatenate(list(hist), axis=0)

    def reset(self) -> dict:
        """履歴を捨てて初期観測を返す"""
        out = {}
        for name in self._terms_cfg:
            self._history[name].clear()
            out[name] = self._push(name, self._terms[name].compute())
        return out

    def compute(self) -> dict:
        """各termを評価し、stack_sizeぶんスタックした観測dictを返す"""
        out = {}
        for name in self._terms_cfg:
            out[name] = self._push(name, self._terms[name].compute())
        return out
