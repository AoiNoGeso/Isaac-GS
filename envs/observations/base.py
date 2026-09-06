"""観測項目(term)の設定クラスと本体プロトコル。

Isaac Sim なしで import できる必要がある(deploy.py が観測空間の構築のみに使うため)。
EnvConfig(pydantic BaseModel)からdictで参照される想定だが、ObsTermCfg自体は
pydantic BaseModelではなくplainなdataclassにする。pydantic v2はBaseModelの
サブクラスをフィールド宣言時の型へ静かにダウンキャストすることがあり、
RGBCameraCfg.resolutionのようなサブクラス固有フィールドが失われるため。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from gymnasium import spaces


@dataclass
class ObsTermCfg:
    """観測項目の共通設定"""

    stack_size: int = 1  # 1でスタック無し。Nで直近Nフレームをチャンネル方向に連結

    @property
    def frame_shape(self) -> tuple[int, ...]:
        """1フレーム分の形状。termを実体化せずに観測空間を構築するために使う"""
        raise NotImplementedError

    def frame_space(self) -> spaces.Box:
        """1フレーム分の観測空間"""
        raise NotImplementedError


@dataclass
class RGBCameraCfg(ObsTermCfg):
    resolution: tuple[int, int] = (84, 84)  # (W, H)。観測は(3, H, W)になる点に注意

    @property
    def frame_shape(self) -> tuple[int, ...]:
        w, h = self.resolution
        return (3, h, w)

    def frame_space(self) -> spaces.Box:
        return spaces.Box(0.0, 1.0, shape=self.frame_shape, dtype=np.float32)


@dataclass
class GoalCfg(ObsTermCfg):
    @property
    def frame_shape(self) -> tuple[int, ...]:
        return (2,)

    def frame_space(self) -> spaces.Box:
        return spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([np.inf, 1.0], dtype=np.float32),
            dtype=np.float32,
        )


class ObsTerm(Protocol):
    def compute(self) -> np.ndarray:
        """1フレーム分の観測を返す"""
        ...
