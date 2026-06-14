"""
gymnasium.Env ラッパー

PointNavIsaacEnv を gymnasium の標準インターフェースでラップする．
skrl は gymnasium.Env を直接受け取れるため，このラッパーを経由して学習する．
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .isaac_env import PointNavIsaacEnv, PointNavEnvCfg


class PointNavGymEnv(gym.Env):
    """
    observation_space:
        rgb:  Box(0, 1, (3, 84, 84), float32)
        goal: Box(-1, 1, (2,), float32)

    action_space:
        Box(-1, 1, (2,), float32)  # [v_x_norm, ω_z_norm]
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg: PointNavEnvCfg | None = None, render_mode: str | None = None):
        super().__init__()
        self.cfg = cfg or PointNavEnvCfg()
        self.render_mode = render_mode

        W, H = self.cfg.camera_resolution
        self.observation_space = spaces.Dict(
            {
                "rgb":  spaces.Box(0.0, 1.0, shape=(3, H, W), dtype=np.float32),
                "goal": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self._env: PointNavIsaacEnv | None = None

    def _lazy_init(self):
        if self._env is None:
            self._env = PointNavIsaacEnv(self.cfg)

    # ──────────────────────────────────────────────────
    # gymnasium API
    # ──────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)
        self._lazy_init()
        obs = self._env.reset()
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array" and self._env is not None:
            rgb, _ = self._env._camera.get_rgbd()
            return rgb
        return None

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
