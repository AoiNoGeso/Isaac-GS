from __future__ import annotations

import numpy as np


class EpisodeTracker:
    """1エピソード分の統計 (SPL 含む) を蓄積し, wandb ログ用 dict を返す."""

    def __init__(self, window: int = 100):
        self._window = window
        self._ep_reward = 0.0
        self._ep_steps = 0
        self._ep_path_len = 0.0
        self._ep_init_dist = 0.0
        self._prev_xy = None
        self._ep_count = 0
        self._total_success = 0
        self._total_collision = 0
        self._total_human_collision = 0
        self._success_buf: list[float] = []
        self._collision_buf: list[float] = []

    def step(self, reward: float, info: dict):
        self._ep_reward += reward
        self._ep_steps += 1
        xy = info.get("robot_xz")
        if xy is not None:
            xy = np.array(xy, dtype=np.float32)
            if self._prev_xy is not None:
                self._ep_path_len += float(np.linalg.norm(xy - self._prev_xy))
            self._prev_xy = xy

    def reset(self, obs: dict, info: dict | None = None):
        self._ep_reward = 0.0
        self._ep_steps = 0
        self._ep_path_len = 0.0
        self._prev_xy = None
        if "goal" in obs:
            self._ep_init_dist = float(obs["goal"][0])  # dist [m]
        elif info is not None and "dist" in info:
            self._ep_init_dist = float(info["dist"])  # goal 未観測時のフォールバック

    def finish(self, info: dict) -> dict:
        self._ep_count += 1
        success = bool(info.get("success", False))
        collision = bool(info.get("collision", False))
        human_collision = bool(info.get("human_collision", False))
        timeout = bool(info.get("timeout", False))

        self._total_success += int(success)
        self._total_collision += int(collision)
        self._total_human_collision += int(human_collision)
        self._success_buf.append(float(success))
        self._collision_buf.append(float(collision))
        if len(self._success_buf) > self._window:
            self._success_buf.pop(0)
        if len(self._collision_buf) > self._window:
            self._collision_buf.pop(0)

        spl = self.compute_spl(success, self._ep_init_dist, self._ep_path_len)

        return {
            "episode/reward": self._ep_reward,
            "episode/steps": self._ep_steps,
            "episode/success": float(success),
            "episode/collision": float(collision),
            "episode/human_collision": float(human_collision),
            "episode/timeout": float(timeout),
            "episode/spl": spl,
            "episode/dist_final": float(info.get("dist", 0.0)),
            "episode/success_rate": self._total_success / self._ep_count,
            "episode/collision_rate": self._total_collision / self._ep_count,
            "episode/human_collision_rate": self._total_human_collision / self._ep_count,
            "episode/count": self._ep_count,
        }

    @staticmethod
    def compute_spl(success: bool, init_dist: float, path_len: float) -> float:
        """SPL = success * (l / max(p, l))."""
        if init_dist <= 0:
            return 0.0
        return float(success) * (init_dist / max(path_len, init_dist))
