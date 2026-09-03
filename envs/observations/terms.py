"""観測項目の実装。RGBCameraTermはIsaac Sim依存のimportを__init__内で遅延させること。"""

from __future__ import annotations

from typing import Any

import numpy as np

from envs.observations.base import GoalCfg, RGBCameraCfg
from envs.observations.registry import register_obs_term


@register_obs_term(RGBCameraCfg)
class RGBCameraTerm:
    def __init__(self, cfg: RGBCameraCfg, env: Any):
        from envs.sensors.camera_sensor import RGBCamera  # 遅延import: isaacsim依存

        robot = env.robot_cfg
        self._cam = RGBCamera(
            camera_prim_path=robot.camera_prim_path,
            resolution=cfg.resolution,
            translation=(
                np.array(robot.camera_translation)
                if robot.camera_translation is not None
                else None
            ),
            orientation=(
                np.array(robot.camera_orientation)
                if robot.camera_orientation is not None
                else None
            ),
        )

    def compute(self) -> np.ndarray:
        rgb = self._cam.get_rgb()
        return (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)


@register_obs_term(GoalCfg)
class GoalTerm:
    def __init__(self, cfg: GoalCfg, env: Any):
        self._env = env

    def compute(self) -> np.ndarray:
        return self._env._compute_goal_vec()
