import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.isaac_env import PointNavIsaacEnv
from tasks.point_navigation.config import PointNavEnvCfg


class PointNavGymEnv(gym.Env):
    def __init__(self, cfg: PointNavEnvCfg | None = None):
        super().__init__()
        self.cfg = cfg or PointNavEnvCfg()
        W, H = self.cfg.camera_resolution
        self.observation_space = spaces.Dict({
            "rgb":  spaces.Box(0.0, 1.0, shape=(3, H, W), dtype=np.float32),
            "goal": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self._env: PointNavIsaacEnv | None = None

    def _lazy_init(self):
        if self._env is None:
            self._env = PointNavIsaacEnv(self.cfg)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._lazy_init()
        return self._env.reset(), {}

    def step(self, action: np.ndarray):
        return self._env.step(action)

    def render(self):
        pass

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
