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

        obs: dict = {}
        if self.cfg.input_rgb:
            obs["rgb"] = spaces.Box(0.0, 1.0, shape=(3, H, W), dtype=np.float32)
        if self.cfg.input_goal:
            obs["goal"] = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        assert obs, "input_rgb と input_goal の少なくとも一方は True にしてください"

        self.observation_space = spaces.Dict(obs)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self._env: PointNavIsaacEnv | None = None

    def _lazy_init(self):
        if self._env is None:
            self._env = PointNavIsaacEnv(self.cfg)

    def _filter_obs(self, obs: dict) -> dict:
        return {k: obs[k] for k in self.observation_space.spaces}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._lazy_init()
        return self._filter_obs(self._env.reset()), {}

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self._env.step(action)
        return self._filter_obs(obs), reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
