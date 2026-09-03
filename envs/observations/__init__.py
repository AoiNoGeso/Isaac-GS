from envs.observations.base import GoalCfg, ObsTerm, ObsTermCfg, RGBCameraCfg
from envs.observations.manager import ObservationManager, observation_space_from_cfg
from envs.observations.registry import build_term, register_obs_term

# termsをimportしてレジストリへの登録を発火させる
from envs.observations import terms  # noqa: F401,E402

__all__ = [
    "ObsTermCfg",
    "ObsTerm",
    "RGBCameraCfg",
    "GoalCfg",
    "ObservationManager",
    "observation_space_from_cfg",
    "register_obs_term",
    "build_term",
]
