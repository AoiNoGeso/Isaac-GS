from envs.config import EnvConfig, RobotConfig, STAGE_PRESETS

__all__ = [
    "PointNavGymEnv",
    "PointNavIsaacEnv",
    "EnvConfig",
    "RobotConfig",
    "STAGE_PRESETS",
]

_LAZY = {"PointNavGymEnv", "PointNavIsaacEnv"}


def __getattr__(name: str):
    """PointNavGymEnv/PointNavIsaacEnv(→torch/gymnasium/isaacsim一式)を遅延import する
    (PEP 562)。`envs.config`だけを使いたい呼び出し元(argparseの選択肢生成等)がこれらの
    重量級依存を巻き込まないようにするため"""
    if name in _LAZY:
        from envs import isaac_env

        value = getattr(isaac_env, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
