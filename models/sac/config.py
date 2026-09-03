import numpy as np
from gymnasium import spaces
from pydantic import BaseModel

_UINT8_KEYS = frozenset({"rgb"})  # [0,1] floatで受け取りuint8で保持するキー


def replay_buffer_spec(
    observation_space: spaces.Dict,
) -> tuple[dict[str, tuple[int, ...]], dict[str, type]]:
    """ReplayBufferに保持するキーとdtypeを観測空間から決定する
    (rgbはuint8で保持してメモリを節約する)"""
    obs_spec = {k: tuple(sp.shape) for k, sp in observation_space.spaces.items()}
    obs_dtypes = {k: np.uint8 for k in observation_space.spaces if k in _UINT8_KEYS}
    return obs_spec, obs_dtypes


def env_overrides() -> dict:
    """EnvConfig.from_preset() へ渡すモデル固有の上書き(sacでは無し)"""
    return {}


class TestConfig(BaseModel):
    """test.pyの評価設定"""

    episodes_per_stage: int = 100


class TrainConfig(BaseModel):
    """学習の実験設定"""

    stage: str  # 学習対象ステージ名(必須)
    total_timesteps: int = 1_500_000
    project_name: str | None = "Isaac-GS"
    run_name: str | None = "S1-PGB+G_corridor2_0824"
    log_dir: str = "runs/S1-RGB+G/corridor2/0824"
    log_interval: int = 1_000
    checkpoint_interval: int = 50_000
    val_interval: int = 100_000  # 何stepごとにバリデーションを実行するか
    val_episodes: int = 10 # バリデーション1回あたりのエピソード数10
    val_video_episodes: int = 10  # バリデーション毎に動画を撮るエピソード数(0で無効)

    # SACハイパーパラメータ
    buffer_size: int = 100_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005  # ターゲットネットワークの更新率
    learning_rate: float = 3e-4
    learning_starts: int = 3_000
    train_freq: int = 1  # 何stepごとに更新するか
    gradient_steps: int = 1  # 1回の更新あたりの勾配ステップ数
    target_entropy: float | str = "auto"  # "auto"で-action_dim
