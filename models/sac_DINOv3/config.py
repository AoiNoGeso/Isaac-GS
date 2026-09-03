import numpy as np
from gymnasium import spaces
from pydantic import BaseModel

from envs.observations import GoalCfg, RGBCameraCfg

_FLOAT16_KEYS = frozenset({"dino_feat"})  # メモリ節約のためfloat16で保持するキー
DINO_IMAGE_SIZE = 224  # DINOv3(dino_backbone.MODEL_ID)の学習解像度


def replay_buffer_spec(
    observation_space: spaces.Dict,
) -> tuple[dict[str, tuple[int, ...]], dict[str, type]]:
    """ReplayBufferに保持するキーとdtypeを観測空間から決定する
    (dino_featはfloat16で保持してメモリを節約する)"""
    obs_spec = {k: tuple(sp.shape) for k, sp in observation_space.spaces.items()}
    obs_dtypes = {k: np.float16 for k in observation_space.spaces if k in _FLOAT16_KEYS}
    return obs_spec, obs_dtypes


def env_overrides() -> dict:
    """EnvConfig.from_preset() へ渡すモデル固有の上書き"""
    return {
        "observations": {
            "rgb": RGBCameraCfg(resolution=(DINO_IMAGE_SIZE, DINO_IMAGE_SIZE)),
            "goal": GoalCfg(),
        }
    }


class TestConfig(BaseModel):
    """test.pyの評価設定"""

    episodes_per_stage: int = 100


class TrainConfig(BaseModel):
    """学習の実験設定"""

    stage: str  # 学習対象ステージ名(必須)
    total_timesteps: int = 1_500_000
    project_name: str | None = "Isaac-GS"
    run_name: str | None = "S2-PGB+G_corridor2_0904_buff30k"
    log_dir: str = "runs_forSI/corridor2/S2-RGB+G/0904_buff30k"
    log_interval: int = 1_000
    checkpoint_interval: int = 100_000
    val_interval: int = 50_000  # 何stepごとにバリデーションを実行するか
    val_episodes: int = 10 # バリデーション1回あたりのエピソード数10
    val_video_episodes: int = 10  # バリデーション毎に動画を撮るエピソード数(0で無効)

    # SACハイパーパラメータ
    buffer_size: int = 30_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005  # ターゲットネットワークの更新率
    learning_rate: float = 3e-4
    learning_starts: int = 3_000
    train_freq: int = 1  # 何stepごとに更新するか
    gradient_steps: int = 1  # 1回の更新あたりの勾配ステップ数
    target_entropy: float | str = "auto"  # "auto"で-action_dim
