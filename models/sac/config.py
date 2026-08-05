import numpy as np
from pydantic import BaseModel


class ModelConfig(BaseModel):
    input_rgb: bool = True  # RGB画像をエンコーダに入力する
    input_goal: bool = True  # ゴールベクトルをエンコーダに入力する
    # depthは常時取得されるが未使用(将来input_depth等で拡張予定)


def replay_buffer_spec(
    model_cfg: ModelConfig, camera_resolution: tuple[int, int]
) -> tuple[dict[str, tuple[int, ...]], dict[str, type]]:
    """ReplayBufferに保持するキーとdtypeをModelConfigから決定する。
    depthは未使用のため除外し、rgbはuint8で保持してメモリを節約する。"""
    W, H = camera_resolution
    obs_spec: dict[str, tuple[int, ...]] = {}
    obs_dtypes: dict[str, type] = {}
    if model_cfg.input_rgb:
        obs_spec["rgb"] = (3, H, W)
        obs_dtypes["rgb"] = np.uint8
    if model_cfg.input_goal:
        obs_spec["goal"] = (2,)
    return obs_spec, obs_dtypes


class TestConfig(BaseModel):
    """test.pyの評価設定"""

    episodes_per_stage: int = 100


class TrainConfig(BaseModel):
    """学習の実験設定"""

    stage: str  # 学習対象ステージ名(必須)
    total_timesteps: int = 300_000
    run_name: str | None = "S2-PGB+G_corridor2_0805"
    log_dir: str = "runs/SocialNav-RGB+Goal/corridor2/0805"
    log_interval: int = 1_000
    checkpoint_interval: int = 20_000
    val_interval: int = 50_000  # 何stepごとにバリデーションを実行するか
    val_episodes: int = 100  # バリデーション1回あたりのエピソード数
    val_video_episodes: int = 1  # バリデーション毎に動画を撮るエピソード数(0で無効)

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
