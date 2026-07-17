from pydantic import BaseModel


class ModelConfig(BaseModel):
    input_rgb: bool = True  # RGB 画像をエンコーダに入力する
    input_goal: bool = True  # ゴールベクトルをエンコーダに入力する
    # depth は env が常時発行するが現状 SAC エンコーダは未使用（将来 input_depth 等で拡張予定）


class TrainConfig(BaseModel):
    total_timesteps: int = 500_000
    run_name: str | None = "PointNav_room1"
    log_dir: str = "runs/PointNav/room1"
    log_interval: int = 1_000
    checkpoint_interval: int = 20_000

    # SAC ハイパーパラメータ
    buffer_size: int = 100_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005  # ターゲットネットワーク更新率（polyak）
    learning_rate: float = 3e-4
    learning_starts: int = 3_000
    train_freq: int = 1  # N ステップごとに更新
    gradient_steps: int = 1  # 1 更新あたりの勾配ステップ数
    target_entropy: float | str = "auto"  # "auto" → -action_dim
