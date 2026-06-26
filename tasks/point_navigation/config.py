from pydantic import BaseModel, Field


class PointNavEnvCfg(BaseModel):
    stage_path: str = "sample_data/stages/corridor1_2d/stage.usda"
    robot_usd: str = (
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
        "/Assets/Isaac/6.0/Isaac/Robots/NVIDIA/Carter/carter_v1.usd"
    )
    robot_prim_path: str = "/World/Robot"
    camera_prim_path: str = "/World/Robot/chassis_link/front_cam"
    camera_resolution: tuple[int, int] = (84, 84)
    show_camera_viewport: bool = True

    input_rgb: bool = True  # RGB 画像をエンコーダに入力する
    input_goal: bool = True  # ゴールベクトルをエンコーダに入力する

    physics_dt: float = 1.0 / 60.0
    rendering_dt: float = 1.0 / 30.0
    decimation: int = 6

    goal_threshold: float = 0.4  # m
    min_goal_dist: float = 2.5  # m
    max_episode_steps: int = 1000

    r_dist: float = 4.0  # 距離短縮に対する報酬係数
    r_collision: float = -20.0  # 衝突ペナルティ
    r_success: float = 20.0  # ゴール到達報酬
    r_heading: float = 0.0  # cos(angle_rel) に乗じる逐次報酬係数
    r_rollover: float = -20.0  # 転倒ペナルティ
    r_spin: float = -0.05  # 回転ペナルティ係数（r_spin × ω²）
    r_time: float = -0.025  # 毎ステップ定数ペナルティ
    r_timeout: float = 0.0  # タイムアウトペナルティ（r_timeで代替のため0）

    navmesh_exit_threshold: float = 0.3  # m
    collision_grace_steps: int = 5
    rollover_threshold: float = -0.7  # up_z < この値で転倒判定

    fixed_spawn_pos: tuple[float, float, float] | None = (0.4, 1.4, -1.0)
    fixed_goal_pos: tuple[float, float, float] | None = (-0.3, -3.4, -0.8)
    random_spawn_yaw: bool = False


class SACCfg(BaseModel):
    buffer_size: int = 100_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005  # ターゲットネットワーク更新率（polyak）
    learning_rate: float = 3e-4
    learning_starts: int = 3_000
    train_freq: int = 1  # N ステップごとに更新
    gradient_steps: int = 1  # 1 更新あたりの勾配ステップ数
    target_entropy: float | str = "auto"  # "auto" → -action_dim


class PointNavTrainCfg(BaseModel):
    total_timesteps: int = 300_000
    input_rgb: bool = True
    input_goal: bool = True
    run_name: str | None = "PointNav-SAC-RGB+Goal"
    log_dir: str = "runs/PointNav-SAC-RGB+Goal_0626"
    log_interval: int = 1_000
    checkpoint_interval: int = 20_000
    sac: SACCfg = Field(default_factory=SACCfg)
