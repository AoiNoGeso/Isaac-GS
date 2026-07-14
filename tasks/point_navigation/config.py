from pydantic import BaseModel


class EnvConfig(BaseModel):
    # ── ステージ/ロボット ────────────────────────────────────────
    stage_path: str = "sample_data/stages/room1/stage.usda"
    robot_usd: str = (
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
        "/Assets/Isaac/6.0/Isaac/Robots/NVIDIA/Carter/carter_v1.usd"
    )
    robot_prim_path: str = "/World/Robot"
    camera_prim_path: str = "/World/Robot/chassis_link/front_cam"
    show_camera_viewport: bool = True

    # ── 物理/エピソード ──────────────────────────────────────────
    physics_dt: float = 1.0 / 60.0
    rendering_dt: float = 1.0 / 30.0
    decimation: int = 6

    goal_threshold: float = 0.4  # m
    min_goal_dist: float = 2.5  # m
    max_episode_steps: int = 1000

    # ── 報酬 ────────────────────────────────────────────────────
    r_dist: float = 4.0  # 距離短縮に対する報酬係数
    r_collision: float = -20.0  # 衝突ペナルティ
    r_success: float = 50.0  # ゴール到達報酬
    r_heading: float = 0.0  # cos(angle_rel) に乗じる逐次報酬係数
    r_rollover: float = -20.0  # 転倒ペナルティ
    r_spin: float = -0.05  # 回転ペナルティ係数（r_spin × ω²）
    r_time: float = -0.025  # 毎ステップ定数ペナルティ
    r_timeout: float = 0.0  # タイムアウトペナルティ（r_timeで代替のため0）
    r_human_collision: float = -20.0  # ロボット-人物 接近ペナルティ（検知で即終了）

    # ── 判定 ────────────────────────────────────────────────────
    navmesh_exit_threshold: float = 0.3  # m
    collision_grace_steps: int = 5
    rollover_threshold: float = -0.7  # up_z < この値で転倒判定

    # ── スポーン（room1 既定）────────────────────────────────────
    # corridor1_2d
    # fixed_spawn_pos: tuple[float, float, float] | None = (0.4, 1.4, -1.0)
    # fixed_goal_pos: tuple[float, float, float] | None = (-0.1, -1.3, -0.8)
    # room1
    fixed_spawn_pos: tuple[float, float, float] | None = (0.9, -0.19, -2.6)
    fixed_goal_pos: tuple[float, float, float] | None = (-3.0, 1.6, -2.6)
    # None でランダム、値（度）を指定で固定 (corridor1_2d: -90, room1: 137)
    fixed_spawn_yaw_deg: float | None = 137

    # ── 人物キャラ（num_humans=0 で従来の point navigation）───────
    num_humans: int = 0
    human_speed_range: tuple[float, float] = (0.8, 1.5)  # Wander 歩行速度 [m/s]
    human_distance_range: tuple[float, float] = (3.0, 8.0)  # Wander 目標距離 [m]
    human_seed: int = 42
    reset_humans_each_episode: bool = True  # エピソード毎に再配置（ベストエフォート）

    # ── NavMesh bake（壁＋床の統合 bake）────────────────────────
    # navmesh bake 時の agent 半径。壁からこの距離だけ内側に walkable 面を狭める。
    # 単位は cm（omni.anim.navigation.core の既定 agentMinRadius=20=0.2m）。0 以下で変更しない。
    navmesh_agent_radius_cm: float = 20.0


class ModelConfig(BaseModel):
    input_rgb: bool = True  # RGB 画像をエンコーダに入力する
    input_goal: bool = True  # ゴールベクトルをエンコーダに入力する
    camera_resolution: tuple[int, int] = (84, 84)


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
