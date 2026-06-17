from dataclasses import dataclass, field


@dataclass
class PointNavEnvCfg:
    stage_path: str = "sample_data/stages/corridor1_2d/stage.usda"
    robot_usd: str = (
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
        "/Assets/Isaac/6.0/Isaac/Robots/NVIDIA/Carter/carter_v1.usd"
    )
    robot_prim_path: str = "/World/Robot"
    camera_prim_path: str = "/World/Robot/chassis_link/front_cam"
    camera_resolution: tuple[int, int] = (84, 84)
    show_camera_viewport: bool = True

    physics_dt: float = 1.0 / 60.0
    rendering_dt: float = 1.0 / 30.0
    decimation: int = 6

    goal_threshold: float = 0.4  # m
    min_goal_dist: float = 2.5  # m
    max_episode_steps: int = 1000

    r_dist: float = 4.0        # 距離短縮に対する報酬係数
    r_collision: float = -20.0 # 衝突ペナルティ
    r_success: float = 20.0    # ゴール到達報酬
    r_heading: float = 0.0     # cos(angle_rel) に乗じる逐次報酬係数
    r_rollover: float = -20.0  # 転倒ペナルティ
    r_spin: float = -0.05      # 回転ペナルティ係数（r_spin × ω²）
    r_time: float = -0.025     # 毎ステップ定数ペナルティ
    r_timeout: float = 0.0     # タイムアウトペナルティ（r_timeで代替のため0）

    navmesh_exit_threshold: float = 0.3  # m
    collision_grace_steps: int = 5
    rollover_threshold: float = -0.7  # up_z < この値で転倒判定

    fixed_spawn_pos: tuple[float, float, float] | None = None
    fixed_goal_pos: tuple[float, float, float] | None = None
    random_spawn_yaw: bool = True


@dataclass
class PPOCfg:
    rollouts: int = 512
    learning_epochs: int = 5
    mini_batches: int = 4
    discount_factor: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    grad_norm_clip: float = 1.0
    ratio_clip: float = 0.2
    value_loss_scale: float = 1.0
    entropy_loss_scale: float = 0.005
    kl_threshold: float = 0.01


@dataclass
class SACCfg:
    memory_size: int = 100_000
    batch_size: int = 256
    discount_factor: float = 0.99
    polyak: float = 0.005
    learning_rate: float = 3e-4
    random_timesteps: int = 3_000
    learning_starts: int = 3_000
    grad_norm_clip: float = 1.0
    learn_entropy: bool = True
    initial_entropy_value: float = 1.0


@dataclass
class PointNavTrainCfg:
    algo: str = "ppo"  # "ppo" or "sac"
    total_timesteps: int = 500_000
    fixed_spawn_pos: tuple[float, float, float] | None = (0.4, 1.4, -1.0)
    fixed_goal_pos: tuple[float, float, float] | None = (-0.3, -3.4, -0.8)
    run_name: str | None = "PointNav-SAC"
    log_interval: int = 1_000
    checkpoint_interval: int = 10_000
    ppo: PPOCfg = field(default_factory=PPOCfg)
    sac: SACCfg = field(default_factory=SACCfg)
