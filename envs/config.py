from pydantic import BaseModel, Field

from envs.stage_config import STAGE_PRESETS, StagePreset, get_preset, stage_names

__all__ = [
    "RobotConfig",
    "JACKAL",
    "StagePreset",
    "STAGE_PRESETS",
    "get_preset",
    "stage_names",
    "EnvConfig",
]


class RobotConfig(BaseModel):
    name: str = "jackal"
    usd_url: str = (
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
        "/Assets/Isaac/6.0/Isaac/Robots/Clearpath/Jackal/jackal.usd"
    )
    prim_path: str = "/World/Robot"
    chassis_link: str = "base_link"
    camera_prim_path: str = (
        "/World/Robot/base_link/bumblebee_stereo_camera_frame"
        "/bumblebee_stereo_left_frame/bumblebee_stereo_left_camera"
    )

    camera_translation: tuple[float, float, float] | None = (0.0, 0.5, 0.0)  # ロボット座標系でのカメラ位置
    camera_orientation: tuple[float, float, float, float] | None = None
    left_wheel_joints: list[str] = ["front_left_wheel_joint", "rear_left_wheel_joint"]
    right_wheel_joints: list[str] = ["front_right_wheel_joint", "rear_right_wheel_joint"]
    wheel_base: float = 0.376  # トレッド幅 [m]
    wheel_radius: float = 0.1  # ホイール半径 [m]
    v_linear_max: float = 0.5  # 最大直進速度 [m/s]
    v_angular_max: float = 0.5  # 最大角速度 [rad/s]
    spawn_offset: float = 0.1  # スポーン時のz方向オフセット [m]
    rollover_threshold: float = -0.7  # 転倒判定のしきい値


JACKAL = RobotConfig()


class EnvConfig(BaseModel):
    # ステージ/ロボット
    stage_path: str  # 必須。EnvConfig.from_preset()経由での生成を推奨
    floor_prim_path: str = "/World/env/floor_mesh"
    wall_prim_path: str = "/World/env/wall_mesh"
    robot: RobotConfig = Field(default_factory=lambda: JACKAL)
    show_camera_viewport: bool = False
    camera_resolution: tuple[int, int] = (84, 84)

    # 物理/エピソード
    physics_dt: float = 1.0 / 60.0
    rendering_dt: float = 1.0 / 30.0
    decimation: int = 6

    goal_threshold: float = 0.4  # ゴール到達判定の距離 [m]
    min_goal_dist: float = 2.5  # スポーン-ゴール間の最小距離 [m]
    max_episode_steps: int = 1000

    # 報酬
    r_dist: float = 4.0  # 距離短縮に対する係数
    r_collision: float = -20.0  # 壁衝突ペナルティ
    r_success: float = 50.0  # ゴール到達報酬
    r_heading: float = 0.0  # ゴール方向を向くほど加点する係数
    r_rollover: float = -20.0  # 転倒ペナルティ
    r_spin: float = -0.05  # 過度な旋回へのペナルティ係数
    r_time: float = -0.025  # 毎ステップの定数ペナルティ
    r_timeout: float = 0.0  # タイムアウトペナルティ
    r_human_collision: float = -50.0  # 人物衝突ペナルティ

    # 判定
    collision_grace_steps: int = 5  # エピソード開始直後の衝突判定猶予ステップ数

    # スポーン (Noneはランダムサンプリング)
    fixed_spawn_pos: tuple[float, float, float] | None = None
    fixed_goal_pos: tuple[float, float, float] | None = None
    fixed_spawn_yaw_deg: float | None = None

    # 人物キャラ (num_humans=0でPointNavigationのみ, envs/human_controller/参照)
    num_humans: int = 2
    human_controller: str = "orca"  # 衝突回避アルゴリズム。今のところ"orca"のみ実装
    human_speed_range: tuple[float, float] = (0.8, 1.5)  # 歩行速度 [m/s]
    # エージェント半径 [m] (CrowdControllerに渡す)。NavMesh境界は既にnavmesh_agent_radius_cm分
    # 壁から内側にオフセットされているため、大きくしすぎると余白が二重取りになり通行不能になる。
    human_radius: float = 0.2
    human_collision_dist: float = 0.55  # ロボットとの距離ベース衝突判定しきい値 [m]
    human_min_goal_dist: float = 2.0  # 人物のスポーン-ゴール間の最小距離 [m] (min_goal_distと同じ役割)

    # NavMesh bake
    navmesh_agent_radius_cm: float = 50.0  # bake時のエージェント半径 [cm]
    navmesh_agent_height_cm: float = 200.0  # bake時のエージェント最小天井高 [cm]

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "EnvConfig":
        """StagePresetの座標をもとにEnvConfigを生成する"""
        preset = get_preset(name)
        return cls(
            stage_path=preset.stage_path,
            fixed_spawn_pos=preset.fixed_spawn_pos,
            fixed_goal_pos=preset.fixed_goal_pos,
            fixed_spawn_yaw_deg=preset.fixed_spawn_yaw_deg,
            **overrides,
        )
