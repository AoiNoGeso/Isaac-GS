from pydantic import BaseModel, ConfigDict, Field

from envs.observations import GoalCfg, ObsTermCfg, RGBCameraCfg
from envs.stage_config import STAGE_PRESETS, StagePreset, get_preset, stage_names

__all__ = [
    "RobotConfig",
    "JACKAL",
    "CYLINDER_ROBOT",
    "StagePreset",
    "STAGE_PRESETS",
    "get_preset",
    "stage_names",
    "EnvConfig",
]


class RobotConfig(BaseModel):
    name: str = "jackal"
    drive_mode: str = "differential"  # "differential"(Jackal本体、ホイールjoint角速度目標経由) |
    # "direct_velocity"(単純な円柱剛体、cmd_vel[v_x, omega]を車輪動力学を介さず本体速度へ直接反映)
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
    footprint_radius: float = 0.35  # 衝突回避(CrowdController)に登録する実効半径 [m](Jackal footprint 0.508x0.430mの外接円概算、
    # drive_mode="direct_velocity"の場合は円柱の半径としても使う)
    cylinder_height: float = 0.3  # drive_mode="direct_velocity"時、生成する円柱剛体の高さ [m]


JACKAL = RobotConfig()

CYLINDER_ROBOT = RobotConfig(
    name="cylinder",
    drive_mode="direct_velocity",
    camera_prim_path="/World/Robot/Camera",
    camera_translation=(0.0, 0.0, 0.15),  # 円柱の中心から少し上、正面向き
    camera_orientation=None,
    v_linear_max=0.5,
    v_angular_max=0.5,
    spawn_offset=0.05,
    footprint_radius=0.3,
    cylinder_height=0.3,
)


class EnvConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ステージ/ロボット
    stage_path: str  # 必須。EnvConfig.from_preset()経由での生成を推奨
    floor_prim_path: str = "/World/env/floor_mesh"
    wall_prim_path: str = "/World/env/wall_mesh"
    robot: RobotConfig = Field(default_factory=lambda: JACKAL)
    observations: dict[str, ObsTermCfg] = Field(
        default_factory=lambda: {"rgb": RGBCameraCfg(), "goal": GoalCfg()}
    )

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
    human_speed_range: tuple[float, float] = (1.0, 1.5)  # 歩行速度 [m/s]
    human_radius: float = 0.2
    human_avoidance_margin: float = 0.05  # human_radius+human_avoidance_marginはnavmesh_agent_radius_cm以下に収めること
    human_collision_dist: float = 0.55  # ロボットとの距離ベース衝突判定しきい値 [m]
    human_min_goal_dist: float = 2.0  # 人物のスポーン-ゴール間の最小距離 [m] (min_goal_distと同じ役割)
    human_robot_min_spawn_dist: float = 1.0  # 人物スポーン位置とロボットスポーン位置の最小距離 [m] (1step目での衝突判定を防ぐ)
    human_anim_stride: int = 6  # IRAへのmove_to/set_speed再送信、およびORCAへのIRA実位置・実速度同期を何物理サブステップに1回行うか
    human_avatar_character: str | None = None  # IRAキャラクターUSD URL。Noneなら`Isaac/People/Characters/`から自動検出
    human_motion_library: str = "Isaac/People/MotionLibrary/HumanMotionLibrary.usd"
    human_ira_lookahead_s: float = 2.0  # move_toターゲットの先読み時間(ORCA速度×この秒数、reissue_policy="reissue"時のみ使用)
    human_ira_reissue_policy: str = "reissue"  # "hold"(IRA単体用。ORCA方向には追従しない) | "reissue"(40万step耐久検証済み。ORCA指令を再送信)
    human_ira_min_command_speed: float = 0.00  # この閾値未満のORCA速度はidle()として扱う [m/s]

    # NavMesh bake
    navmesh_agent_radius_cm: float = 40.0  # bake時のエージェント半径 [cm] (ORCA半径human_radius+human_avoidance_marginはこれ以下に収めること。上回るとORCAが要求する間隔がNavMesh上の通路幅を超え、agentがidle()に落ち続けて長時間停止する)
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
