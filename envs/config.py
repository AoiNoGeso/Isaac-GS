from pydantic import BaseModel, Field


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

    camera_translation: tuple[float, float, float] | None = (0.0, 0.5, 0.0)  # カメラのオフセット (ロボット座標系)
    camera_orientation: tuple[float, float, float, float] | None = None
    drive_model: str = "skid4"  # "skid4" | "differential"
    left_wheel_joints: list[str] = ["front_left_wheel_joint", "rear_left_wheel_joint"]
    right_wheel_joints: list[str] = ["front_right_wheel_joint", "rear_right_wheel_joint"]
    wheel_base: float = 0.376  # トレッド幅 [m] (ω→左右輪速度)
    wheel_radius: float = 0.1  # ホイール半径 [m]
    v_linear_max: float = 0.5  # 最大直進速度 [m/s]
    v_angular_max: float = 1.0  # 最大角速度 [rad/s]
    spawn_offset: float = 0.1  # spawn z-offset [m]
    rollover_threshold: float = -0.7


JACKAL = RobotConfig()


class StagePreset(BaseModel):
    stage_path: str
    fixed_spawn_pos: tuple[float, float, float] | None
    fixed_goal_pos: tuple[float, float, float] | None
    fixed_spawn_yaw_deg: float | None


STAGE_PRESETS: dict[str, StagePreset] = {
    "room1": StagePreset(
        stage_path="stages/room1/stage.usda",
        fixed_spawn_pos=(0.9, -0.19, -2.6),
        fixed_goal_pos=(-3.0, 1.6, -2.6),
        fixed_spawn_yaw_deg=137,
    ),
    "corridor1": StagePreset(
        stage_path="stages/corridor1/stage.usda",
        fixed_spawn_pos=(0.4, 1.4, -1.0),
        fixed_goal_pos=(-0.1, -1.3, -0.8),
        fixed_spawn_yaw_deg=-90,
    ),
    "corridor2": StagePreset(
        stage_path="stages/corridor2/stage.usda",
        fixed_spawn_pos=(0.0, -2.2, -1.5),
        fixed_goal_pos=(0.21, 6.9, -1.0),
        fixed_spawn_yaw_deg=90,
    ),
}


class EnvConfig(BaseModel):
    # ── ステージ/ロボット ────────────────────────────────────────
    stage_path: str = STAGE_PRESETS["corridor2"].stage_path
    floor_prim_path: str = "/World/env/floor_mesh"
    wall_prim_path: str = "/World/env/wall_mesh"
    robot: RobotConfig = Field(default_factory=lambda: JACKAL)
    show_camera_viewport: bool = True
    camera_resolution: tuple[int, int] = (84, 84)

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
    r_spin: float = -0.05  # 回転ペナルティ係数 (r_spin × ω²)
    r_time: float = -0.025  # 毎ステップ定数ペナルティ
    r_timeout: float = 0.0  # タイムアウトペナルティ (r_timeで代替のため0)
    r_human_collision: float = -20.0  # ロボット-人物 接触ペナルティ (検知で即終了)

    # ── 判定 ────────────────────────────────────────────────────
    collision_grace_steps: int = 5

    # ── スポーン ────────────────────────────────────────────────
    fixed_spawn_pos: tuple[float, float, float] | None = STAGE_PRESETS[
        "corridor2"
    ].fixed_spawn_pos
    fixed_goal_pos: tuple[float, float, float] | None = STAGE_PRESETS[
        "corridor2"
    ].fixed_goal_pos
    # None でランダム, 値(度)を指定で固定
    fixed_spawn_yaw_deg: float | None = STAGE_PRESETS["corridor2"].fixed_spawn_yaw_deg

    # ── 人物キャラ (num_humans=0 で従来の point navigation) ───────
    num_humans: int = 3
    human_speed_range: tuple[float, float] = (0.8, 1.5)  # Wander 歩行速度 [m/s]
    human_distance_range: tuple[float, float] = (3.0, 8.0)  # Wander 目標距離 [m]
    human_seed: int = 42
    reset_humans_each_episode: bool = True  # エピソード毎に再配置 (ベストエフォート)

    # ── NavMesh bake (壁+床の統合bake) ────────────────────────
    # navmesh bake時のagent半径, 壁からこの距離だけ内側にwalkable面を狭める
    # 単位はcm (omni.anim.navigation.coreの既定agentMinRadius=20=0.2m), 0以下で変更しない
    navmesh_agent_radius_cm: float = 50.0
    # navmesh bake時に要求する天井までの最小高さ, これより低い箇所は歩行不可と判定される
    # 単位はcm (既定agentMinHeight=200=2.0m), 0以下で変更しない
    navmesh_agent_height_cm: float = 200.0
