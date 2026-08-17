from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.config import EnvConfig, RobotConfig
from envs.geometry import goal_vec, quat_to_yaw
from envs.human_controller import HumanManager

# 落下判定
FALL_Z_THRESHOLD = -50.0

# 物理演算破綻とみなす線形速度 [m/s]
VELOCITY_EXPLOSION_THRESHOLD = 10.0


def _to_numpy(value) -> np.ndarray:
    """torch.Tensor / numpy のどちらで返ってきてもnumpy配列に揃える"""
    return value.cpu().numpy() if hasattr(value, "cpu") else np.array(value, dtype=np.float32)


class PointNavIsaacEnv:
    """Isaac Sim上でロボットを操作するPoint Navigation環境本体"""

    def __init__(self, env_cfg: EnvConfig):
        self.env_cfg = env_cfg
        self.robot_cfg: RobotConfig = env_cfg.robot
        self._step_count = 0
        self._goal_pos = np.zeros(3, dtype=np.float32)
        self._prev_dist = 0.0
        self._has_humans = env_cfg.num_humans > 0
        self._last_omega = 0.0
        self._contact_bodies_error_logged = False
        self._velocity_explosion_error_logged = False
        self._setup()

    # ------------------------------------------------------------------
    # 公開アクセサ
    # ------------------------------------------------------------------
    @property
    def robot_pos(self) -> np.ndarray:
        return self._get_robot_pos()

    @property
    def robot_quat(self) -> np.ndarray:
        return self._get_robot_quat()

    @property
    def goal_pos(self) -> np.ndarray:
        return self._goal_pos

    @property
    def goal_vec(self) -> np.ndarray:
        return self._compute_goal_vec()

    @property
    def human_positions_xy(self) -> list[tuple[float, float]]:
        return self._human_mgr.get_world_positions_xy() if self._has_humans else []

    @property
    def initial_goal_dist(self) -> float:
        return self._prev_dist

    # ------------------------------------------------------------------
    # セットアップ
    # ------------------------------------------------------------------
    def _setup(self):
        import omni.kit.app
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import Articulation
        from isaacsim.core.utils.extensions import enable_extension
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.sensors.experimental.physics import Contact, ContactSensor
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

        from envs.sensors.camera_sensor import RGBCamera

        robot = self.robot_cfg
        kit_app = omni.kit.app.get_app()

        enable_extension("omni.anim.navigation.bundle")
        if self._has_humans:
            # 人物の高さ追従用カプセル(envs/human_controller/human_manager.py)を動かすのに必要。
            enable_extension("omni.physx.cct")
        for _ in range(10):
            kit_app.update()

        import omni.anim.navigation.core as nav

        self._inav = nav.acquire_interface()

        add_reference_to_stage(usd_path=self.env_cfg.stage_path, prim_path="/World/env")
        add_reference_to_stage(usd_path=robot.usd_url, prim_path=robot.prim_path)

        # LiDARセンサー機能は維持したまま, デバッグ用の描画光線のみ非表示にする
        robot_stage = omni.usd.get_context().get_stage()
        for p in Usd.PrimRange(robot_stage.GetPrimAtPath(robot.prim_path)):
            if p.GetTypeName() == "Lidar":
                draw_lines_attr = p.GetAttribute("drawLines")
                if draw_lines_attr.IsValid():
                    draw_lines_attr.Set(False)

        self._world = World(
            physics_dt=self.env_cfg.physics_dt,
            rendering_dt=self.env_cfg.rendering_dt,
            stage_units_in_meters=1.0,
        )
        self._world.reset()

        stage = omni.usd.get_context().get_stage()
        self._stage = stage
        physics_scene = UsdPhysics.Scene.Get(stage, "/physicsScene")
        if not physics_scene:
            physics_scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
        physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        physics_scene.CreateGravityMagnitudeAttr().Set(9.81)

        # 無摩擦(0.0)ステージ対策, 横滑り防止のため摩擦係数を上書き
        env_mat_prim = stage.GetPrimAtPath("/World/env/PhysicsMaterial")
        if env_mat_prim.IsValid():
            env_mat = UsdPhysics.MaterialAPI(env_mat_prim)
            env_mat.CreateStaticFrictionAttr().Set(0.8)
            env_mat.CreateDynamicFrictionAttr().Set(0.6)
            PhysxSchema.PhysxMaterialAPI(env_mat_prim).CreateFrictionCombineModeAttr().Set("max")

        self._bake_navmesh(stage)

        self._robot = Articulation(prim_paths_expr=robot.prim_path)
        self._robot.initialize()

        dof_names = list(self._robot.dof_names)
        try:
            self._left_wheel_idx = [dof_names.index(j) for j in robot.left_wheel_joints]
            self._right_wheel_idx = [dof_names.index(j) for j in robot.right_wheel_joints]
        except ValueError as e:
            import carb

            carb.log_error(
                f"[PointNavIsaacEnv] ホイール joint 名が dof_names に見つかりません: {e}"
                f" dof_names={dof_names}"
            )
            raise

        # chassis_linkにPhysxContactReportAPIを付与しContactSensorを設置
        chassis_prim_path = f"{robot.prim_path}/{robot.chassis_link}"
        chassis_prim = stage.GetPrimAtPath(chassis_prim_path)
        contact_report = PhysxSchema.PhysxContactReportAPI.Apply(chassis_prim)
        contact_report.CreateThresholdAttr().Set(0)
        contact_authoring = Contact.create(
            f"{chassis_prim_path}/contact_sensor",
            min_threshold=0.0,
            max_threshold=100000.0,
            radius=-1.0,
        )
        self._contact_sensor = ContactSensor(contact_authoring)
        self._contact_sensor.add_raw_contact_data_to_frame()

        self._camera = RGBCamera(
            camera_prim_path=robot.camera_prim_path,
            resolution=self.env_cfg.camera_resolution,
            translation=(
                np.array(robot.camera_translation)
                if robot.camera_translation is not None
                else None
            ),
            orientation=(
                np.array(robot.camera_orientation)
                if robot.camera_orientation is not None
                else None
            ),
        )

        if self.env_cfg.show_camera_viewport:
            self._setup_camera_viewport()

        self._human_mgr = HumanManager(self.env_cfg, self._world, self._inav)
        if self._has_humans:
            self._human_mgr.inject_humans(stage)

    def _bake_navmesh(self, stage) -> None:
        """床・壁メッシュを一時的に可視化してNavMeshをbakeする
        bake中だけagent半径・天井高の設定を上書きし、終了後に元の値へ戻す"""
        from pxr import UsdGeom

        bake_settings = {}
        if self.env_cfg.navmesh_agent_radius_cm > 0:
            radius = float(self.env_cfg.navmesh_agent_radius_cm)
            bake_settings["/exts/omni.anim.navigation.core/navMesh/config/agentMinRadius"] = radius
            bake_settings["/exts/omni.anim.navigation.core/navMesh/config/agentMaxRadius"] = radius
        if self.env_cfg.navmesh_agent_height_cm > 0:
            bake_settings["/exts/omni.anim.navigation.core/navMesh/config/agentMinHeight"] = float(
                self.env_cfg.navmesh_agent_height_cm
            )

        settings = None
        original = {}
        if bake_settings:
            import carb

            settings = carb.settings.get_settings()
            original = {k: settings.get(k) for k in bake_settings}
            for k, v in bake_settings.items():
                settings.set(k, v)

        # NavMeshは可視なメッシュからしか生成されないため、bakeの間だけ表示する
        vis_prims = []
        for path in (self.env_cfg.floor_prim_path, self.env_cfg.wall_prim_path):
            p = stage.GetPrimAtPath(path)
            if p.IsValid():
                UsdGeom.Imageable(p).MakeVisible()
                vis_prims.append(p)
        for _ in range(10):
            self._world.step(render=True)

        self._inav.start_navmesh_baking_and_wait()

        for p in vis_prims:
            UsdGeom.Imageable(p).MakeInvisible()
        if settings is not None:
            for k, v in original.items():
                if v is not None:
                    settings.set(k, v)

    def _setup_camera_viewport(self):
        try:
            import omni.kit.viewport.utility as vp_util

            vp_win = vp_util.create_viewport_window("Robot Camera", width=320, height=240)
            vp_win.viewport_api.set_active_camera(self.robot_cfg.camera_prim_path)
        except Exception as e:
            import carb

            carb.log_warn(f"[PointNavIsaacEnv] Camera viewport window skipped: {e}")

    # ------------------------------------------------------------------
    # reset / step
    # ------------------------------------------------------------------
    def reset(self) -> dict:
        """ロボット・ゴール・人物キャラを新しいエピソード用に配置し初期観測を返す"""
        self._step_count = 0
        self._last_omega = 0.0

        if self.env_cfg.fixed_spawn_pos is not None:
            robot_pos = np.array(self.env_cfg.fixed_spawn_pos, dtype=np.float32)
        else:
            robot_pos = self._sample_navmesh_point()

        if self.env_cfg.fixed_goal_pos is not None:
            self._goal_pos = np.array(self.env_cfg.fixed_goal_pos, dtype=np.float32)
            self._prev_dist = self._dist_to_goal(robot_pos)
        else:
            best_goal, best_dist = robot_pos.copy(), 0.0
            for _ in range(50):
                goal = self._sample_navmesh_point()
                dist = float(np.linalg.norm(goal[[0, 1]] - robot_pos[[0, 1]]))
                if dist > best_dist:
                    best_goal, best_dist = goal, dist
                if dist >= self.env_cfg.min_goal_dist:
                    break
            self._goal_pos = best_goal
            self._prev_dist = best_dist

        spawn_yaw = self._sample_spawn_yaw()

        for _ in range(10):
            self._teleport_robot(robot_pos, yaw=spawn_yaw)
            self._world.step(render=False)
            if self._check_velocity_explosion():
                self._recover_physics()
                robot_pos = self._sample_navmesh_point()

        for _ in range(5):
            if not self._check_rollover():
                break
            robot_pos = self._sample_navmesh_point()
            self._teleport_robot(robot_pos, yaw=spawn_yaw)
            self._world.step(render=False)

        self._teleport_robot(robot_pos, yaw=spawn_yaw)
        self._world.step(render=True)

        if self._has_humans:
            self._human_mgr.reset_humans()

        return self._get_obs()

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """行動[v_x, ω]を1ステップ実行し(obs, reward, terminated, truncated, info)を返す"""
        self._apply_action(action)

        result = None
        human_hit = False
        for i in range(self.env_cfg.decimation):
            if self._has_humans:
                pos_xy = tuple(self._get_robot_pos()[[0, 1]])
                vel_xy = self._get_robot_velocity_xy()
                self._human_mgr.pre_physics_step(self.env_cfg.physics_dt, pos_xy, vel_xy)
            self._world.step(render=(i == self.env_cfg.decimation - 1))
            if self._has_humans:
                self._human_mgr.post_physics_step()

            human_hit = human_hit or self._check_human_contact()

            if self._check_velocity_explosion():
                self._recover_physics()
                result = self._collision_termination()
                break

            if not self._in_collision_grace() and self._check_wall_contact():
                self._stop_wheels()
                result = self._collision_termination()
                break

        if result is None:
            self._step_count += 1
            obs = self._get_obs()
            reward, info = self._compute_reward()
            terminated = info["success"] or info["collision"]
            truncated = self._step_count >= self.env_cfg.max_episode_steps
            if truncated:
                reward += self.env_cfg.r_timeout
                info["timeout"] = True
            result = (obs, float(reward), terminated, truncated, info)

        obs, reward, terminated, truncated, info = result

        human_collision = human_hit or self._check_human_contact()
        info["human_collision"] = human_collision
        if human_collision and not info["success"]:
            # 壁衝突等と同時発生時は人物衝突を優先(ゴール到達時のみ上書きしない)
            reward = float(self.env_cfg.r_human_collision)
            terminated = True
            truncated = False
            info["collision"] = True

        return obs, reward, terminated, truncated, info

    def _apply_action(self, action: np.ndarray) -> None:
        """行動[v_x, ω]を差動二輪のjoint角速度目標へ変換して指令する"""
        robot = self.robot_cfg
        v_x = float(np.clip(action[0], -1.0, 1.0)) * robot.v_linear_max
        omega = float(np.clip(action[1], -1.0, 1.0)) * robot.v_angular_max
        self._last_omega = omega

        v_left = v_x - omega * robot.wheel_base / 2.0
        v_right = v_x + omega * robot.wheel_base / 2.0
        # ホイール接地点の線形速度 [m/s] → joint角速度 [rad/s]
        w_left = v_left / robot.wheel_radius
        w_right = v_right / robot.wheel_radius

        vel_target = np.zeros(self._robot.num_dof, dtype=np.float32)
        for idx in self._left_wheel_idx:
            vel_target[idx] = w_left
        for idx in self._right_wheel_idx:
            vel_target[idx] = w_right
        self._robot.set_joint_velocity_targets(velocities=vel_target[np.newaxis, :])

    def _stop_wheels(self) -> None:
        self._robot.set_joint_velocity_targets(
            velocities=np.zeros((1, self._robot.num_dof), dtype=np.float32)
        )

    def _collision_termination(self) -> tuple[dict, float, bool, bool, dict]:
        """衝突・物理破綻でエピソードを即座に打ち切る際の戻り値を組み立てる"""
        obs = self._get_obs()
        pos = self._get_robot_pos()
        dist = self._dist_to_goal(pos)
        self._prev_dist = dist
        info = self._episode_info(pos, dist, success=False, collision=True)
        return obs, float(self.env_cfg.r_collision), True, False, info

    def _episode_info(
        self, pos: np.ndarray, dist: float, success: bool, collision: bool, timeout: bool = False
    ) -> dict:
        """全ての終了経路で共通のinfo辞書を組み立てる"""
        return {
            "success": success,
            "collision": collision,
            "timeout": timeout,
            "dist": dist,
            "robot_xy": pos[[0, 1]],
        }

    def _get_obs(self) -> dict:
        """rgb/goalの2キーからなる観測を生成する"""
        rgb = self._camera.get_rgb()
        return {
            "rgb": (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1),
            "goal": self._compute_goal_vec(),
        }

    def _dist_to_goal(self, pos: np.ndarray) -> float:
        """ロボットからゴールまでの水平距離 [m]"""
        return float(np.linalg.norm(self._goal_pos[[0, 1]] - pos[[0, 1]]))

    def _compute_goal_vec(self) -> np.ndarray:
        """ロボットから見たゴールまでの[距離, 相対角度/π]を計算する"""
        pos, quat = self._get_robot_pose()
        yaw = quat_to_yaw(*quat)
        return goal_vec(pos[0], pos[1], yaw, self._goal_pos[0], self._goal_pos[1])

    def _compute_reward(self) -> tuple[float, dict]:
        """今フレームの報酬とinfo(success/collision等)を計算する"""
        pos = self._get_robot_pos()
        cur_dist = self._dist_to_goal(pos)
        success = cur_dist < self.env_cfg.goal_threshold
        collision = self._check_collision()
        rollover = self._check_rollover()
        angle_rel = float(self._compute_goal_vec()[1]) * np.pi
        reward = (
            self.env_cfg.r_dist * (self._prev_dist - cur_dist)
            + self.env_cfg.r_heading * float(np.cos(angle_rel))
            + self.env_cfg.r_collision * float(collision)
            + self.env_cfg.r_rollover * float(rollover)
            + self.env_cfg.r_success * float(success)
            + self.env_cfg.r_spin * float(self._last_omega**2)
            + self.env_cfg.r_time
        )
        self._prev_dist = cur_dist
        return reward, self._episode_info(
            pos, cur_dist, success=success, collision=collision or rollover
        )

    # ------------------------------------------------------------------
    # 衝突・破綻の判定
    # ------------------------------------------------------------------
    def _in_collision_grace(self) -> bool:
        """エピソード開始直後は衝突判定を猶予する"""
        return self._step_count < self.env_cfg.collision_grace_steps

    def _check_rollover(self) -> bool:
        """ロボットが転倒しているかを姿勢から判定する"""
        _, qx, qy, _ = self._get_robot_quat()
        return float(1.0 - 2.0 * (qx * qx + qy * qy)) < self.robot_cfg.rollover_threshold

    def _get_robot_velocity_xy(self) -> tuple[float, float]:
        """ロボットの現在のワールド線形速度(XY)。人物側ORCAの選好速度同期用"""
        try:
            linvel = self._robot.get_linear_velocities()
            if linvel is None:
                return (0.0, 0.0)
            vx, vy = _to_numpy(linvel[0])[:2]
            return (float(vx), float(vy))
        except Exception:
            return (0.0, 0.0)

    def _check_velocity_explosion(self) -> bool:
        """物理演算が破綻して速度や位置が異常値になっていないか確認する"""
        try:
            linvel = self._robot.get_linear_velocities()
            if linvel is not None and float(np.max(np.abs(linvel))) > VELOCITY_EXPLOSION_THRESHOLD:
                return True
            return not all(np.isfinite(self._get_robot_pos()))
        except Exception:
            if not self._velocity_explosion_error_logged:
                import carb

                carb.log_error("[PointNavIsaacEnv] _check_velocity_explosion で例外発生")
                self._velocity_explosion_error_logged = True
            return False

    def _contact_bodies(self) -> list[tuple[str, str]]:
        """ロボットの接触センサーが検知した接触相手のprimパス一覧を返す"""
        from pxr import PhysicsSchemaTools

        try:
            return [
                (
                    str(PhysicsSchemaTools.intToSdfPath(int(contact["body0"]))),
                    str(PhysicsSchemaTools.intToSdfPath(int(contact["body1"]))),
                )
                for contact in self._contact_sensor.get_raw_data()
            ]
        except Exception:
            if not self._contact_bodies_error_logged:
                import carb

                carb.log_error("[PointNavIsaacEnv] _contact_bodies で例外発生")
                self._contact_bodies_error_logged = True
            return []

    def _check_wall_contact(self) -> bool:
        """壁と接触しているか判定する"""
        return any(
            "wall_mesh" in body0 or "wall_mesh" in body1
            for body0, body1 in self._contact_bodies()
        )

    def _check_human_contact(self) -> bool:
        """人物との接触判定。ロボットと各人物のワールド座標間の距離ベースで判定する
        (ai4animationpyで駆動する人物はkinematicでContactSensorだけでは検知漏れが起きるため、
        距離ベースの判定に一本化している)"""
        if not self._has_humans:
            return False
        pos = self._get_robot_pos()
        return any(
            float(np.hypot(hx - pos[0], hy - pos[1])) < self.env_cfg.human_collision_dist
            for hx, hy in self._human_mgr.get_world_positions_xy()
        )

    def _check_collision(self) -> bool:
        """壁衝突または落下による衝突判定(エピソード開始直後は猶予)"""
        if self._in_collision_grace():
            return False
        if self._check_wall_contact():
            return True
        return float(self._get_robot_pos()[2]) < FALL_Z_THRESHOLD

    # ------------------------------------------------------------------
    # 位置・姿勢の操作
    # ------------------------------------------------------------------
    def _recover_physics(self):
        """物理演算破綻からの復帰。安全な位置へテレポートし直す"""
        safe_pos = self._sample_navmesh_point()
        if not np.all(np.isfinite(safe_pos)):
            safe_pos = self._get_robot_pos()
        for _ in range(20):
            self._teleport_robot(safe_pos)
            self._world.step(render=False)
            if not self._check_velocity_explosion():
                break

    def _sample_navmesh_point(self) -> np.ndarray:
        """NavMesh上のランダムな点を返す(ロボットのスポーン/ゴール用)"""
        nm = self._inav.get_navmesh()
        if nm is not None:
            for _ in range(20):
                p = nm.query_random_point()
                pos = np.array([p[0], p[1], p[2]], dtype=np.float32)
                if np.all(np.isfinite(pos)):
                    pos[2] += self.robot_cfg.spawn_offset
                    return pos
        return np.zeros(3, dtype=np.float32)

    def _sample_spawn_yaw(self) -> float:
        """スポーン時のyaw [rad](固定値が未設定ならランダム)"""
        if self.env_cfg.fixed_spawn_yaw_deg is not None:
            return float(np.radians(self.env_cfg.fixed_spawn_yaw_deg))
        return float(np.random.uniform(-np.pi, np.pi))

    def _teleport_robot(self, pos: np.ndarray, yaw: float | None = None):
        """ロボットを指定位置・向きへ瞬間移動させ、速度をリセットする"""
        if yaw is None:
            yaw = self._sample_spawn_yaw()
        half = yaw / 2.0
        quat = np.array([[np.cos(half), 0.0, 0.0, np.sin(half)]], dtype=np.float32)
        self._robot.set_world_poses(
            positions=np.array([[pos[0], pos[1], pos[2]]], dtype=np.float32),
            orientations=quat,
        )
        self._robot.set_joint_velocities(
            np.zeros((1, self._robot.num_dof), dtype=np.float32)
        )
        self._robot.set_linear_velocities(np.zeros((1, 3), dtype=np.float32))
        self._robot.set_angular_velocities(np.zeros((1, 3), dtype=np.float32))

    def _get_robot_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """ロボットの(位置, クォータニオン(w,x,y,z))をまとめて取得する"""
        pos, quat = self._robot.get_world_poses()
        return _to_numpy(pos[0]), _to_numpy(quat[0])

    def _get_robot_pos(self) -> np.ndarray:
        return self._get_robot_pose()[0]

    def _get_robot_quat(self) -> np.ndarray:
        return self._get_robot_pose()[1]

    def close(self):
        self._world.stop()


class PointNavGymEnv(gym.Env):
    """gymnasium.Envラッパー, RGBとgoalを発行"""

    def __init__(self, env_cfg: EnvConfig):
        super().__init__()
        self.env_cfg = env_cfg
        W, H = self.env_cfg.camera_resolution

        self.observation_space = spaces.Dict(
            {
                "rgb": spaces.Box(0.0, 1.0, shape=(3, H, W), dtype=np.float32),
                "goal": spaces.Box(
                    low=np.array([0.0, -1.0], dtype=np.float32),
                    high=np.array([np.inf, 1.0], dtype=np.float32),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self._env: PointNavIsaacEnv | None = None

    def _lazy_init(self):
        if self._env is None:
            self._env = PointNavIsaacEnv(self.env_cfg)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """`seed`は無視される(NavMeshのランダムサンプリングはPythonから制御不能なため、
        完全な再現性は保証されない)。"""
        super().reset(seed=seed)
        self._lazy_init()
        obs = self._env.reset()
        info = {"dist": float(self._env.initial_goal_dist)}
        return obs, info

    def step(self, action: np.ndarray):
        return self._env.step(action)

    def render(self):
        pass

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
