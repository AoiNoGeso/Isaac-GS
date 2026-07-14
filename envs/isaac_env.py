from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tasks.point_navigation.config import EnvConfig, ModelConfig

_V_LINEAR_MAX = 0.3  # m/s
_V_ANGULAR_MAX = 0.3  # rad/s
_WHEEL_BASE = 0.57  # m

_AVOIDANCE_EVENT_NAME = "omni.anim.behavior.core:behavior_agent_avoidance_triggered"

# IRA (isaacsim.replicator.agent) 人物キャラ
MOTION_LIBRARY_PRIM_PATH = "/World/HumanMotionLibrary"
HUMANS_ROOT = "/World/Humans"
CHAR_ASSET_DIR = "Isaac/People/Characters/"


class PointNavIsaacEnv:
    def __init__(self, env_cfg: EnvConfig, model_cfg: ModelConfig):
        self.env_cfg = env_cfg
        self.model_cfg = model_cfg
        self._step_count = 0
        self._goal_pos = np.zeros(3, dtype=np.float32)
        self._prev_dist = 0.0
        self._ira_characters: list = []
        self._avoidance_sub = None
        self._robot_avoidance_pending = False
        self._setup()

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
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

        from envs.sensors.camera_sensor import RGBDCamera

        kit_app = omni.kit.app.get_app()

        # IRA 拡張は physics/World/robot 構築より前に有効化する必要がある。
        if self.env_cfg.num_humans > 0:
            import asyncio

            for ext in (
                "isaacsim.replicator.agent.core",
                "omni.anim.behavior.core",
                "omni.anim.navigation.core",
            ):
                enable_extension(ext)
            for _ in range(5):
                kit_app.update()

            # 人物ありの場合、SimulationApp のデフォルトステージではなく
            # new_stage_async で新規ステージを作ってから参照・World 構築を行う。
            fut = asyncio.ensure_future(omni.usd.get_context().new_stage_async())
            while not fut.done():
                kit_app.update()
            for _ in range(3):
                kit_app.update()

        enable_extension("omni.anim.navigation.bundle")
        for _ in range(10):
            kit_app.update()

        import omni.anim.navigation.core as nav

        self._inav = nav.acquire_interface()

        add_reference_to_stage(usd_path=self.env_cfg.stage_path, prim_path="/World/env")
        add_reference_to_stage(
            usd_path=self.env_cfg.robot_usd, prim_path=self.env_cfg.robot_prim_path
        )

        self._world = World(
            physics_dt=self.env_cfg.physics_dt,
            rendering_dt=self.env_cfg.rendering_dt,
            stage_units_in_meters=1.0,
        )
        self._world.reset()

        stage = omni.usd.get_context().get_stage()
        physics_scene = UsdPhysics.Scene.Get(stage, "/physicsScene")
        if not physics_scene:
            physics_scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
        physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        physics_scene.CreateGravityMagnitudeAttr().Set(9.81)

        # ── NavMesh bake（floor + wall を統合して一度だけ焼く）─────────────
        settings = None
        radius_keys = [
            "/persistent/exts/omni.anim.navigation.core/navMesh/config/agentMinRadius",
            "/persistent/exts/omni.anim.navigation.core/navMesh/config/agentMaxRadius",
        ]
        radius_orig = [None, None]
        if self.env_cfg.navmesh_agent_radius_cm > 0:
            import carb

            settings = carb.settings.get_settings()
            radius_orig = [settings.get(k) for k in radius_keys]
            for k in radius_keys:
                settings.set(k, float(self.env_cfg.navmesh_agent_radius_cm))

        vis_prims = []
        for path in ("/World/env/floor_mesh", "/World/env/wall_mesh"):
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
            for k, v in zip(radius_keys, radius_orig):
                if v is not None:
                    settings.set(k, v)
        self._world.step(render=False)

        self._setup_ground_plane(stage)

        self._robot = Articulation(prim_paths_expr=self.env_cfg.robot_prim_path)
        self._robot.initialize()

        dof_names = self._robot.dof_names
        self._left_wheel_idx = list(dof_names).index("left_wheel")
        self._right_wheel_idx = list(dof_names).index("right_wheel")

        # chassis_link に PhysxContactReportAPI を付与し ContactSensor を設置
        # C++ IContactSensor 経由のため Newton エンジン下でも動作する
        chassis_prim_path = f"{self.env_cfg.robot_prim_path}/chassis_link"
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

        self._camera = RGBDCamera(
            camera_prim_path=self.env_cfg.camera_prim_path,
            resolution=self.model_cfg.camera_resolution,
        )

        if self.env_cfg.show_camera_viewport:
            self._setup_camera_viewport()

        if self.env_cfg.num_humans > 0:
            self._ira_characters = self._inject_ira_humans(stage)
            self._subscribe_avoidance_event()
        else:
            self._ira_characters = []

    def _setup_camera_viewport(self):
        try:
            import omni.kit.viewport.utility as vp_util

            vp_win = vp_util.create_viewport_window(
                "Robot Camera", width=320, height=240
            )
            vp_win.viewport_api.set_active_camera(self.env_cfg.camera_prim_path)
        except Exception as e:
            import carb

            carb.log_warn(f"[PointNavIsaacEnv] Camera viewport window skipped: {e}")

    def _setup_ground_plane(self, stage):
        from pxr import Gf, UsdGeom, UsdPhysics

        nm = self._inav.get_navmesh()
        floor_z = 0.0
        if nm is not None:
            zs = [p[2] for _ in range(30) if (p := nm.query_random_point()) is not None]
            if zs:
                floor_z = float(np.median(zs))
        self._floor_z = floor_z

        plane_geom = UsdGeom.Plane.Define(stage, "/World/GroundPlane")
        plane_geom.CreateAxisAttr("Z")
        UsdPhysics.CollisionAPI.Apply(plane_geom.GetPrim())
        UsdGeom.Xformable(plane_geom.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3d(0.0, 0.0, float(floor_z) - 1.0)
        )

    # ------------------------------------------------------------------
    # IRA 人物キャラ注入
    # ------------------------------------------------------------------
    def _resolve_character_urls(self, num: int, seed: int = 0) -> list[str]:
        """Isaac/People/Characters/ から <name>/<name>.usd を num 体分（seed でシャッフル）返す。"""
        import carb
        import omni.client
        from omni.metropolis.utils.isaac_sim_util import resolve_asset_path

        base = resolve_asset_path(CHAR_ASSET_DIR)
        result, entries = omni.client.list(base)
        if result != omni.client.Result.OK or not entries:
            carb.log_error(f"[PointNavIsaacEnv] キャラクター一覧の取得に失敗: {base}")
            return []

        urls: list[str] = []
        for e in entries:
            rel = e.relative_path.rstrip("/")
            candidate = f"{base.rstrip('/')}/{rel}/{rel}.usd"
            r, _ = omni.client.stat(candidate)
            if r == omni.client.Result.OK:
                urls.append(candidate)

        if not urls:
            carb.log_error(f"[PointNavIsaacEnv] 有効なキャラクターUSDが見つかりません: {base}")
            return []

        rng = np.random.default_rng(seed)
        rng.shuffle(urls)
        return [urls[i % len(urls)] for i in range(num)]

    def _sample_navmesh_point_for_human(self, z_offset: float = 0.0) -> np.ndarray | None:
        """NavMesh 上のランダム点を返す（人物の足元 z、オフセット任意）。"""
        nm = self._inav.get_navmesh()
        if nm is None:
            return None
        for _ in range(20):
            p = nm.query_random_point()
            if p is None:
                continue
            pos = np.array([p[0], p[1], p[2]], dtype=np.float32)
            if np.all(np.isfinite(pos)):
                pos[2] += z_offset
                return pos
        return None

    async def _ira_setup_async(self, stage):
        import carb
        import BehaviorSchema
        import omni.kit.app
        import omni.kit.commands
        import omni.usd
        from isaacsim.replicator.agent.schema import (
            IRA_CHARACTER_API,
            WANDER_PRIM_TYPE,
            WANDER_WALK_DISTANCE_RANGE,
            WANDER_WALK_SPEED_RANGE,
        )
        from omni.metropolis.pipeline.usd_util import set_prim_pos
        from omni.metropolis.schema import (
            METRO_AGENT_GROUP,
            METRO_AGENT_NAME,
            METRO_AGENT_ROUTINES,
            METRO_AGENT_SEED,
        )
        from omni.metropolis.utils.isaac_sim_util import resolve_asset_path
        from pxr import Gf, Usd, UsdSkel

        app = omni.kit.app.get_app()
        ctx = omni.usd.get_context()

        num_humans = self.env_cfg.num_humans
        speed_range = self.env_cfg.human_speed_range
        distance_range = self.env_cfg.human_distance_range
        seed = self.env_cfg.human_seed

        # MotionLibrary は payload arc で 1 回だけロードする（reference arc は retarget を壊す）。
        motion_library_url = resolve_asset_path(
            "Isaac/People/MotionLibrary/HumanMotionLibrary.usd"
        )
        omni.kit.commands.execute(
            "CreatePayload",
            usd_context=ctx,
            path_to=MOTION_LIBRARY_PRIM_PATH,
            asset_path=motion_library_url,
        )
        # BehaviorMotionLibrary 型に合成されるまで待つ
        for _ in range(300):
            await app.next_update_async()
            ml = stage.GetPrimAtPath(MOTION_LIBRARY_PRIM_PATH)
            if ml.IsValid() and ml.IsA(BehaviorSchema.BehaviorMotionLibrary):
                break
        else:
            carb.log_error("[PointNavIsaacEnv] MotionLibrary のロードに失敗")
            return []

        char_urls = self._resolve_character_urls(num_humans, seed=seed)
        if not char_urls:
            return []

        # 親スコープを先に作成（未作成のネストパスへ CreatePayload すると合成されない）
        if not stage.GetPrimAtPath(HUMANS_ROOT).IsValid():
            stage.DefinePrim(HUMANS_ROOT, "Xform")

        skelroot_paths: list[str] = []
        for i, char_url in enumerate(char_urls):
            char_prim_path = f"{HUMANS_ROOT}/Human_{i}"
            omni.kit.commands.execute(
                "CreatePayload",
                usd_context=ctx,
                path_to=char_prim_path,
                asset_path=char_url,
            )

            # payload の合成待ち: SkelRoot が現れるまでポーリング
            skelroot = None
            for _ in range(600):
                await app.next_update_async()
                char_prim = stage.GetPrimAtPath(char_prim_path)
                if char_prim.IsValid():
                    skelroot = next(
                        (p for p in Usd.PrimRange(char_prim) if p.IsA(UsdSkel.Root)),
                        None,
                    )
                    if skelroot is not None:
                        break
            if skelroot is None:
                carb.log_warn(f"[PointNavIsaacEnv] Human_{i} SkelRoot 未検出、スキップ")
                continue

            char_prim = stage.GetPrimAtPath(char_prim_path)
            pt = self._sample_navmesh_point_for_human()
            if pt is not None:
                set_prim_pos(char_prim, Gf.Vec3f(float(pt[0]), float(pt[1]), float(pt[2])))

            # BehaviorAgentAPI（payload arc + このコマンドで retarget が正常動作）
            omni.kit.commands.execute(
                "ApplyBehaviorAgentAPICommand",
                skelroot_prim_paths=[skelroot.GetPath()],
                motion_library_prim_path=MOTION_LIBRARY_PRIM_PATH,
                motion_library_skeleton_rig="Human",
            )
            await app.next_update_async()

            skelroot.ApplyAPI(IRA_CHARACTER_API)
            await app.next_update_async()

            skelroot.GetAttribute(METRO_AGENT_NAME).Set(f"human_{i}")
            skelroot.GetAttribute(METRO_AGENT_GROUP).Set("humans")
            skelroot.GetAttribute(METRO_AGENT_SEED).Set(int(seed) + i)

            # Wander behavior（idle 属性は付けない = ランタイムが壊れる）
            wander = stage.DefinePrim(f"{skelroot.GetPath()}/behavior_01", WANDER_PRIM_TYPE)
            wander.GetAttribute(WANDER_WALK_SPEED_RANGE).Set(Gf.Vec2f(*speed_range))
            wander.GetAttribute(WANDER_WALK_DISTANCE_RANGE).Set(Gf.Vec2f(*distance_range))
            skelroot.GetRelationship(METRO_AGENT_ROUTINES).SetTargets([wander.GetPath()])
            await app.next_update_async()

            skelroot_paths.append(str(skelroot.GetPath()))

        return skelroot_paths

    def _inject_ira_humans(self, stage, collect_timeout: int = 600) -> list:
        """人物キャラを stage に注入し、timeline.play() 後に IRA_Character のリストを返す。"""
        import asyncio

        import carb
        import omni.kit.app
        import omni.timeline

        if self.env_cfg.num_humans <= 0:
            return []

        from isaacsim.replicator.agent.core.character import IRA_Character
        from omni.metropolis.pipeline.agent import AgentsManager

        app = omni.kit.app.get_app()
        timeline = omni.timeline.get_timeline_interface()

        # キャラ設定は timeline 停止状態で行う（再生状態だと retarget/behavior ランタイムが競合する）。
        timeline.stop()
        for _ in range(3):
            app.update()

        fut = asyncio.ensure_future(self._ira_setup_async(stage))
        while not fut.done():
            app.update()
        if fut.exception():
            raise fut.exception()
        skelroot_paths = fut.result()
        if not skelroot_paths:
            carb.log_error("[PointNavIsaacEnv] 有効な人物キャラが1体もセットアップできませんでした")
            return []

        # timeline 再生 → AgentsManager 収集。physics と timeline を同時進行させる
        # （app.update() だけだと physics articulation と behavior ランタイムが競合する）。
        pump = lambda: self._world.step(render=True)  # noqa: E731
        timeline.play()
        for _ in range(10):
            pump()

        manager = AgentsManager.get_instance()
        waited = 0
        while not manager.is_runtime_agents_collect_done():
            pump()
            waited += 1
            if waited > collect_timeout:
                carb.log_warn("[PointNavIsaacEnv] AgentsManager 収集タイムアウト")
                break

        return manager.get_agents_by_type(IRA_Character)

    def _subscribe_avoidance_event(self):
        # kinematic な IRA キャラは ContactSensor に現れないため、
        # IRA 純正 AvoidanceHandler の発火イベントを衝突信号として使う。
        import carb.eventdispatcher

        def _on_event(event):
            payload = dict(event.payload) if hasattr(event, "payload") else {}
            other = str(payload.get("object_rigidbody", ""))
            if other.startswith(self.env_cfg.robot_prim_path):
                self._robot_avoidance_pending = True

        self._avoidance_sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
            event_name=_AVOIDANCE_EVENT_NAME,
            on_event=_on_event,
            observer_name="PointNavIsaacEnv._on_avoidance_triggered",
        )

    def _reset_humans(self):
        """各人物を NavMesh 上の新しいランダム点へ移動（runtime が上書きする場合あり）。"""
        import omni.usd
        from omni.metropolis.pipeline.usd_util import set_prim_pos
        from pxr import Gf

        stage = omni.usd.get_context().get_stage()
        for i in range(len(self._ira_characters)):
            prim = stage.GetPrimAtPath(f"{HUMANS_ROOT}/Human_{i}")
            if not prim.IsValid():
                continue
            pt = self._sample_navmesh_point_for_human()
            if pt is not None:
                set_prim_pos(prim, Gf.Vec3f(float(pt[0]), float(pt[1]), float(pt[2])))
        self._world.step(render=False)

    # ------------------------------------------------------------------
    # reset / step
    # ------------------------------------------------------------------
    def reset(self) -> dict:
        self._step_count = 0
        self._last_omega = 0.0

        if self.env_cfg.fixed_spawn_pos is not None:
            robot_pos = np.array(self.env_cfg.fixed_spawn_pos, dtype=np.float32)
        else:
            robot_pos = self._sample_navmesh_point()

        if self.env_cfg.fixed_goal_pos is not None:
            self._goal_pos = np.array(self.env_cfg.fixed_goal_pos, dtype=np.float32)
            self._prev_dist = float(
                np.linalg.norm(self._goal_pos[[0, 1]] - robot_pos[[0, 1]])
            )
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

        if self.env_cfg.fixed_spawn_yaw_deg is not None:
            spawn_yaw = float(np.radians(self.env_cfg.fixed_spawn_yaw_deg))
        else:
            spawn_yaw = float(np.random.uniform(-np.pi, np.pi))

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

        if self._ira_characters and self.env_cfg.reset_humans_each_episode:
            self._reset_humans()

        self._robot_avoidance_pending = False  # 前エピソードの残骸を持ち越さない
        return self._get_obs()

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        v_x = float(np.clip(action[0], -1.0, 1.0)) * _V_LINEAR_MAX
        omega = float(np.clip(action[1], -1.0, 1.0)) * _V_ANGULAR_MAX
        self._last_omega = omega
        v_L = v_x - omega * _WHEEL_BASE / 2.0
        v_R = v_x + omega * _WHEEL_BASE / 2.0

        vel_target = np.zeros(self._robot.num_dof, dtype=np.float32)
        vel_target[self._left_wheel_idx] = v_L
        vel_target[self._right_wheel_idx] = v_R
        self._robot.set_joint_velocity_targets(velocities=vel_target[np.newaxis, :])

        result = None
        for i in range(self.env_cfg.decimation):
            self._world.step(render=(i == self.env_cfg.decimation - 1))

            if self._check_velocity_explosion():
                self._recover_physics()
                obs = self._get_obs()
                pos = self._get_robot_pos()
                dist = float(np.linalg.norm(self._goal_pos[[0, 1]] - pos[[0, 1]]))
                result = (
                    obs,
                    float(self.env_cfg.r_collision),
                    True,
                    False,
                    {
                        "success": False,
                        "collision": True,
                        "timeout": False,
                        "dist": dist,
                        "dist_final": dist,
                        "robot_xz": pos[[0, 1]],
                    },
                )
                break

            if (
                self._step_count >= self.env_cfg.collision_grace_steps
                and self._check_wall_contact()
            ):
                self._robot.set_joint_velocity_targets(
                    velocities=np.zeros((1, self._robot.num_dof), dtype=np.float32)
                )
                obs = self._get_obs()
                pos = self._get_robot_pos()
                dist = float(np.linalg.norm(self._goal_pos[[0, 1]] - pos[[0, 1]]))
                self._prev_dist = dist
                result = (
                    obs,
                    float(self.env_cfg.r_collision),
                    True,
                    False,
                    {
                        "success": False,
                        "collision": True,
                        "timeout": False,
                        "dist": dist,
                        "dist_final": dist,
                        "robot_xz": pos[[0, 1]],
                    },
                )
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

        # 人物衝突検知（壁衝突・速度爆発の早期 return でも terminated=True の場合があるため、
        # 情報反映（info["human_collision"]）は常に行い、壁と同時衝突でも計上する）。
        hc = self._robot_avoidance_pending
        self._robot_avoidance_pending = False
        info["human_collision"] = hc
        if hc and not terminated:
            reward = float(self.env_cfg.r_human_collision)
            terminated = True
            truncated = False
            info["collision"] = True

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> dict:
        rgb, _ = self._camera.get_rgbd()
        return {
            "rgb": (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1),
            "goal": self._compute_goal_vec(),
        }

    def _compute_goal_vec(self) -> np.ndarray:
        pos = self._get_robot_pos()
        w, qx, qy, qz = self._get_robot_quat()
        dx = self._goal_pos[0] - pos[0]
        dy = self._goal_pos[1] - pos[1]
        dist = float(np.sqrt(dx**2 + dy**2))
        yaw = float(np.arctan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2)))
        angle_rel = (float(np.arctan2(dy, dx)) - yaw + np.pi) % (2.0 * np.pi) - np.pi
        return np.array([dist, float(angle_rel / np.pi)], dtype=np.float32)

    def _compute_reward(self) -> tuple[float, dict]:
        pos = self._get_robot_pos()
        cur_dist = float(np.linalg.norm(self._goal_pos[[0, 1]] - pos[[0, 1]]))
        success = cur_dist < self.env_cfg.goal_threshold
        collision = self._check_collision()
        rollover = self._check_rollover()
        angle_rel = float(self._compute_goal_vec()[1]) * np.pi
        omega = getattr(self, "_last_omega", 0.0)
        reward = (
            self.env_cfg.r_dist * (self._prev_dist - cur_dist)
            + self.env_cfg.r_heading * float(np.cos(angle_rel))
            + self.env_cfg.r_collision * float(collision)
            + self.env_cfg.r_rollover * float(rollover)
            + self.env_cfg.r_success * float(success)
            + self.env_cfg.r_spin * float(omega**2)
            + self.env_cfg.r_time
        )
        self._prev_dist = cur_dist
        return reward, {
            "success": success,
            "collision": collision or rollover,
            "timeout": False,
            "dist": cur_dist,
            "dist_final": cur_dist,
            "robot_xz": pos[[0, 1]],
        }

    def _check_rollover(self) -> bool:
        w, qx, qy, qz = self._get_robot_quat()
        return float(1.0 - 2.0 * (qx * qx + qy * qy)) < self.env_cfg.rollover_threshold

    def _check_velocity_explosion(self) -> bool:
        try:
            linvel = self._robot.get_linear_velocities()
            if linvel is not None and float(np.max(np.abs(linvel))) > 10.0:
                return True
            if not all(np.isfinite(self._get_robot_pos())):
                return True
        except Exception:
            pass
        return False

    def _check_wall_contact(self) -> bool:
        try:
            frame = self._contact_sensor.get_data()
            if not frame.get("in_contact", False):
                return False
            for contact in frame.get("contacts", []):
                if "wall_mesh" in str(contact.get("body0", "")) or "wall_mesh" in str(
                    contact.get("body1", "")
                ):
                    return True
        except Exception:
            pass
        return False

    def _check_collision(self) -> bool:
        if self._step_count < self.env_cfg.collision_grace_steps:
            return False
        if self._check_wall_contact():
            return True
        pos = self._get_robot_pos()
        if float(pos[2]) < self._floor_z - 0.5:
            return True
        px, py = float(pos[0]), float(pos[1])
        margin = 0.3
        if (
            px < self._NM_X_MIN - margin
            or px > self._NM_X_MAX + margin
            or py < self._NM_Y_MIN - margin
            or py > self._NM_Y_MAX + margin
        ):
            return True
        return self._is_out_of_navmesh()

    # corridor1 NavMeshVolume AABB (Z-up: X=幅, Y=廊下長手方向)
    _NM_X_MIN: float = -0.185 - 4.841 / 2
    _NM_X_MAX: float = -0.185 + 4.841 / 2
    _NM_Y_MIN: float = 5.250 - 23.052 / 2
    _NM_Y_MAX: float = 5.250 + 23.052 / 2

    def _is_out_of_navmesh(self) -> bool:
        import carb

        nm = self._inav.get_navmesh()
        if nm is None:
            return False
        pos = self._get_robot_pos()
        result = nm.query_closest_point(
            carb.Float3(float(pos[0]), float(pos[1]), float(pos[2]))
        )
        closest = result[0] if isinstance(result, tuple) else result
        if closest is None:
            return True
        try:
            cx, cy = float(closest.x), float(closest.y)
        except AttributeError:
            cx, cy = float(closest[0]), float(closest[1])
        return (cx - float(pos[0])) ** 2 + (
            cy - float(pos[1])
        ) ** 2 > self.env_cfg.navmesh_exit_threshold**2

    _CHASSIS_HALF_HEIGHT = 0.20  # m

    def _recover_physics(self):
        safe_pos = np.array(
            [-0.168, 5.302, self._floor_z + self._CHASSIS_HALF_HEIGHT], dtype=np.float32
        )
        for _ in range(20):
            self._teleport_robot(safe_pos)
            self._world.step(render=False)
            if not self._check_velocity_explosion():
                break

    def _sample_navmesh_point(self) -> np.ndarray:
        nm = self._inav.get_navmesh()
        if nm is not None:
            for _ in range(20):
                p = nm.query_random_point()
                pos = np.array([p[0], p[1], p[2]], dtype=np.float32)
                if np.all(np.isfinite(pos)):
                    pos[2] += self._CHASSIS_HALF_HEIGHT
                    return pos
        return np.zeros(3, dtype=np.float32)

    def _teleport_robot(self, pos: np.ndarray, yaw: float | None = None):
        if yaw is None:
            if self.env_cfg.fixed_spawn_yaw_deg is not None:
                yaw = float(np.radians(self.env_cfg.fixed_spawn_yaw_deg))
            else:
                yaw = float(np.random.uniform(-np.pi, np.pi))
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

    def _get_robot_pos(self) -> np.ndarray:
        pos, _ = self._robot.get_world_poses()
        return (
            pos[0].cpu().numpy()
            if hasattr(pos[0], "cpu")
            else np.array(pos[0], dtype=np.float32)
        )

    def _get_robot_quat(self) -> np.ndarray:
        _, quat = self._robot.get_world_poses()
        return (
            quat[0].cpu().numpy()
            if hasattr(quat[0], "cpu")
            else np.array(quat[0], dtype=np.float32)
        )

    def close(self):
        if self._avoidance_sub is not None:
            self._avoidance_sub.reset()
            self._avoidance_sub = None
        self._world.stop()


class PointNavGymEnv(gym.Env):
    """gymnasium.Env ラッパー。人物観測は含めない（RGB + goal のみ）。"""

    def __init__(
        self,
        env_cfg: EnvConfig | None = None,
        model_cfg: ModelConfig | None = None,
    ):
        super().__init__()
        self.env_cfg = env_cfg or EnvConfig()
        self.model_cfg = model_cfg or ModelConfig()
        W, H = self.model_cfg.camera_resolution

        obs: dict = {}
        if self.model_cfg.input_rgb:
            obs["rgb"] = spaces.Box(0.0, 1.0, shape=(3, H, W), dtype=np.float32)
        if self.model_cfg.input_goal:
            obs["goal"] = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        assert obs, "input_rgb と input_goal の少なくとも一方は True にしてください"

        self.observation_space = spaces.Dict(obs)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self._env: PointNavIsaacEnv | None = None

    def _lazy_init(self):
        if self._env is None:
            self._env = PointNavIsaacEnv(self.env_cfg, self.model_cfg)

    def _filter_obs(self, obs: dict) -> dict:
        return {k: obs[k] for k in self.observation_space.spaces}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._lazy_init()
        obs = self._env.reset()
        info = {"dist": float(self._env._prev_dist)}
        return self._filter_obs(obs), info

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self._env.step(action)
        return self._filter_obs(obs), reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
