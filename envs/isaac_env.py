from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.config import EnvConfig, RobotConfig
from envs.geometry import goal_vec, quat_to_yaw

# IRA (isaacsim.replicator.agent) 人物キャラ
MOTION_LIBRARY_PRIM_PATH = "/World/HumanMotionLibrary"
HUMANS_ROOT = "/World/Humans"
CHAR_ASSET_DIR = "Isaac/People/Characters/"

# 落下判定
FALL_Z_THRESHOLD = -50.0


class PointNavIsaacEnv:
    def __init__(self, env_cfg: EnvConfig):
        self.env_cfg = env_cfg
        self.robot_cfg: RobotConfig = env_cfg.robot
        self._step_count = 0
        self._goal_pos = np.zeros(3, dtype=np.float32)
        self._prev_dist = 0.0
        self._ira_characters: list = []
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
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

        from envs.sensors.camera_sensor import RGBDCamera

        robot = self.robot_cfg
        kit_app = omni.kit.app.get_app()

        # IRA拡張はphysics/World/robot構築より前に有効化する必要がある
        if self.env_cfg.num_humans > 0:
            import asyncio

            for ext in (
                "isaacsim.replicator.agent.core",
                "omni.anim.behavior.core",
                "omni.anim.navigation.core",
                "omni.anim.retarget.core",
            ):
                enable_extension(ext)
            for _ in range(5):
                kit_app.update()

            # new_stage_asyncで新規ステージ作成後に参照・World構築を行う
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

        # NavMesh bake
        settings = None
        bake_keys = []
        bake_vals = []
        if self.env_cfg.navmesh_agent_radius_cm > 0:
            bake_keys += [
                "/exts/omni.anim.navigation.core/navMesh/config/agentMinRadius",
                "/exts/omni.anim.navigation.core/navMesh/config/agentMaxRadius",
            ]
            bake_vals += [float(self.env_cfg.navmesh_agent_radius_cm)] * 2
        if self.env_cfg.navmesh_agent_height_cm > 0:
            bake_keys.append(
                "/exts/omni.anim.navigation.core/navMesh/config/agentMinHeight"
            )
            bake_vals.append(float(self.env_cfg.navmesh_agent_height_cm))

        bake_orig = [None] * len(bake_keys)
        if bake_keys:
            import carb

            settings = carb.settings.get_settings()
            bake_orig = [settings.get(k) for k in bake_keys]
            for k, v in zip(bake_keys, bake_vals):
                settings.set(k, v)

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
            for k, v in zip(bake_keys, bake_orig):
                if v is not None:
                    settings.set(k, v)

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

        self._camera = RGBDCamera(
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

        if self.env_cfg.num_humans > 0:
            self._ira_characters = self._inject_ira_humans(stage)
        else:
            self._ira_characters = []

    def _setup_camera_viewport(self):
        try:
            import omni.kit.viewport.utility as vp_util

            vp_win = vp_util.create_viewport_window("Robot Camera", width=320, height=240)
            vp_win.viewport_api.set_active_camera(self.robot_cfg.camera_prim_path)
        except Exception as e:
            import carb

            carb.log_warn(f"[PointNavIsaacEnv] Camera viewport window skipped: {e}")

    # ------------------------------------------------------------------
    # IRA 人物キャラ注入
    # ------------------------------------------------------------------
    def _resolve_character_urls(self, num: int, seed: int = 0) -> list[str]:
        """Isaac/People/Characters/ から <name>/<name>.usd を num 体分(seedでシャッフル)返す"""
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
        """NavMesh上のランダム点を返す(人物の足元z, オフセット任意)"""
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
        import NavSchema
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

        # MotionLibraryはpayload arcで1回だけロードする(reference arcはretargetを壊す)
        motion_library_url = resolve_asset_path(
            "Isaac/People/MotionLibrary/HumanMotionLibrary.usd"
        )
        omni.kit.commands.execute(
            "CreatePayload",
            usd_context=ctx,
            path_to=MOTION_LIBRARY_PRIM_PATH,
            asset_path=motion_library_url,
        )
        # BehaviorMotionLibrary型に合成されるまで待つ
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

        # 親スコープを先に作成(未作成のネストパスへCreatePayloadすると合成されない)
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

            # payloadの合成待ち: SkelRootが現れるまでポーリング
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
                carb.log_warn(f"[PointNavIsaacEnv] Human_{i} SkelRoot 未検出, スキップ")
                continue

            char_prim = stage.GetPrimAtPath(char_prim_path)
            pt = self._sample_navmesh_point_for_human()
            if pt is not None:
                set_prim_pos(char_prim, Gf.Vec3f(float(pt[0]), float(pt[1]), float(pt[2])))

            # BehaviorAgentAPI(payload arc + このコマンドでretargetが正常動作)
            omni.kit.commands.execute(
                "ApplyBehaviorAgentAPICommand",
                skelroot_prim_paths=[skelroot.GetPath()],
                motion_library_prim_path=MOTION_LIBRARY_PRIM_PATH,
                motion_library_skeleton_rig="Human",
            )

            # キャラクター自身の体がnavmeshの障害物として扱われるのを防ぐ
            if not skelroot.HasAPI(NavSchema.NavMeshExcludeAPI):
                omni.kit.commands.execute(
                    "ApplyNavMeshAPICommand", prim_path=skelroot.GetPath(), api=NavSchema.NavMeshExcludeAPI
                )

            skelroot.ApplyAPI(IRA_CHARACTER_API)

            skelroot.GetAttribute(METRO_AGENT_NAME).Set(f"human_{i}")
            skelroot.GetAttribute(METRO_AGENT_GROUP).Set("humans")
            skelroot.GetAttribute(METRO_AGENT_SEED).Set(int(seed) + i)

            # Wander behavior(idle属性は付けない, 付けるとランタイムが壊れる)
            wander = stage.DefinePrim(f"{skelroot.GetPath()}/behavior_01", WANDER_PRIM_TYPE)
            wander.GetAttribute(WANDER_WALK_SPEED_RANGE).Set(Gf.Vec2f(*speed_range))
            wander.GetAttribute(WANDER_WALK_DISTANCE_RANGE).Set(Gf.Vec2f(*distance_range))
            skelroot.GetRelationship(METRO_AGENT_ROUTINES).SetTargets([wander.GetPath()])
            await app.next_update_async()

            skelroot_paths.append(str(skelroot.GetPath()))

        return skelroot_paths

    def _inject_ira_humans(self, stage, collect_timeout: int = 600) -> list:
        """人物キャラをstageに注入し, timeline.play()後にIRA_Characterのリストを返す"""
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

        # キャラ設定はtimeline停止状態で行う(再生状態だとretarget/behaviorランタイムが競合する)
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

        # timeline再生 → AgentsManager収集, physicsとtimelineを同時進行させる
        pump = lambda: self._world.step(render=True)
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

    def _relocate_human(self, character, pt: np.ndarray) -> None:
        """人物をptへ再配置する"""
        import carb

        bh = character.get_bh_agent()
        if bh is None:
            return
        bh.reset(target=carb.Float3(float(pt[0]), float(pt[1]), float(pt[2])), facing=None)

    def _reset_humans(self):
        """各人物をNavMesh上の新しいランダム点へ再配置する"""
        for character in self._ira_characters:
            pt = self._sample_navmesh_point_for_human()
            if pt is not None:
                self._relocate_human(character, pt)
        self._world.step(render=False)

    def _recover_stray_humans(self, threshold: float = 0.5) -> None:
        """navmeshから外れた人物を検知し, 新しい navmesh 上の点へ復帰させる"""
        import carb

        nm = self._inav.get_navmesh()
        if nm is None:
            return
        for character in self._ira_characters:
            hp = character.get_world_position()
            if hp is None:
                continue
            result = nm.query_closest_point(carb.Float3(float(hp.x), float(hp.y), float(hp.z)))
            closest = result[0] if isinstance(result, tuple) else result
            if closest is None:
                continue
            try:
                cx, cy = float(closest.x), float(closest.y)
            except AttributeError:
                cx, cy = float(closest[0]), float(closest[1])
            dist_sq = (cx - float(hp.x)) ** 2 + (cy - float(hp.y)) ** 2
            if dist_sq > threshold**2:
                pt = self._sample_navmesh_point_for_human()
                if pt is not None:
                    self._relocate_human(character, pt)

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

        return self._get_obs()

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        robot = self.robot_cfg
        v_x = float(np.clip(action[0], -1.0, 1.0)) * robot.v_linear_max
        omega = float(np.clip(action[1], -1.0, 1.0)) * robot.v_angular_max
        self._last_omega = omega
        v_L = v_x - omega * robot.wheel_base / 2.0
        v_R = v_x + omega * robot.wheel_base / 2.0
        # ホイール接地点の線形速度 [m/s] → joint角速度 [rad/s]
        w_L = v_L / robot.wheel_radius
        w_R = v_R / robot.wheel_radius

        vel_target = np.zeros(self._robot.num_dof, dtype=np.float32)
        for idx in self._left_wheel_idx:
            vel_target[idx] = w_L
        for idx in self._right_wheel_idx:
            vel_target[idx] = w_R
        self._robot.set_joint_velocity_targets(velocities=vel_target[np.newaxis, :])

        result = None
        human_hit = False
        for i in range(self.env_cfg.decimation):
            self._world.step(render=(i == self.env_cfg.decimation - 1))
            human_hit = human_hit or self._check_human_contact()

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

        hc = human_hit or self._check_human_contact()
        info["human_collision"] = hc
        if hc and not terminated:
            reward = float(self.env_cfg.r_human_collision)
            terminated = True
            truncated = False
            info["collision"] = True

        if self._ira_characters:
            self._recover_stray_humans()

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> dict:
        rgb, depth = self._camera.get_rgbd()
        return {
            "rgb": (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1),
            "depth": depth[np.newaxis, :, :].astype(np.float32),
            "goal": self._compute_goal_vec(),
        }

    def _compute_goal_vec(self) -> np.ndarray:
        pos = self._get_robot_pos()
        w, qx, qy, qz = self._get_robot_quat()
        yaw = quat_to_yaw(w, qx, qy, qz)
        return goal_vec(pos[0], pos[1], yaw, self._goal_pos[0], self._goal_pos[1])

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
        return float(1.0 - 2.0 * (qx * qx + qy * qy)) < self.robot_cfg.rollover_threshold

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

    def _contact_bodies(self) -> list[tuple[str, str]]:
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
            return []

    def _check_wall_contact(self) -> bool:
        return any(
            "wall_mesh" in body0 or "wall_mesh" in body1
            for body0, body1 in self._contact_bodies()
        )

    def _check_human_contact(self) -> bool:
        if not self._ira_characters:
            return False
        return any(
            HUMANS_ROOT in body0 or HUMANS_ROOT in body1
            for body0, body1 in self._contact_bodies()
        )

    def _check_collision(self) -> bool:
        if self._step_count < self.env_cfg.collision_grace_steps:
            return False
        if self._check_wall_contact():
            return True
        pos = self._get_robot_pos()
        return float(pos[2]) < FALL_Z_THRESHOLD

    def _recover_physics(self):
        safe_pos = self._sample_navmesh_point()
        if not np.all(np.isfinite(safe_pos)):
            safe_pos = self._get_robot_pos()
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
                    pos[2] += self.robot_cfg.spawn_offset
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
        self._world.stop()


class PointNavGymEnv(gym.Env):
    """gymnasium.Envラッパー, RGB・Depth・goalを発行"""

    def __init__(self, env_cfg: EnvConfig | None = None):
        super().__init__()
        self.env_cfg = env_cfg or EnvConfig()
        W, H = self.env_cfg.camera_resolution

        self.observation_space = spaces.Dict(
            {
                "rgb": spaces.Box(0.0, 1.0, shape=(3, H, W), dtype=np.float32),
                "depth": spaces.Box(0.0, np.inf, shape=(1, H, W), dtype=np.float32),
                "goal": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self._env: PointNavIsaacEnv | None = None

    def _lazy_init(self):
        if self._env is None:
            self._env = PointNavIsaacEnv(self.env_cfg)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._lazy_init()
        obs = self._env.reset()
        info = {"dist": float(self._env._prev_dist)}
        return obs, info

    def step(self, action: np.ndarray):
        return self._env.step(action)

    def render(self):
        pass

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
