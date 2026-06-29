import numpy as np

from tasks.point_navigation.config import PointNavEnvCfg

_V_LINEAR_MAX = 0.3  # m/s
_V_ANGULAR_MAX = 0.3  # rad/s
_WHEEL_BASE = 0.57  # m


class PointNavIsaacEnv:
    def __init__(self, cfg: PointNavEnvCfg):
        self.cfg = cfg
        self._step_count = 0
        self._goal_pos = np.zeros(3, dtype=np.float32)
        self._prev_dist = 0.0
        self._setup()

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

        enable_extension("omni.anim.navigation.bundle")

        kit_app = omni.kit.app.get_app()
        for _ in range(10):
            kit_app.update()

        import omni.anim.navigation.core as nav

        self._inav = nav.acquire_interface()

        add_reference_to_stage(usd_path=self.cfg.stage_path, prim_path="/World/env")
        add_reference_to_stage(
            usd_path=self.cfg.robot_usd, prim_path=self.cfg.robot_prim_path
        )

        self._world = World(
            physics_dt=self.cfg.physics_dt,
            rendering_dt=self.cfg.rendering_dt,
            stage_units_in_meters=1.0,
        )
        self._world.reset()

        stage = omni.usd.get_context().get_stage()
        physics_scene = UsdPhysics.Scene.Get(stage, "/physicsScene")
        if not physics_scene:
            physics_scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
        physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        physics_scene.CreateGravityMagnitudeAttr().Set(9.81)

        floor_prim = stage.GetPrimAtPath("/World/env/floor_mesh")
        if floor_prim.IsValid():
            UsdGeom.Imageable(floor_prim).MakeVisible()
            for _ in range(10):
                kit_app.update()
        else:
            print("[NavMesh] Warning: /World/env/floor_mesh not found")

        try:
            nm_settings = self._inav.get_navmesh_settings()
            nm_settings.agent_radius = 0.7
            nm_settings.agent_height = 1.2
            nm_settings.agent_max_climb = 0.1
            self._inav.set_navmesh_settings(nm_settings)
        except Exception:
            pass

        self._inav.start_navmesh_baking()
        print("[NavMesh] Baking ...")
        baked = False
        for i in range(500):
            self._world.step(render=True)
            if self._inav.get_navmesh() is not None:
                print(f"[NavMesh] Bake 完了 (frame {i + 1})")
                baked = True
                break
        if not baked:
            print("[NavMesh] Warning: Bake 未完了")

        if floor_prim.IsValid():
            UsdGeom.Imageable(floor_prim).MakeInvisible()

        self._setup_ground_plane(stage)

        self._robot = Articulation(prim_paths_expr=self.cfg.robot_prim_path)
        self._robot.initialize()

        dof_names = self._robot.dof_names
        self._left_wheel_idx = list(dof_names).index("left_wheel")
        self._right_wheel_idx = list(dof_names).index("right_wheel")

        # chassis_link に PhysxContactReportAPI を付与し ContactSensor を設置
        # C++ IContactSensor 経由のため Newton エンジン下でも動作する
        chassis_prim_path = f"{self.cfg.robot_prim_path}/chassis_link"
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
            camera_prim_path=self.cfg.camera_prim_path,
            resolution=self.cfg.camera_resolution,
        )

        if self.cfg.show_camera_viewport:
            self._setup_camera_viewport()

    def _setup_camera_viewport(self):
        try:
            import omni.kit.viewport.utility as vp_util

            vp_win = vp_util.create_viewport_window(
                "Robot Camera", width=320, height=240
            )
            vp_win.viewport_api.set_active_camera(self.cfg.camera_prim_path)
            print("[Camera] Robot viewport window created")
        except Exception as e:
            print(f"[Camera] Viewport window skipped: {e}")

    def _setup_ground_plane(self, stage):
        from pxr import Gf, UsdGeom, UsdPhysics

        nm = self._inav.get_navmesh()
        floor_z = 0.0
        if nm is not None:
            zs = [p[2] for _ in range(30) if (p := nm.query_random_point()) is not None]
            if zs:
                floor_z = float(np.median(zs))
        print(f"[Ground Plane] floor_z={floor_z:.3f}m")
        self._floor_z = floor_z

        plane_geom = UsdGeom.Plane.Define(stage, "/World/GroundPlane")
        plane_geom.CreateAxisAttr("Z")
        UsdPhysics.CollisionAPI.Apply(plane_geom.GetPrim())
        UsdGeom.Xformable(plane_geom.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3d(0.0, 0.0, float(floor_z) - 1.0)
        )

    def reset(self) -> dict:
        self._step_count = 0
        self._last_omega = 0.0

        if self.cfg.fixed_spawn_pos is not None:
            robot_pos = np.array(self.cfg.fixed_spawn_pos, dtype=np.float32)
        else:
            robot_pos = self._sample_navmesh_point()

        if self.cfg.fixed_goal_pos is not None:
            self._goal_pos = np.array(self.cfg.fixed_goal_pos, dtype=np.float32)
            self._prev_dist = float(
                np.linalg.norm(self._goal_pos[[0, 1]] - robot_pos[[0, 1]])
            )
            if self._prev_dist < self.cfg.min_goal_dist:
                print(
                    f"[Warning] spawn-goal XY 距離 {self._prev_dist:.2f}m が "
                    f"min_goal_dist ({self.cfg.min_goal_dist}m) を下回っています"
                )
        else:
            best_goal, best_dist = robot_pos.copy(), 0.0
            for _ in range(50):
                goal = self._sample_navmesh_point()
                dist = float(np.linalg.norm(goal[[0, 1]] - robot_pos[[0, 1]]))
                if dist > best_dist:
                    best_goal, best_dist = goal, dist
                if dist >= self.cfg.min_goal_dist:
                    break
            self._goal_pos = best_goal
            self._prev_dist = best_dist

        if self.cfg.fixed_spawn_yaw_deg is not None:
            spawn_yaw = float(np.radians(self.cfg.fixed_spawn_yaw_deg))
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

        for i in range(self.cfg.decimation):
            self._world.step(render=(i == self.cfg.decimation - 1))

            if self._check_velocity_explosion():
                self._recover_physics()
                obs = self._get_obs()
                pos = self._get_robot_pos()
                dist = float(np.linalg.norm(self._goal_pos[[0, 1]] - pos[[0, 1]]))
                return (
                    obs,
                    float(self.cfg.r_collision),
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

            if (
                self._step_count >= self.cfg.collision_grace_steps
                and self._check_wall_contact()
            ):
                self._robot.set_joint_velocity_targets(
                    velocities=np.zeros((1, self._robot.num_dof), dtype=np.float32)
                )
                obs = self._get_obs()
                pos = self._get_robot_pos()
                dist = float(np.linalg.norm(self._goal_pos[[0, 1]] - pos[[0, 1]]))
                self._prev_dist = dist
                return (
                    obs,
                    float(self.cfg.r_collision),
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

        self._step_count += 1
        obs = self._get_obs()
        reward, info = self._compute_reward()
        terminated = info["success"] or info["collision"]
        truncated = self._step_count >= self.cfg.max_episode_steps
        if truncated:
            reward += self.cfg.r_timeout
            info["timeout"] = True
        return obs, float(reward), terminated, truncated, info

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
        success = cur_dist < self.cfg.goal_threshold
        collision = self._check_collision()
        rollover = self._check_rollover()
        angle_rel = float(self._compute_goal_vec()[1]) * np.pi
        omega = getattr(self, "_last_omega", 0.0)
        reward = (
            self.cfg.r_dist * (self._prev_dist - cur_dist)
            + self.cfg.r_heading * float(np.cos(angle_rel))
            + self.cfg.r_collision * float(collision)
            + self.cfg.r_rollover * float(rollover)
            + self.cfg.r_success * float(success)
            + self.cfg.r_spin * float(omega**2)
            + self.cfg.r_time
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
        return float(1.0 - 2.0 * (qx * qx + qy * qy)) < self.cfg.rollover_threshold

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
        if self._step_count < self.cfg.collision_grace_steps:
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
        ) ** 2 > self.cfg.navmesh_exit_threshold**2

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
            if self.cfg.fixed_spawn_yaw_deg is not None:
                yaw = float(np.radians(self.cfg.fixed_spawn_yaw_deg))
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
