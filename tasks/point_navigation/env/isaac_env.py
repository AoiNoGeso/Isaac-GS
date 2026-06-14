"""
isaacsim.core ベースの Point Navigation 環境コア（IsaacSim 6.0 対応）

観測: RGB (3,84,84) + goal ベクトル (2,)
行動: [v_x_norm, ω_z_norm] → ユニサイクルモデルで車輪速度に変換
NavMesh: 毎起動時にランタイム bake（floor_mesh を一時可視化）
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# Carter V1 物理定数
_V_LINEAR_MAX  = 1.0   # m/s
_V_ANGULAR_MAX = 1.0   # rad/s
_WHEEL_BASE    = 0.57  # m


@dataclass
class PointNavEnvCfg:
    stage_path: str = "/home/kato/Programs/Isaac-GS/sample_data/stages/corridor1/stage.usda"
    robot_usd: str = (
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
        "/Assets/Isaac/6.0/Isaac/Robots/NVIDIA/Carter/carter_v1.usd"
    )
    robot_prim_path: str = "/World/Robot"
    camera_prim_path: str = "/World/Robot/chassis_link/front_cam"
    camera_resolution: tuple[int, int] = (84, 84)

    physics_dt: float = 1.0 / 60.0
    rendering_dt: float = 1.0 / 30.0
    decimation: int = 2

    goal_threshold: float = 0.4       # m
    collision_threshold: float = 5.0  # N
    min_goal_dist: float = 2.0        # m
    max_episode_steps: int = 1800

    w_dist: float = 0.5
    w_collision: float = -5.0
    w_success: float = 100.0


class PointNavIsaacEnv:
    def __init__(self, cfg: PointNavEnvCfg):
        self.cfg = cfg
        self._step_count = 0
        self._goal_pos = np.zeros(3, dtype=np.float32)
        self._prev_dist = 0.0
        self._wall_contact_force = 0.0

        self._setup()

    # ──────────────────────────────────────────────────
    # 初期化
    # ──────────────────────────────────────────────────

    def _setup(self):
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.prims import Articulation
        from isaacsim.core.utils.extensions import enable_extension
        import omni.usd
        import omni.kit.app
        from pxr import UsdGeom

        enable_extension("omni.anim.navigation.bundle")

        kit_app = omni.kit.app.get_app()
        for _ in range(10):
            kit_app.update()

        import omni.anim.navigation.core as nav
        self._inav = nav.acquire_interface()

        add_reference_to_stage(usd_path=self.cfg.stage_path, prim_path="/World/env")
        add_reference_to_stage(usd_path=self.cfg.robot_usd, prim_path=self.cfg.robot_prim_path)

        self._world = World(
            physics_dt=self.cfg.physics_dt,
            rendering_dt=self.cfg.rendering_dt,
            stage_units_in_meters=1.0,
        )
        self._world.reset()

        # floor_mesh を一時的に visible にして NavMesh をランタイム bake
        stage = omni.usd.get_context().get_stage()
        floor_prim = stage.GetPrimAtPath("/World/env/floor_mesh")
        if floor_prim.IsValid():
            UsdGeom.Imageable(floor_prim).MakeVisible()
            for _ in range(10):
                kit_app.update()

        self._inav.start_navmesh_baking()
        print("[NavMesh] Baking ...")
        baked = False
        for i in range(500):
            kit_app.update()
            if self._inav.get_navmesh() is not None:
                print(f"[NavMesh] Bake 完了 (frame {i + 1})")
                baked = True
                break
        if not baked:
            print("[NavMesh] Warning: Bake 未完了．スポーン位置の NavMesh サンプリングが機能しません．")

        if floor_prim.IsValid():
            UsdGeom.Imageable(floor_prim).MakeInvisible()

        self._robot = Articulation(prim_paths_expr=self.cfg.robot_prim_path)
        self._robot.initialize()

        dof_names = self._robot.dof_names
        self._left_wheel_idx  = list(dof_names).index("left_wheel")
        self._right_wheel_idx = list(dof_names).index("right_wheel")

        self._setup_contact_callback()

        from .camera_sensor import setup_rgbd_camera
        self._camera = setup_rgbd_camera(
            prim_path=self.cfg.camera_prim_path,
            resolution=self.cfg.camera_resolution,
        )

    def _setup_contact_callback(self):
        try:
            from omni.physx import get_physx_simulation_interface

            wall_path = "/World/env/wall_mesh"

            def on_contact_report(contact_headers, contact_data):
                total_force = 0.0
                for header in contact_headers:
                    a = str(header.actor0)
                    b = str(header.actor1)
                    if wall_path in a or wall_path in b:
                        start = header.contact_data_offset
                        end   = start + header.num_contact_data
                        for cd in contact_data[start:end]:
                            total_force += abs(cd.normal_force)
                self._wall_contact_force = total_force

            physx_sim = get_physx_simulation_interface()
            self._contact_sub = physx_sim.subscribe_contact_report_events(on_contact_report)
            print("[Collision] PhysX contact callback registered")
        except Exception as e:
            print(f"[Collision] Contact callback setup failed: {e}")
            self._contact_sub = None

    # ──────────────────────────────────────────────────
    # リセット
    # ──────────────────────────────────────────────────

    def reset(self) -> dict:
        self._step_count = 0
        self._wall_contact_force = 0.0
        self._world.reset()

        robot_pos = self._sample_navmesh_point()
        self._teleport_robot(robot_pos)

        for _ in range(20):
            goal = self._sample_navmesh_point()
            dist = float(np.linalg.norm(goal[[0, 2]] - robot_pos[[0, 2]]))
            if dist >= self.cfg.min_goal_dist:
                break
        self._goal_pos = goal
        self._prev_dist = dist

        self._world.step(render=True)
        return self._get_obs()

    # ──────────────────────────────────────────────────
    # ステップ
    # ──────────────────────────────────────────────────

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        # ユニサイクルモデル → 左右車輪速度
        v_x   = float(np.clip(action[0], -1.0, 1.0)) * _V_LINEAR_MAX
        omega  = float(np.clip(action[1], -1.0, 1.0)) * _V_ANGULAR_MAX
        v_L   = v_x - omega * _WHEEL_BASE / 2.0
        v_R   = v_x + omega * _WHEEL_BASE / 2.0

        vel_target = np.zeros(self._robot.num_dof, dtype=np.float32)
        vel_target[self._left_wheel_idx]  = v_L
        vel_target[self._right_wheel_idx] = v_R
        self._robot.set_joint_velocity_targets(velocities=vel_target[np.newaxis, :])

        for i in range(self.cfg.decimation):
            self._world.step(render=(i == self.cfg.decimation - 1))

        self._step_count += 1
        obs = self._get_obs()
        reward, info = self._compute_reward()
        terminated = info["success"] or info["collision"]
        truncated  = self._step_count >= self.cfg.max_episode_steps
        return obs, float(reward), terminated, truncated, info

    # ──────────────────────────────────────────────────
    # 観測
    # ──────────────────────────────────────────────────

    def _get_obs(self) -> dict:
        rgb, _ = self._camera.get_rgbd()
        rgb_t = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)
        goal_vec = self._compute_goal_vec()
        return {"rgb": rgb_t, "goal": goal_vec}

    def _compute_goal_vec(self) -> np.ndarray:
        """ゴールへの相対極座標ベクトル (d_norm, angle_norm) を返す．Y-up 座標系．"""
        pos  = self._get_robot_pos()
        quat = self._get_robot_quat()  # (w, x, y, z)

        dx = self._goal_pos[0] - pos[0]
        dz = self._goal_pos[2] - pos[2]
        dist = float(np.sqrt(dx ** 2 + dz ** 2))
        d_norm = float(np.clip(dist / 10.0, 0.0, 1.0))

        # Y 軸周り回転角（yaw）
        w, x, y, z = quat
        yaw = float(np.arctan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y ** 2 + z ** 2)))

        angle_to_goal = float(np.arctan2(dx, -dz))
        angle_rel = (angle_to_goal - yaw + np.pi) % (2.0 * np.pi) - np.pi
        angle_norm = float(angle_rel / np.pi)

        return np.array([d_norm, angle_norm], dtype=np.float32)

    # ──────────────────────────────────────────────────
    # 報酬
    # ──────────────────────────────────────────────────

    def _compute_reward(self) -> tuple[float, dict]:
        pos      = self._get_robot_pos()
        cur_dist = float(np.linalg.norm(self._goal_pos[[0, 2]] - pos[[0, 2]]))
        success  = cur_dist < self.cfg.goal_threshold
        collision = self._check_collision()
        reward = (
            self.cfg.w_dist      * (self._prev_dist - cur_dist)
            + self.cfg.w_collision * float(collision)
            + self.cfg.w_success   * float(success)
        )
        self._prev_dist = cur_dist
        self._wall_contact_force = 0.0
        return reward, {"success": success, "collision": collision, "dist": cur_dist}

    def _check_collision(self) -> bool:
        return self._wall_contact_force > self.cfg.collision_threshold

    # ──────────────────────────────────────────────────
    # ユーティリティ
    # ──────────────────────────────────────────────────

    def _sample_navmesh_point(self) -> np.ndarray:
        nm = self._inav.get_navmesh()
        if nm is not None:
            p = nm.query_random_point()
            return np.array([p[0], p[1], p[2]], dtype=np.float32)
        return np.zeros(3, dtype=np.float32)

    def _teleport_robot(self, pos: np.ndarray):
        from pxr import Gf, UsdGeom
        prim = self._world.stage.GetPrimAtPath(self.cfg.robot_prim_path)
        for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
            if "translate" in op.GetOpName():
                op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
                break
        self._robot.set_joint_velocities(
            velocities=np.zeros((1, self._robot.num_dof), dtype=np.float32)
        )

    def _get_robot_pos(self) -> np.ndarray:
        pos, _ = self._robot.get_world_poses()
        if torch.is_tensor(pos):
            return pos[0].cpu().numpy()
        return np.array(pos[0], dtype=np.float32)

    def _get_robot_quat(self) -> np.ndarray:
        """クォータニオン (w, x, y, z) を返す．"""
        _, quat = self._robot.get_world_poses()
        if torch.is_tensor(quat):
            return quat[0].cpu().numpy()
        return np.array(quat[0], dtype=np.float32)

    def close(self):
        self._world.stop()
