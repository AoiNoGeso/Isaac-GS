"""IsaacSim World・CrowdController・IRA(omni.anim.behavior.core)を接続する人物マネージャ"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from envs.human_controller import ira_agent
from envs.human_controller.base import CrowdController
from envs.human_controller.health import HealthMonitor

HUMANS_ROOT = "/World/Humans"

GOAL_REACH_THRESHOLD = 0.6

# 歩行モデル名 -> CrowdController生成関数のレジストリSocialForceModel/AVOCADO等を追加する際はここに1行足すだけでenvs/config.pyのhuman_controllerから選べる
_CONTROLLER_REGISTRY: dict[str, Callable[[float, float, float], CrowdController]] = {}


def _make_orca(max_speed: float, radius: float, physics_dt: float) -> CrowdController:
    from envs.human_controller.controllers.orca.orca import ORCAConfig, ORCASimulator

    # 呼び出し側はORCAとIRAに共通の制御周期を渡す。
    return ORCASimulator(ORCAConfig(max_speed=max_speed, radius=radius, time_step=physics_dt))


_CONTROLLER_REGISTRY["orca"] = _make_orca


def _default_controller_factory(name: str) -> Callable[[float, float, float], CrowdController]:
    try:
        return _CONTROLLER_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"未知のhuman_controller: {name!r} (登録済み: {sorted(_CONTROLLER_REGISTRY)})"
        )


@dataclass
class _HumanState:
    agent_id: int
    goal_xy: tuple[float, float] = (0.0, 0.0)
    ira_path: str = ""  # スポーンしたSkelRootのパス
    ira_agent: object = None  # IBehaviorAgentハンドル
    task_id: object = None  # 現在実行中のmove_to/idleタスク


class HumanManager:
    """複数の人間アバターをCrowdController(既定ORCA) + IRA(omni.anim.behavior.core)で駆動するマネージャ

    使い方:
        mgr = HumanManager(env_cfg, world, inav)
        mgr.inject_humans(stage)                          # setup時
        mgr.reset_humans()                                # エピソード開始時
        mgr.pre_physics_step(physics_dt, robot_pos_xy, robot_vel_xy)  # world.step()の直前
        world.step(...)
        mgr.post_physics_step()                           # world.step()の直後

    ロボットはCrowdControllerへagent(人物と対等な回避責任1:1のreciprocal avoidance対象)として登録

    ORCAが衝突回避の「脳」を担い、IRAは実位置・実速度の供給元兼move_to/set_speedによる
    「アニメーション実行役」に徹する(IRA自前の回避は無効化済み、tests/reIRA/README.md参照)。
    """

    def __init__(self, env_cfg, world, inav, controller_factory: Optional[Callable] = None):
        self.env_cfg = env_cfg
        self._world = world
        self._inav = inav
        self._controller_factory = controller_factory or _default_controller_factory(
            env_cfg.human_controller
        )
        self._controller: Optional[CrowdController] = None
        self._states: list[_HumanState] = []
        self._health: Optional[HealthMonitor] = None
        self._max_speed = env_cfg.human_speed_range[-1]  # 歩行の目標速度 [m/s]
        self._radius = env_cfg.human_radius
        self._robot_agent_id: Optional[int] = None  # 初回pre_physics_step()で遅延登録する
        self._anim_stride = max(1, env_cfg.human_anim_stride)
        self._substep_counter = 0
        self._last_control_pos: list[tuple[float, float]] = []  # 測定速度算出用(前回制御tick時点の位置)
        self._cached_world_positions_xy: list[tuple[float, float]] = []

    # ------------------------------------------------------------------
    # NavMeshサンプリング
    # ------------------------------------------------------------------
    def _sample_navmesh_point_xyz(self) -> Optional[tuple[float, float, float]]:
        """NavMesh上のランダム点を高さ込みで返す"""
        nm = self._inav.get_navmesh()
        if nm is None:
            return None
        for _ in range(20):
            p = nm.query_random_point()
            if p is None:
                continue
            pos = np.array([p[0], p[1], p[2]], dtype=np.float32)
            if np.all(np.isfinite(pos)):
                return float(pos[0]), float(pos[1]), float(pos[2])
        return None

    def _sample_goal_min_dist_xy(
        self, from_xy: tuple[float, float], attempts: int = 10
    ) -> Optional[tuple[float, float]]:
        """`from_xy`から`human_min_goal_dist`以上離れたNavMesh上の点を探す
        (既定回数試行し、満たせなければ一番遠かった点で妥協する)"""
        min_dist = self.env_cfg.human_min_goal_dist
        best_goal, best_dist = from_xy, 0.0
        for _ in range(attempts):
            point = self._sample_navmesh_point_xyz()
            if point is None:
                continue
            candidate = (point[0], point[1])
            dist = float(np.hypot(candidate[0] - from_xy[0], candidate[1] - from_xy[1]))
            if dist > best_dist:
                best_goal, best_dist = candidate, dist
            if dist >= min_dist:
                break
        return best_goal

    def _sample_spawn_and_goal_xyz(
        self, attempts: int = 10
    ) -> tuple[Optional[tuple[float, float, float]], Optional[tuple[float, float]]]:
        """スポーン(高さ込み)と、そこから`human_min_goal_dist`以上離れたゴールをNavMesh上から
        サンプリングする"""
        spawn = self._sample_navmesh_point_xyz()
        if spawn is None:
            return None, None
        goal = self._sample_goal_min_dist_xy((spawn[0], spawn[1]), attempts=attempts)
        return spawn, goal

    def _sample_spawn_min_dist_from_xyz(
        self, ref_xy: tuple[float, float], min_dist: float, attempts: int = 10
    ) -> Optional[tuple[float, float, float]]:
        """`ref_xy`(ロボットのスポーン位置)から`min_dist`以上離れたNavMesh上の点を
        (高さ込みで)探す(既定回数試行し、満たせなければ一番遠かった点で妥協する)。
        1step目でロボットと人物が至近距離でスポーンし即座にhuman_collisionになる事故を防ぐ"""
        best_point, best_dist = None, -1.0
        for _ in range(attempts):
            point = self._sample_navmesh_point_xyz()
            if point is None:
                continue
            dist = float(np.hypot(point[0] - ref_xy[0], point[1] - ref_xy[1]))
            if dist > best_dist:
                best_point, best_dist = point, dist
            if dist >= min_dist:
                break
        return best_point

    # ------------------------------------------------------------------
    # セットアップ
    # ------------------------------------------------------------------
    def inject_humans(self, stage) -> None:
        """人物アバター(IRAキャラクター)をstageに注入する(num_humans<=0の場合は何もしない)"""
        num_humans = self.env_cfg.num_humans
        if num_humans <= 0:
            return

        import omni.timeline
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        for _ in range(3):
            self._world.app.update()

        max_speed, radius = self._max_speed, self._radius + self.env_cfg.human_avoidance_margin
        self._controller = self._controller_factory(max_speed, radius, self.env_cfg.physics_dt * self._anim_stride)
        self._register_navmesh_boundary_obstacles()

        spawns_xyz = []
        goals = []
        for _ in range(num_humans):
            spawn, goal = self._sample_spawn_and_goal_xyz()
            if spawn is None:
                raise RuntimeError('IRA spawn requires a valid baked NavMesh')
            spawns_xyz.append(spawn)
            goals.append(goal or (spawns_xyz[-1][0], spawns_xyz[-1][1]))
        spawns = [(x, y) for x, y, _ in spawns_xyz]

        character_url, motion_library_url = ira_agent.resolve_character_and_library(
            self.env_cfg.human_avatar_character, self.env_cfg.human_motion_library
        )
        import omni.usd

        ctx = omni.usd.get_context()
        ira_paths = ira_agent.spawn_humans(
            stage, ctx, HUMANS_ROOT, spawns_xyz, character_url, motion_library_url,
            app_update_fn=self._world.app.update,
        )
        timeline.play()
        for _ in range(60):
            self._world.step(render=True)
        ira_agents = ira_agent.bind_agents(ira_paths, app_update_fn=self._world.app.update)

        self._states = []
        for i in range(num_humans):
            agent_id = self._controller.add_agent(spawns[i], radius=radius, max_speed=max_speed)
            self._states.append(
                _HumanState(
                    agent_id=agent_id,
                    goal_xy=goals[i],
                    ira_path=ira_paths[i],
                    ira_agent=ira_agents[i],
                )
            )
        self._health = HealthMonitor(num_humans, window=max(2, round(5 / self.env_cfg.physics_dt)))
        self._last_control_pos = [tuple(a.get_world_translation())[:2] for a in ira_agents]
        self._cached_world_positions_xy = list(self._last_control_pos)
        self._commands = np.zeros((num_humans, 2))

    def _register_navmesh_boundary_obstacles(self) -> None:
        """NavMesh境界線をORCAの静的障害物として登録する(inject_humans時に1回のみ)
        `get_draw_lines`が返すバラバラな2点セグメントは`_stitch_segments_to_chains`で
        連続したポリラインに復元してから登録する(継ぎ目からのすり抜け対策)"""
        nm = self._inav.get_navmesh()
        if nm is None:
            return
        add_obstacle = getattr(self._controller, "add_static_obstacle", None)
        process_obstacles = getattr(self._controller, "process_obstacles", None)
        if add_obstacle is None or process_obstacles is None:
            return  # このCrowdController実装は静的障害物に対応していない
        lines = nm.get_draw_lines(True)
        segments = [
            ((lines[i][0], lines[i][1]), (lines[i + 1][0], lines[i + 1][1]))
            for i in range(0, len(lines) - 1, 2)
        ]
        for chain in _stitch_segments_to_chains(segments):
            # RVO2は時計回りの頂点列を要求するget_draw_linesは反時計回りで返すため反転する
            # (tests/human_phys/orca_navmesh_obstacle_narrow_corridor_test.pyで実測確認済み)
            add_obstacle(list(reversed(chain)))
        process_obstacles()

    # ------------------------------------------------------------------
    # reset / step
    # ------------------------------------------------------------------
    def reset_humans(self, robot_pos_xy: tuple[float, float] | None = None) -> None:
        """各人物の位置・ゴールをNavMesh上の新しいランダム点へ再サンプリングし、
        IRAエージェントをreset()で再配置する(prim自体は作り直さない)。
        `robot_pos_xy`を渡すと、そこから`human_robot_min_spawn_dist`以上離れた位置に
        スポーンさせる(1step目での至近距離スポーンによるhuman_collisionを防ぐ)"""
        if self._controller is None:
            return

        self._substep_counter = 0
        for i, state in enumerate(self._states):
            if robot_pos_xy is not None:
                spawn = self._sample_spawn_min_dist_from_xyz(
                    robot_pos_xy, self.env_cfg.human_robot_min_spawn_dist
                )
                goal = self._sample_goal_min_dist_xy((spawn[0], spawn[1])) if spawn is not None else None
            else:
                spawn, goal = self._sample_spawn_and_goal_xyz()
            if spawn is not None:
                sx, sy, sz = spawn
                if state.task_id is not None:
                    state.ira_agent.cancel_task(state.task_id)
                    state.task_id = None
                import carb
                if not state.ira_agent.reset(carb.Float3(sx, sy, sz)):
                    raise RuntimeError(f'IRA reset failed: {state.ira_path}')
                self._controller.set_position(state.agent_id, (sx, sy))
                self._controller.set_velocity(state.agent_id, (0.0, 0.0))
                self._last_control_pos[i] = (sx, sy)
                self._cached_world_positions_xy[i] = (sx, sy)
            if goal is not None:
                state.goal_xy = goal
        self._health = HealthMonitor(len(self._states), window=max(2, round(5 / self.env_cfg.physics_dt)))
        self._commands = np.zeros((len(self._states), 2))

    def pre_physics_step(
        self, dt: float, robot_pos_xy: tuple[float, float], robot_vel_xy: tuple[float, float]
    ) -> None:
        """実位置・実速度を同期 → ORCA計算 → IRA指令。すべて同じ制御周期で実行。"""
        if self._controller is None:
            return
        control_tick = self._substep_counter % self._anim_stride == 0
        self._substep_counter += 1
        if not control_tick:
            return
        control_dt = dt * self._anim_stride
        if self._robot_agent_id is None:
            self._robot_agent_id = self._controller.add_agent(
                robot_pos_xy, radius=self.env_cfg.robot.footprint_radius,
                max_speed=self.env_cfg.robot.v_linear_max)
        self._controller.set_position(self._robot_agent_id, robot_pos_xy)
        self._controller.set_velocity(self._robot_agent_id, robot_vel_xy)
        self._controller.set_preferred_velocity(self._robot_agent_id, robot_vel_xy)
        positions = [tuple(s.ira_agent.get_world_translation()) for s in self._states]
        if not np.isfinite(positions).all():
            from envs.human_controller.health import HumanLocomotionAnomaly
            raise HumanLocomotionAnomaly('Nonfinite IRA position before ORCA')
        for i, (state, xyz) in enumerate(zip(self._states, positions)):
            pos = np.asarray(xyz[:2])
            vel = (pos - self._last_control_pos[i]) / control_dt
            self._last_control_pos[i] = tuple(pos)
            self._controller.set_position(state.agent_id, tuple(pos))
            self._controller.set_velocity(state.agent_id, tuple(vel))
            delta = np.asarray(state.goal_xy) - pos
            distance = float(np.linalg.norm(delta))
            if distance < GOAL_REACH_THRESHOLD:
                state.goal_xy = self._sample_goal_min_dist_xy(tuple(pos))
                delta = np.asarray(state.goal_xy) - pos
                distance = float(np.linalg.norm(delta))
            pref = delta / max(distance, 1e-9) * self._max_speed
            self._controller.set_preferred_velocity(state.agent_id, tuple(pref))
        self._controller.step()
        for i, (state, xyz) in enumerate(zip(self._states, positions)):
            velocity = self._controller.get_velocity(state.agent_id)
            self._commands[i] = velocity
            ira_agent.issue_command(state.ira_agent, state, velocity, xyz,
                (*state.goal_xy, xyz[2]), self.env_cfg.human_ira_lookahead_s,
                self.env_cfg.human_ira_reissue_policy,
                min_speed=self.env_cfg.human_ira_min_command_speed,
                navmesh=self._inav.get_navmesh())

    def post_physics_step(self) -> None:
        """毎物理stepの実位置で衝突判定と健全性監視を更新する。"""
        if not self._states:
            return
        positions = [tuple(s.ira_agent.get_world_translation()) for s in self._states]
        rotations = [tuple(s.ira_agent.get_world_rotation()) for s in self._states]
        self._health.check(np.asarray(positions), np.asarray(rotations), self._commands)
        self._cached_world_positions_xy = [p[:2] for p in positions]

    def get_world_positions_xy(self) -> list[tuple[float, float]]:
        """ロボット側の距離ベース接触判定(_check_human_contact)用。
        直近のpost_physics_step時点でのIRA実位置を返す。"""
        return list(self._cached_world_positions_xy)


def _stitch_segments_to_chains(
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
    tol: float = 1e-3,
) -> list[list[tuple[float, float]]]:
    """端点同士をつなぎ合わせ、連続したポリライン/ポリゴンの頂点列に復元する"""

    def key(p: tuple[float, float]) -> tuple[int, int]:
        return (round(p[0] / tol), round(p[1] / tol))

    # 端点 -> そこに繋がる未使用セグメントindexの一覧
    endpoint_to_segments: dict[tuple[int, int], list[int]] = {}
    for i, (p0, p1) in enumerate(segments):
        endpoint_to_segments.setdefault(key(p0), []).append(i)
        endpoint_to_segments.setdefault(key(p1), []).append(i)

    used = [False] * len(segments)
    chains: list[list[tuple[float, float]]] = []

    def pop_neighbor(pt: tuple[float, float], exclude_idx: int) -> Optional[int]:
        for idx in endpoint_to_segments.get(key(pt), []):
            if not used[idx] and idx != exclude_idx:
                return idx
        return None

    for start in range(len(segments)):
        if used[start]:
            continue
        used[start] = True
        p0, p1 = segments[start]
        chain = [p0, p1]

        # 末尾(p1)方向へ伸ばせるだけ伸ばす
        tail = p1
        while True:
            nxt = pop_neighbor(tail, -1)
            if nxt is None:
                break
            used[nxt] = True
            a, b = segments[nxt]
            tail = b if key(a) == key(tail) else a
            if key(tail) == key(chain[0]):
                break
            chain.append(tail)

        chains.append(chain)

    return chains
