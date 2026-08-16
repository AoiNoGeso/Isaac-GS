"""IsaacSim World・CrowdController・ai4animationpyを接続する人物マネージャ"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from envs.human_controller.base import CrowdController
from envs.human_controller.locomotion import BONE_PARENTS, MODEL_GLB, LocomotionPool

HUMANS_ROOT = "/World/Humans"

GOAL_REACH_THRESHOLD = 0.3

# スタック検出用しきい値(障害物の隙間・角で身動きが取れなくなった場合の強制ゴール再選定)
STUCK_MOVE_THRESHOLD = 0.01  # [m/step]
STUCK_STEPS_LIMIT = 60  # 30fps換算で概ね2秒

# 高さ追従専用カプセル(PhysX Character Controller)の寸法・パラメータ
CAPSULE_CYLINDER_HEIGHT = 1.0  # 円柱部分の高さ[m](半球キャップ含む全高はこれ+2*radius)
CAPSULE_SLOPE_LIMIT_DEG = 45.0
CAPSULE_STEP_OFFSET = 0.3  # 乗り越えられる段差の高さ[m]

# 歩行モデル名 -> CrowdController生成関数のレジストリSocialForceModel/AVOCADO等を
# 追加する際はここに1行足すだけでenvs/config.pyのhuman_controllerから選べる
_CONTROLLER_REGISTRY: dict[str, Callable[[float, float], CrowdController]] = {}


def _make_orca(max_speed: float, radius: float) -> CrowdController:
    from envs.human_controller.controllers.orca.orca import ORCAConfig, ORCASimulator

    return ORCASimulator(ORCAConfig(max_speed=max_speed, radius=radius))


_CONTROLLER_REGISTRY["orca"] = _make_orca


def _default_controller_factory(name: str) -> Callable[[float, float], CrowdController]:
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
    height_z: float = 0.0
    avatar_prim: object = None
    translate_op: object = None
    translations_attr: object = None
    rotations_attr: object = None
    leaf_to_idx: dict = field(default_factory=dict)
    last_pos: tuple[float, float] | None = None  # スタック検出用
    stuck_steps: int = 0
    capsule_prim: object = None
    capsule_translate_op: object = None  # 物理解決後の高さ(Z)読み出し先
    cct_api: object = None
    capsule_path: str = ""  # get_physx_cct_interface()呼び出し用
    capsule_half_height: float = 0.0  # カプセル中心Z -> 床面Zの変換値


class HumanManager:
    """複数の人間アバターをCrowdController(既定ORCA) + ai4animationpyで駆動するマネージャ

    使い方:
        mgr = HumanManager(env_cfg, world, inav)
        mgr.inject_humans(stage)                # setup時
        mgr.reset_humans()                      # エピソード開始時
        mgr.pre_physics_step(physics_dt)        # world.step()の直前
        world.step(...)
        mgr.post_physics_step()                 # world.step()の直後
    """

    def __init__(self, env_cfg, world, inav, controller_factory: Optional[Callable] = None):
        self.env_cfg = env_cfg
        self._world = world
        self._inav = inav
        self._controller_factory = controller_factory or _default_controller_factory(
            env_cfg.human_controller
        )
        self._controller: Optional[CrowdController] = None
        self._pool: Optional[LocomotionPool] = None
        self._states: list[_HumanState] = []
        self._cct_iface = None  # _build_capsule_cctで初期化する高さ追従用CCTインターフェース
        self._max_speed = env_cfg.human_speed_range[-1]  # 歩行の目標速度 [m/s]
        self._radius = env_cfg.human_radius

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

    # ------------------------------------------------------------------
    # セットアップ
    # ------------------------------------------------------------------
    def inject_humans(self, stage) -> None:
        """人物アバターをstageに注入する(num_humans<=0の場合は何もしない)"""
        num_humans = self.env_cfg.num_humans
        if num_humans <= 0:
            return

        max_speed, radius = self._max_speed, self._radius
        self._controller = self._controller_factory(max_speed, radius)
        self._register_navmesh_boundary_obstacles()

        spawns_xyz = []
        goals = []
        for _ in range(num_humans):
            spawn, goal = self._sample_spawn_and_goal_xyz()
            spawns_xyz.append(spawn or (0.0, 0.0, 0.0))
            goals.append(goal or (spawns_xyz[-1][0], spawns_xyz[-1][1]))
        spawns = [(x, y) for x, y, _ in spawns_xyz]

        self._pool = LocomotionPool(num_humans, world_offsets_xy=spawns)

        self._states = []
        for i in range(num_humans):
            agent_id = self._controller.add_agent(spawns[i], radius=radius, max_speed=max_speed)
            avatar_prim, translate_op, t_attr, r_attr, leaf_to_idx = self._build_avatar_usd(
                stage, i, spawns_xyz[i]
            )
            capsule_prim, capsule_translate_op, cct_api, capsule_path = self._build_capsule_cct(
                stage, i, spawns_xyz[i], radius
            )
            self._states.append(
                _HumanState(
                    agent_id=agent_id,
                    goal_xy=goals[i],
                    height_z=spawns_xyz[i][2],
                    avatar_prim=avatar_prim,
                    translate_op=translate_op,
                    translations_attr=t_attr,
                    rotations_attr=r_attr,
                    leaf_to_idx=leaf_to_idx,
                    capsule_prim=capsule_prim,
                    capsule_translate_op=capsule_translate_op,
                    cct_api=cct_api,
                    capsule_path=capsule_path,
                    capsule_half_height=CAPSULE_CYLINDER_HEIGHT / 2.0 + radius,
                )
            )

    def _build_capsule_cct(self, stage, index: int, spawn_xyz: tuple[float, float, float], radius: float):
        """高さ(Z)追従専用の、見えないPhysX Character Controllerカプセルを用意する
        水平位置はai4animationpy側の結果にteleportで追従させるだけで物理制御はしない
        (壁越え防止はORCAの静的障害物が担う)カプセル中心Zは床面から半分の高さ分だけ上"""
        from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema
        from omni.physxcct.scripts.ifaces import get_physx_cct_interface

        path = f"{HUMANS_ROOT}/Human{index}_CCT"
        capsule = UsdGeom.Capsule.Define(stage, path)
        capsule.CreateRadiusAttr(radius)
        capsule.CreateHeightAttr(CAPSULE_CYLINDER_HEIGHT)
        capsule.CreateAxisAttr("Z")
        prim = capsule.GetPrim()
        UsdGeom.Imageable(prim).MakeInvisible()

        half_total_height = CAPSULE_CYLINDER_HEIGHT / 2.0 + radius
        translate_op = UsdGeom.Xformable(prim).AddTranslateOp()
        translate_op.Set(Gf.Vec3d(spawn_xyz[0], spawn_xyz[1], spawn_xyz[2] + half_total_height))

        UsdPhysics.CollisionAPI.Apply(prim)
        cct_api = PhysxSchema.PhysxCharacterControllerAPI.Apply(prim)
        cct_api.CreateSlopeLimitAttr(CAPSULE_SLOPE_LIMIT_DEG)
        cct_api.CreateStepOffsetAttr(CAPSULE_STEP_OFFSET)
        cct_api.CreateUpAxisAttr("Z")

        cct_iface = get_physx_cct_interface()
        cct_iface.disable_first_person(path)
        cct_iface.enable_gravity(path)
        self._cct_iface = cct_iface

        return prim, translate_op, cct_api, path

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

    def _build_avatar_usd(self, stage, index: int, spawn_xyz: tuple[float, float, float]):
        """1体分のUSD Skeletonを`HUMANS_ROOT`配下に用意する"""
        from pxr import Gf, UsdSkel, UsdGeom

        from stage_generation.coords import rotation_quat

        path = f"{HUMANS_ROOT}/Human{index}"
        avatar_xform = UsdGeom.Xform.Define(stage, path)
        avatar_prim = avatar_xform.GetPrim()
        # translateのX/Yはスポーン地点に1回だけ設定し以降更新しない(移動量はHipsジョイントの
        # ローカル姿勢が担う)Zはpost_physics_step()で毎ステップ更新する
        translate_op = UsdGeom.Xformable(avatar_prim).AddTranslateOp()
        translate_op.Set(Gf.Vec3d(spawn_xyz[0], spawn_xyz[1], spawn_xyz[2]))
        q_inv = rotation_quat("Y").GetInverse()
        UsdGeom.Xformable(avatar_prim).AddOrientOp().Set(Gf.Quatf(q_inv))

        ref_prim = stage.DefinePrim(f"{path}/Model")
        ref_prim.GetReferences().AddReference(_ensure_model_usd(), "/World")

        armature_prim = stage.GetPrimAtPath(f"{path}/Model/Armature")
        for op in UsdGeom.Xformable(armature_prim).GetOrderedXformOps():
            if op.GetOpName() == "xformOp:orient":
                op.Set(Gf.Quatf(1, 0, 0, 0))  # glTF由来の軸変換を中立化

        skel_root_prim = stage.GetPrimAtPath(f"{path}/Model/Armature/Hips")
        skel_prim = stage.GetPrimAtPath(f"{path}/Model/Armature/Hips/Skeleton")
        skel = UsdSkel.Skeleton(skel_prim)
        joint_tokens = list(skel.GetJointsAttr().Get())
        leaf_to_idx = {str(j).split("/")[-1]: idx for idx, j in enumerate(joint_tokens)}

        anim_path = skel_root_prim.GetPath().AppendChild("Anim")
        anim = UsdSkel.Animation.Define(stage, anim_path)
        anim.CreateJointsAttr().Set(skel.GetJointsAttr().Get())
        rest = skel.GetRestTransformsAttr().Get()
        translations, rotations, scales = [], [], []
        for m in rest:
            t, r, s = UsdSkel.DecomposeTransform(m)
            translations.append(t)
            rotations.append(r)
            scales.append(s)
        t_attr = anim.CreateTranslationsAttr()
        r_attr = anim.CreateRotationsAttr()
        anim.CreateScalesAttr().Set(scales)
        t_attr.Set(translations)
        r_attr.Set(rotations)

        binding = UsdSkel.BindingAPI.Apply(skel_root_prim)
        binding.CreateAnimationSourceRel().SetTargets([anim_path])

        return avatar_prim, translate_op, t_attr, r_attr, leaf_to_idx

    # ------------------------------------------------------------------
    # reset / step
    # ------------------------------------------------------------------
    def reset_humans(self) -> None:
        """各人物の位置・ゴールをNavMesh上の新しいランダム点へ再サンプリングし、
        アバターの描画位置(x,y,z)も瞬間移動させる(prim自体は作り直さない)"""
        if self._controller is None:
            return
        from pxr import Gf

        for i, state in enumerate(self._states):
            spawn, goal = self._sample_spawn_and_goal_xyz()
            if spawn is not None:
                sx, sy, sz = spawn
                self._controller.set_position(state.agent_id, (sx, sy))
                self._pool.teleport_agent_world_xy(i, (sx, sy))
                state.height_z = sz
                state.translate_op.Set(Gf.Vec3d(sx, sy, sz))
                state.capsule_translate_op.Set(Gf.Vec3d(sx, sy, sz + state.capsule_half_height))
            if goal is not None:
                state.goal_xy = goal

    def pre_physics_step(self, dt: float) -> None:
        """物理ステップ実行前の更新: ORCA→ai4animationpy(歩行アニメーション+水平位置)の順に
        進め、高さ追従用カプセルを現在位置へteleportする(水平方向の壁越え防止はORCAの
        静的障害物が担うため、カプセル自体は物理制御しない)"""
        if self._controller is None or self._pool is None:
            return

        for i, state in enumerate(self._states):
            pos = np.array(self._controller.get_position(state.agent_id))

            # スタック検出: 移動できていない状態が続いたら強制的にゴールを選び直す
            if state.last_pos is not None:
                moved = float(np.hypot(pos[0] - state.last_pos[0], pos[1] - state.last_pos[1]))
                state.stuck_steps = state.stuck_steps + 1 if moved < STUCK_MOVE_THRESHOLD else 0
            state.last_pos = (float(pos[0]), float(pos[1]))

            to_goal = np.array(state.goal_xy) - pos
            dist = float(np.linalg.norm(to_goal))
            if dist > GOAL_REACH_THRESHOLD and state.stuck_steps < STUCK_STEPS_LIMIT:
                pref = tuple(to_goal / dist * self._max_speed)
            else:
                reason = "スタック検出" if dist > GOAL_REACH_THRESHOLD else "ゴール"
                new_goal = self._sample_goal_min_dist_xy(tuple(pos))
                if new_goal is not None:
                    state.goal_xy = new_goal
                    state.stuck_steps = 0
                pref = (0.0, 0.0)
            self._controller.set_preferred_velocity(state.agent_id, pref)

        self._controller.step()

        velocities = [self._controller.get_velocity(s.agent_id) for s in self._states]
        self._pool.set_velocities_world_xy(velocities)
        self._pool.step(dt)

        world_positions = self._pool.ground_positions_world_xy()
        for state, (wx, wy) in zip(self._states, world_positions):
            self._controller.set_position(state.agent_id, (wx, wy))
            self._cct_iface.set_position(state.capsule_path, (wx, wy, state.height_z + state.capsule_half_height))
            self._cct_iface.set_move(state.capsule_path, (0.0, 0.0, 0.0))

    def post_physics_step(self) -> None:
        """物理ステップ実行後の更新: カプセルの高さ(Z)を読み戻してUSDへ反映する
        translate_opのX,Yはスポーン時の固定値のまま変更しない(Hipsボーンのローカル移動量との
        二重加算を防ぐため、水平位置の更新はHips側だけが担う)"""
        if self._controller is None or self._pool is None:
            return

        from pxr import Gf

        for state in self._states:
            cur = state.translate_op.Get()
            resolved = state.capsule_translate_op.Get()
            state.height_z = float(resolved[2]) - state.capsule_half_height
            state.translate_op.Set(Gf.Vec3d(cur[0], cur[1], state.height_z))

        all_bones = self._pool.all_bone_local_transforms()
        for state, bones in zip(self._states, all_bones):
            _push_bones_to_usd(bones, state)

    def get_world_positions_xy(self) -> list[tuple[float, float]]:
        """ロボット側の距離ベース接触判定(_check_human_contact)用"""
        if self._pool is None:
            return []
        return self._pool.ground_positions_world_xy()


def _ensure_model_usd() -> str:
    """Model.glb -> USD変換をキャッシュ付きで行う(assets/ai4animation/_generated/に
    生成物を保存、.gitignore対象)Y-up->Z-up変換はここでは焼き込まない(ルートXform側で行う)"""
    import asyncio
    import os

    out_dir = os.path.join(os.path.dirname(MODEL_GLB), "_generated")
    out_path = os.path.join(out_dir, "Model.usd")
    if os.path.exists(out_path):
        return out_path
    os.makedirs(out_dir, exist_ok=True)

    import omni.kit.app
    import omni.kit.asset_converter as ac

    async def _convert():
        ctx = ac.AssetConverterContext()
        ctx.use_meter_as_world_unit = True
        ctx.embed_textures = True
        task = ac.get_instance().create_converter_task(MODEL_GLB, out_path, None, ctx)
        ok = await task.wait_until_finished()
        if not ok:
            raise RuntimeError(f"Model.glb -> USD変換に失敗: {task.get_status()} {task.get_error_message()}")

    app = omni.kit.app.get_app()
    fut = asyncio.ensure_future(_convert())
    while not fut.done():
        app.update()
    if fut.exception():
        raise fut.exception()
    return out_path


def _stitch_segments_to_chains(
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
    tol: float = 1e-3,
) -> list[list[tuple[float, float]]]:
    """端点が(誤差tol以内で)一致するセグメント同士をつなぎ合わせ、連続したポリライン/ポリゴンの
    頂点列に復元する(_register_navmesh_boundary_obstacles参照)"""

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
                break  # 一周して閉じた(終点=始点は重複させず、閉じ辺はRVO2側で自動的に補われる)
            chain.append(tail)

        chains.append(chain)

    return chains


def _push_bones_to_usd(bone_transforms: dict, state: _HumanState) -> None:
    """ボーンのローカル姿勢をUSD SkelAnimationへ書き込む"""
    from pxr import Gf

    tvals = list(state.translations_attr.Get())
    rvals = list(state.rotations_attr.Get())
    for name, idx in state.leaf_to_idx.items():
        if name not in BONE_PARENTS:
            continue
        local_m = bone_transforms[name]
        pos = local_m[:3, 3]
        rot_row = local_m[:3, :3].T
        quat = Gf.Matrix3d(*rot_row.flatten().tolist()).ExtractRotation().GetQuat()
        tvals[idx] = Gf.Vec3f(*pos.tolist())
        rvals[idx] = Gf.Quatf(quat)
    state.translations_attr.Set(tvals)
    state.rotations_attr.Set(rvals)
