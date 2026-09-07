"""IRAのpayload生成、再生後のハンドル取得、ORCA速度から移動タスクへの変換。

40万step試験と同じアセット経路・payload・Behavior APIを使用する。
拡張はWorld生成前、spawnはTimeline停止中、bindは再生開始後に呼ぶ。
"""

from __future__ import annotations

from typing import Callable, Sequence

MOTION_LIBRARY_PATH = "/World/HumanMotionLibrary"


def enable_ira_extensions() -> None:
    """IRA関連の拡張機能を有効化する。isaac_env.pyのWorld()構築より前に呼ぶこと。"""
    from isaacsim.core.utils.extensions import enable_extension

    for ext in (
        "isaacsim.replicator.agent.core",
        "omni.anim.behavior.core",
        "omni.anim.navigation.core",
        "omni.anim.retarget.core",
    ):
        enable_extension(ext)


def resolve_character_and_library(
    character_override: str | None, motion_library_override: str
) -> tuple[str, str]:
    """キャラクター/モーションライブラリのUSD URLを解決する。character_overrideがNoneなら
    `Isaac/People/Characters/`配下から最初に見つかった有効なキャラクターを自動選択する。"""
    import omni.client
    from omni.metropolis.utils.isaac_sim_util import resolve_asset_path

    motion_library_url = resolve_asset_path(motion_library_override)

    if character_override is not None:
        return resolve_asset_path(character_override), motion_library_url

    base = resolve_asset_path("Isaac/People/Characters/").rstrip("/")
    result, entries = omni.client.list(base)
    if result != omni.client.Result.OK:
        raise RuntimeError(f"IRAキャラクター一覧の取得に失敗しました: {base}")
    for entry in sorted(entries, key=lambda x: x.relative_path):
        name = entry.relative_path.rstrip("/")
        url = f"{base}/{name}/{name}.usd"
        if omni.client.stat(url)[0] == omni.client.Result.OK:
            return url, motion_library_url
    raise RuntimeError(f"利用可能なIRAキャラクターが見つかりません: {base}")


def spawn_humans(
    stage,
    ctx,
    humans_root: str,
    spawns_xyz: Sequence[tuple[float, float, float]],
    character_url: str,
    motion_library_url: str,
    app_update_fn: Callable[[], None],
    poll_frames: int = 600,
) -> list[str]:
    """モーションライブラリ+各人物キャラクターをpayload arcでスポーンし、
    ApplyBehaviorAgentAPICommand + NavMeshExcludeAPIを適用する。
    スポーンしたSkelRootのパス文字列のリストを、spawns_xyzと同じ順序で返す。"""
    import omni.kit.commands
    import BehaviorSchema
    import NavSchema
    from pxr import Gf, UsdGeom, UsdSkel
    from omni.metropolis.pipeline.usd_util import set_prim_pos

    if not stage.GetPrimAtPath(humans_root).IsValid():
        UsdGeom.Xform.Define(stage, humans_root)

    if not stage.GetPrimAtPath(MOTION_LIBRARY_PATH).IsValid():
        omni.kit.commands.execute(
            "CreatePayload",
            usd_context=ctx,
            path_to=MOTION_LIBRARY_PATH,
            asset_path=motion_library_url,
        )
        for _ in range(poll_frames):
            app_update_fn()
            library = stage.GetPrimAtPath(MOTION_LIBRARY_PATH)
            if library and library.IsA(BehaviorSchema.BehaviorMotionLibrary):
                break
        else:
            raise RuntimeError("IRAモーションライブラリの合成がタイムアウトしました")

    paths: list[str] = []
    for i, pos in enumerate(spawns_xyz):
        path = f"{humans_root}/Human_{i}"
        omni.kit.commands.execute(
            "CreatePayload", usd_context=ctx, path_to=path, asset_path=character_url
        )
        roots = []
        for _ in range(poll_frames):
            app_update_fn()
            roots = [
                p
                for p in stage.Traverse()
                if str(p.GetPath()).startswith(path + "/") and p.IsA(UsdSkel.Root)
            ]
            if roots:
                break
        else:
            raise RuntimeError(f"IRAキャラクターの合成がタイムアウトしました: {path}")

        skelroot = roots[0]
        set_prim_pos(stage.GetPrimAtPath(path), Gf.Vec3f(*pos))
        omni.kit.commands.execute(
            "ApplyBehaviorAgentAPICommand",
            skelroot_prim_paths=[skelroot.GetPath()],
            motion_library_prim_path=MOTION_LIBRARY_PATH,
            motion_library_skeleton_rig="Human",
        )
        omni.kit.commands.execute(
            "ApplyNavMeshAPICommand",
            prim_path=skelroot.GetPath(),
            api=NavSchema.NavMeshExcludeAPI,
        )
        paths.append(str(skelroot.GetPath()))
    return paths


def bind_agents(
    paths: Sequence[str],
    app_update_fn: Callable[[], None] | None = None,
    poll_frames: int = 600,
) -> list:
    """スポーン済みのSkelRootパスからIBehaviorAgentハンドルを取得し、
    IRA自前の回避(ORCAと二重に効くと衝突判定が破綻するため)を無効化して返す。

    ApplyBehaviorAgentAPICommand後、実際にget_agent()が非Noneを返すようになるまでには
    数フレームのラグがある(エージェント登録が非同期のため)。app_update_fnを渡した場合は
    全パス解決できるまでポーリングする(渡さない場合は即座に1回だけ試す)。"""
    import omni.anim.behavior.core as bh

    iface = bh.acquire_interface()
    agents: list = [None] * len(paths)
    for _ in range(poll_frames if app_update_fn is not None else 1):
        for i, p in enumerate(paths):
            if agents[i] is None:
                agents[i] = iface.get_agent(p)
        if all(a is not None for a in agents):
            break
        if app_update_fn is not None:
            app_update_fn()
    missing = [p for p, a in zip(paths, agents) if a is None]
    if missing:
        raise RuntimeError(f"IRAエージェントの取得に失敗しました: {missing}")
    for a in agents:
        a.set_auto_avoidance_enabled(False)
        a.set_obstacle_avoidance_enabled(False)
    return agents


def issue_command(
    agent,
    state,
    velocity_xy: tuple[float, float],
    position_xyz: tuple[float, float, float],
    goal_xyz: tuple[float, float, float],
    lookahead: float,
    reissue_policy: str,
    min_speed: float = 0.03,
    navmesh=None,
) -> None:
    """ORCA計算後の速度をIRAのidle/set_speed/move_toへ変換する。state.task_idを更新する。

    reissue_policy="hold": 実行中のmove_toタスクがあれば何もしない(歩容の自然さ優先、
        タスクが空くまではmove_to先=実際のゴール地点をIRA自身のnavmesh経路探索に任せる)。
    reissue_policy="reissue": 毎回タスクをキャンセルして、ORCA速度方向へのlookahead秒先の
        地点を新たなmove_to先として再発行する(ORCA回避への追従性優先、より粗い動きになりうる)。
    """
    import numpy as np
    import carb
    if reissue_policy not in ('hold', 'reissue'):
        raise ValueError(f'Unknown IRA reissue policy: {reissue_policy}')
    if not np.isfinite([*velocity_xy, *position_xyz]).all():
        from envs.human_controller.health import HumanLocomotionAnomaly
        raise HumanLocomotionAnomaly('Nonfinite IRA command')

    if reissue_policy == "hold" and state.task_id is not None and agent.is_task_running(state.task_id):
        return

    if state.task_id is not None:
        agent.cancel_task(state.task_id)

    v = np.asarray(velocity_xy, dtype=float)
    speed = float(np.linalg.norm(v))
    if speed < min_speed:
        state.task_id = agent.idle()
        return

    agent.set_speed(speed)
    if reissue_policy == "hold":
        target = carb.Float3(float(goal_xyz[0]), float(goal_xyz[1]), float(position_xyz[2]))
    else:
        target = carb.Float3(
            float(position_xyz[0] + v[0] * lookahead),
            float(position_xyz[1] + v[1] * lookahead),
            float(position_xyz[2]),
        )
    if navmesh is not None:
        # スキャン床には勾配がある。現在のZを流用すると移動先がNavMesh外になり、
        # move_toが即終了するため、先読み位置を歩行面へ投影する。
        point, _ = navmesh.query_closest_point(target=target)
        if point is None or not np.isfinite(tuple(point)).all():
            raise RuntimeError('IRA target could not be projected onto NavMesh')
        target = carb.Float3(*point)
    state.task_id = agent.move_to(target, auto_brake=False)
