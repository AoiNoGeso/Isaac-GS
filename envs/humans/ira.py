"""IRA (isaacsim.replicator.agent) 人物キャラの注入・管理ロジック。
注入手順(payload arc / timeline停止→設定→再生 / idle属性禁止)はIsaac Simランタイム制約のため変更禁止。"""

import numpy as np

MOTION_LIBRARY_PRIM_PATH = "/World/HumanMotionLibrary"
HUMANS_ROOT = "/World/Humans"
CHAR_ASSET_DIR = "Isaac/People/Characters/"


class IRAHumanManager:
    """IRA人物キャラの注入・NavMeshサンプリング・再配置を担当する。
    `_check_human_contact`(ContactSensor依存)は呼び出し元のPointNavIsaacEnvに残す。"""

    def __init__(self, env_cfg, world, inav):
        self.env_cfg = env_cfg
        self._world = world
        self._inav = inav

    # ------------------------------------------------------------------
    # IRA 人物キャラ注入
    # ------------------------------------------------------------------
    def resolve_character_urls(self, num: int, seed: int = 0) -> list[str]:
        """Isaac/People/Characters/ から <name>/<name>.usd を num 体分(seedでシャッフル)返す"""
        import carb
        import omni.client
        from omni.metropolis.utils.isaac_sim_util import resolve_asset_path

        base = resolve_asset_path(CHAR_ASSET_DIR)
        result, entries = omni.client.list(base)
        if result != omni.client.Result.OK or not entries:
            carb.log_error(f"[IRAHumanManager] キャラクター一覧の取得に失敗: {base}")
            return []

        urls: list[str] = []
        for e in entries:
            rel = e.relative_path.rstrip("/")
            candidate = f"{base.rstrip('/')}/{rel}/{rel}.usd"
            r, _ = omni.client.stat(candidate)
            if r == omni.client.Result.OK:
                urls.append(candidate)

        if not urls:
            carb.log_error(f"[IRAHumanManager] 有効なキャラクターUSDが見つかりません: {base}")
            return []

        rng = np.random.default_rng(seed)
        rng.shuffle(urls)
        return [urls[i % len(urls)] for i in range(num)]

    def sample_navmesh_point_for_human(self, z_offset: float = 0.0) -> np.ndarray | None:
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
            carb.log_error("[IRAHumanManager] MotionLibrary のロードに失敗")
            return []

        char_urls = self.resolve_character_urls(num_humans, seed=seed)
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
                carb.log_warn(f"[IRAHumanManager] Human_{i} SkelRoot 未検出, スキップ")
                continue

            char_prim = stage.GetPrimAtPath(char_prim_path)
            pt = self.sample_navmesh_point_for_human()
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

    def inject_ira_humans(self, stage, collect_timeout: int = 600) -> list:
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
            carb.log_error("[IRAHumanManager] 有効な人物キャラが1体もセットアップできませんでした")
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
                carb.log_warn("[IRAHumanManager] AgentsManager 収集タイムアウト")
                break

        return manager.get_agents_by_type(IRA_Character)

    def relocate_human(self, character, pt: np.ndarray) -> None:
        """人物をptへ再配置する"""
        import carb

        bh = character.get_bh_agent()
        if bh is None:
            return
        bh.teleport(target=carb.Float3(float(pt[0]), float(pt[1]), float(pt[2])), facing=None)

    def reset_humans(self, characters: list) -> None:
        """各人物をNavMesh上の新しいランダム点へ再配置する"""
        for character in characters:
            pt = self.sample_navmesh_point_for_human()
            if pt is not None:
                self.relocate_human(character, pt)
        self._world.step(render=False)

    def regenerate(self, stage, characters: list) -> list:
        """人物prim + HumanMotionLibraryを削除し, inject_ira_humansで作り直す。
        omni.anim.behavior.coreのモーションマッチングが累積30万step程度でNaNを生成し
        該当キャラクターが恒久フリーズするバグの回避策(tests/manual/ira_freeze_repro.pyで検証済み)"""
        import omni.kit.app
        import omni.kit.commands
        import omni.timeline

        # timeline再生中にアクティブなキャラクターprimを削除するとネイティブ側がクラッシュするため、
        # 削除前に必ずtimelineを停止する
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()

        app = omni.kit.app.get_app()
        for _ in range(3):
            app.update()

        omni.kit.commands.execute("DeletePrims", paths=[HUMANS_ROOT, MOTION_LIBRARY_PRIM_PATH])
        for _ in range(3):
            self._world.step(render=False)

        return self.inject_ira_humans(stage)
