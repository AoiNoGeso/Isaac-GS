"""gs.usdc / floor_mesh.usd / wall_mesh.usd を統合しstage.usdaを生成する。
CollisionAPI付与とNavMeshVolumeの自動配置まで行う(NavMesh自体のbakeは環境起動時にランタイムで実行される)。"""

import os


def _apply_scale(prim, scale: float, Gf, UsdGeom):
    """xformOp:scaleを設定する(既存があれば上書き)"""
    xformable = UsdGeom.Xformable(prim)
    scale_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale_op = op
            break
    if scale_op is None:
        scale_op = xformable.AddScaleOp()
    scale_op.Set(Gf.Vec3f(scale, scale, scale))


def run(
    input_dir: str,
    gs_filename: str = "gs.usdc",
    floor_filename: str = "floor_mesh.usd",
    wall_filename: str = "wall_mesh.usd",
    output_filename: str = "stage.usda",
    margin_xy: float = 3.0,
    margin_z_bot: float = 2.0,
    margin_z_top: float = 5.0,
    scale: float = 1.0,
):
    gs_path = os.path.join(input_dir, gs_filename)
    floor_path = os.path.join(input_dir, floor_filename)
    wall_path = os.path.join(input_dir, wall_filename)

    for path in [gs_path, floor_path, wall_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"必要なファイルが見つかりません: {path}")

    out_path = os.path.abspath(os.path.join(input_dir, output_filename))
    if os.path.exists(out_path):
        os.remove(out_path)

    print("Starting SimulationApp...")
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})

    import omni.usd
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

    # Step 1: ステージ作成・prim配置・保存
    stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # defaultPrimを"env"にすることで参照時のパス解決が安定する
    env_prim = UsdGeom.Xform.Define(stage, "/env")
    stage.SetDefaultPrim(env_prim.GetPrim())

    splat = stage.OverridePrim("/env/gs")
    splat.GetReferences().AddReference(f"./{gs_filename}")
    if scale != 1.0:
        _apply_scale(splat, scale, Gf, UsdGeom)

    floor_prim = stage.OverridePrim("/env/floor_mesh")
    floor_prim.GetReferences().AddReference(f"./{floor_filename}")
    if scale != 1.0:
        _apply_scale(floor_prim, scale, Gf, UsdGeom)
    UsdGeom.Imageable(floor_prim).MakeInvisible()

    wall_prim = stage.OverridePrim("/env/wall_mesh")
    wall_prim.GetReferences().AddReference(f"./{wall_filename}")
    if scale != 1.0:
        _apply_scale(wall_prim, scale, Gf, UsdGeom)
    UsdGeom.Imageable(wall_prim).MakeInvisible()

    stage.GetRootLayer().Save()
    print("Step 1 完了: ステージ作成")

    # Step 2: CollisionAPIの付与
    stage2 = Usd.Stage.Open(out_path)

    mat_prim = UsdShade.Material.Define(stage2, "/env/PhysicsMaterial")
    UsdPhysics.MaterialAPI.Apply(mat_prim.GetPrim())
    PhysxSchema.PhysxMaterialAPI.Apply(mat_prim.GetPrim())

    def apply_collision(root_path: str) -> int:
        root = stage2.GetPrimAtPath(root_path)
        if not root.IsValid():
            return 0
        count = 0
        for prim in Usd.PrimRange(root):
            if prim.GetTypeName() == "Mesh":
                UsdPhysics.CollisionAPI.Apply(prim)
                mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
                mesh_api.CreateApproximationAttr(UsdPhysics.Tokens.none)
                for child in prim.GetChildren():
                    if child.GetTypeName() == "GeomSubset":
                        child.SetActive(False)
                count += 1
        UsdShade.MaterialBindingAPI.Apply(root).Bind(
            mat_prim,
            UsdShade.Tokens.strongerThanDescendants,
            "physics",
        )
        return count

    floor_count = apply_collision("/env/floor_mesh")
    wall_count = apply_collision("/env/wall_mesh")

    # 衝突検知にはwall_meshのみContactReportAPIが必要
    wall_root = stage2.GetPrimAtPath("/env/wall_mesh")
    for prim in Usd.PrimRange(wall_root):
        if prim.GetTypeName() == "Mesh":
            PhysxSchema.PhysxContactReportAPI.Apply(prim)

    stage2.GetRootLayer().Save()
    print(f"Step 2 完了: CollisionAPI + PhysicsMaterial 付与 (floor={floor_count}, wall={wall_count}), PhysxContactReportAPI 付与 (wall)")

    # Step 3: floor_meshのAABBからNavMeshVolumeを自動配置
    omni.usd.get_context().open_stage(out_path)
    for _ in range(30):
        app.update()

    active_stage = omni.usd.get_context().get_stage()

    bb_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
    )
    floor_root = active_stage.GetPrimAtPath("/env/floor_mesh")
    aabb = bb_cache.ComputeWorldBound(floor_root).GetBox()
    bmin, bmax = aabb.GetMin(), aabb.GetMax()
    print(
        f"floor_mesh AABB: min=({bmin[0]:.2f},{bmin[1]:.2f},{bmin[2]:.2f}) "
        f"max=({bmax[0]:.2f},{bmax[1]:.2f},{bmax[2]:.2f})"
    )

    # Z-up(X/Yが水平, Zが垂直)。extentが±0.5なのでscaleがそのまま辺長になる
    sx = (bmax[0] - bmin[0]) + margin_xy * 2
    sy = (bmax[1] - bmin[1]) + margin_xy * 2
    sz = (bmax[2] - bmin[2]) + margin_z_bot + margin_z_top

    cx = (bmin[0] + bmax[0]) / 2.0
    cy = (bmin[1] + bmax[1]) / 2.0
    cz = (bmin[2] + bmax[2]) / 2.0 + (margin_z_top - margin_z_bot) / 2.0

    vol_prim = active_stage.DefinePrim("/env/NavMeshVolume", "NavMeshVolume")
    vol_prim.CreateAttribute("nav:area", Sdf.ValueTypeNames.Token).Set("Walkable")
    vol_prim.CreateAttribute("nav:volume:type", Sdf.ValueTypeNames.Token).Set("Include")
    vol_prim.CreateAttribute("extent", Sdf.ValueTypeNames.Float3Array).Set(
        Vt.Vec3fArray([Gf.Vec3f(-0.5, -0.5, -0.5), Gf.Vec3f(0.5, 0.5, 0.5)])
    )

    xformable = UsdGeom.Xformable(vol_prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(cx, cy, cz))
    xformable.AddRotateZYXOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    xformable.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))

    active_stage.GetRootLayer().Save()
    print(
        f"Step 3 完了: NavMeshVolume 配置 "
        f"center=({cx:.2f},{cy:.2f},{cz:.2f}) half_scale=({sx:.2f},{sy:.2f},{sz:.2f})"
    )

    print(f"\nStage saved: {out_path}")
    print("NavMeshは環境起動時(envs/isaac_env.py)にランタイムでbakeされるため、ここでの手動bakeは不要")

    print("\n--- Prim 構成 ---")
    for prim in Usd.PrimRange(active_stage.GetPseudoRoot()):
        print(f"  {prim.GetPath()} [{prim.GetTypeName()}]")

    app.close()
