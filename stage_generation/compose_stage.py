"""
ステージ生成スクリプト

gs.usdc / floor_mesh.usd / wall_mesh.usd を統合し，
CollisionAPI 付与・NavMeshVolume 自動配置を行った stage.usda を生成する．

NavMesh Bake は API から実行できないため，生成後に Isaac Sim GUI で手動実施．

実行方法:
  cd ~/Programs/Isaac-GS
  uv run stage_generation/compose_stage.py -i sample_data/stages/corridor1
"""

import argparse
import os

from isaacsim import SimulationApp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_dir",
        required=True,
        help="Directory containing gs.usdc, floor_mesh.usd, wall_mesh.usd",
    )
    args = parser.parse_args()

    gs_path = os.path.join(args.input_dir, "gs.usdc")
    floor_path = os.path.join(args.input_dir, "floor_mesh.usd")
    wall_path = os.path.join(args.input_dir, "wall_mesh.usd")

    for path in [gs_path, floor_path, wall_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"必要なファイルが見つかりません: {path}")

    out_path = os.path.abspath(os.path.join(args.input_dir, "stage.usda"))
    if os.path.exists(out_path):
        os.remove(out_path)

    print("Starting SimulationApp...")
    app = SimulationApp({"headless": True})

    import omni.usd
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

    # ─────────────────────────────────────────────────────────────────
    # Step 1: ステージ作成・prim 配置・保存
    # ─────────────────────────────────────────────────────────────────
    stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Xform.Define(stage, "/World/env")

    splat = stage.OverridePrim("/World/env/gs")
    splat.GetReferences().AddReference("./gs.usdc")

    floor_prim = stage.OverridePrim("/World/env/floor_mesh")
    floor_prim.GetReferences().AddReference("./floor_mesh.usd")
    UsdGeom.Imageable(floor_prim).MakeInvisible()

    wall_prim = stage.OverridePrim("/World/env/wall_mesh")
    wall_prim.GetReferences().AddReference("./wall_mesh.usd")
    UsdGeom.Imageable(wall_prim).MakeInvisible()

    physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics_scene.CreateGravityDirectionAttr().Set((0.0, -1.0, 0.0))
    physics_scene.CreateGravityMagnitudeAttr().Set(9.81)

    stage.GetRootLayer().Save()
    print("Step 1 完了: ステージ作成")

    # ─────────────────────────────────────────────────────────────────
    # Step 2: CollisionAPI の付与
    # ─────────────────────────────────────────────────────────────────
    stage2 = Usd.Stage.Open(out_path)

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
                count += 1
        return count

    floor_count = apply_collision("/World/env/floor_mesh")
    wall_count = apply_collision("/World/env/wall_mesh")
    stage2.GetRootLayer().Save()
    print(f"Step 2 完了: CollisionAPI 付与 (floor={floor_count}, wall={wall_count})")

    # ─────────────────────────────────────────────────────────────────
    # Step 3: NavMeshVolume 配置（AABB を floor_mesh から自動計算）
    # ─────────────────────────────────────────────────────────────────
    omni.usd.get_context().open_stage(out_path)
    for _ in range(30):
        app.update()

    active_stage = omni.usd.get_context().get_stage()

    bb_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
    )
    floor_root = active_stage.GetPrimAtPath("/World/env/floor_mesh")
    aabb = bb_cache.ComputeWorldBound(floor_root).GetBox()
    bmin, bmax = aabb.GetMin(), aabb.GetMax()
    print(
        f"floor_mesh AABB: min=({bmin[0]:.2f},{bmin[1]:.2f},{bmin[2]:.2f}) "
        f"max=({bmax[0]:.2f},{bmax[1]:.2f},{bmax[2]:.2f})"
    )

    MARGIN_XZ = 3.0
    MARGIN_Y_BOT = 2.0
    MARGIN_Y_TOP = 5.0

    cx = (bmin[0] + bmax[0]) / 2.0
    cy = (bmin[1] + bmax[1]) / 2.0
    cz = (bmin[2] + bmax[2]) / 2.0
    sx = (bmax[0] - bmin[0]) / 2.0 + MARGIN_XZ
    sy = (bmax[1] - bmin[1]) / 2.0 + (MARGIN_Y_BOT + MARGIN_Y_TOP) / 2.0
    sz = (bmax[2] - bmin[2]) + MARGIN_XZ * 2

    vol_prim = active_stage.DefinePrim("/World/NavMeshVolume", "NavMeshVolume")
    vol_prim.CreateAttribute("nav:area", Sdf.ValueTypeNames.String).Set("Walkable")
    vol_prim.CreateAttribute("nav:volume:type", Sdf.ValueTypeNames.String).Set(
        "Include"
    )
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
    print("\n⚠️  NavMesh Bake は GUI で手動実施してください:")
    print("  1. Isaac Sim で stage.usda を開く")
    print("  2. Window > Navigation > NavMesh パネルを開く")
    print("  3. Bake ボタンを押す")
    print("  4. File > Save で保存する")

    print("\n--- Prim 構成 ---")
    for prim in Usd.PrimRange(active_stage.GetPseudoRoot()):
        print(f"  {prim.GetPath()} [{prim.GetTypeName()}]")

    app.close()


if __name__ == "__main__":
    main()
