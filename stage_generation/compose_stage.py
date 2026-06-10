import argparse
import os

from isaacsim import SimulationApp


def main():
    # 引数の設定（対象ディレクトリを1つ受け取る）
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_dir",
        required=True,
        help="Directory containing gs.usdc and mesh.usd",
    )
    args = parser.parse_args()

    # 出力パスの構築
    out_path = os.path.join(args.input_dir, "stage.usda")

    if os.path.exists(out_path):
        os.remove(out_path)

    # SimulationAppの起動
    print("Starting SimulationApp...")
    app = SimulationApp({"headless": True})

    # Isaac Simの仕様上，pxr関連はSimulationApp起動後にインポートする必要があります
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)  # Y-up
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    # 視覚: 3DGS
    splat = stage.OverridePrim("/World/gs_splat")
    splat.GetReferences().AddReference("./gs.usdc")

    # 物理コライダ: メッシュ (invisible)
    mesh_prim = stage.OverridePrim("/World/mesh_collider")
    mesh_prim.GetReferences().AddReference("./mesh.usd")
    UsdGeom.Imageable(mesh_prim).MakeInvisible()

    stage.GetRootLayer().Save()

    # 再オープンして Mesh プリムに CollisionAPI を付与
    stage2 = Usd.Stage.Open(out_path)
    collider_root = stage2.GetPrimAtPath("/World/mesh_collider")
    mesh_count = 0

    for prim in Usd.PrimRange(collider_root):
        if prim.GetTypeName() == "Mesh":
            UsdPhysics.CollisionAPI.Apply(prim)
            mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_api.CreateApproximationAttr(UsdPhysics.Tokens.none)
            mesh_count += 1

    stage2.GetRootLayer().Save()
    print(f"Stage saved: {out_path}")
    print(f"Collision API applied to {mesh_count} mesh(es)")

    # プリム構成を確認
    for prim in Usd.PrimRange(stage2.GetPseudoRoot()):
        print(f"  {prim.GetPath()} [{prim.GetTypeName()}]")

    app.close()


if __name__ == "__main__":
    main()
