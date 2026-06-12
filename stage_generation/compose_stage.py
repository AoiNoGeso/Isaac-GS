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
        help="Directory containing gs.usdc, floor_mesh.usd, and wall_mesh.usd",
    )
    args = parser.parse_args()

    # 入力ファイルの存在確認
    gs_path    = os.path.join(args.input_dir, "gs.usdc")
    floor_path = os.path.join(args.input_dir, "floor_mesh.usd")
    wall_path  = os.path.join(args.input_dir, "wall_mesh.usd")

    for path in [gs_path, floor_path, wall_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"必要なファイルが見つかりません: {path}")

    # 出力パスの構築
    out_path = os.path.join(args.input_dir, "stage.usda")
    if os.path.exists(out_path):
        os.remove(out_path)

    # SimulationAppの起動
    print("Starting SimulationApp...")
    app = SimulationApp({"headless": True})

    # Isaac Simの仕様上，pxr関連はSimulationApp起動後にインポートする必要があります
    from pxr import Usd, UsdGeom, UsdPhysics

    # ステージの作成
    stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # /World をデフォルトprimに設定
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    # /World/env を環境ルートとして定義
    UsdGeom.Xform.Define(stage, "/World/env")

    # 視覚: 3DGS → /World/env/gs
    splat = stage.OverridePrim("/World/env/gs")
    splat.GetReferences().AddReference("./gs.usdc")

    # 床メッシュ → /World/env/floor_mesh (invisible)
    floor_prim = stage.OverridePrim("/World/env/floor_mesh")
    floor_prim.GetReferences().AddReference("./floor_mesh.usd")
    UsdGeom.Imageable(floor_prim).MakeInvisible()

    # 壁メッシュ → /World/env/wall_mesh (invisible)
    wall_prim = stage.OverridePrim("/World/env/wall_mesh")
    wall_prim.GetReferences().AddReference("./wall_mesh.usd")
    UsdGeom.Imageable(wall_prim).MakeInvisible()

    stage.GetRootLayer().Save()

    # 再オープンして各 Mesh プリムに CollisionAPI を付与
    stage2 = Usd.Stage.Open(out_path)

    def apply_collision(root_path: str) -> int:
        """指定パス以下の全 Mesh プリムに CollisionAPI を付与する．"""
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
    wall_count  = apply_collision("/World/env/wall_mesh")

    stage2.GetRootLayer().Save()

    print(f"Stage saved: {out_path}")
    print(f"CollisionAPI 付与: floor_mesh={floor_count} mesh(es), wall_mesh={wall_count} mesh(es)")

    # prim構成を表示
    print("\n--- Prim 構成 ---")
    for prim in Usd.PrimRange(stage2.GetPseudoRoot()):
        print(f"  {prim.GetPath()} [{prim.GetTypeName()}]")

    app.close()


if __name__ == "__main__":
    main()
