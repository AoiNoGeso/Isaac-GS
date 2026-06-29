import argparse
import asyncio
import os

import numpy as np
from isaacsim import SimulationApp


def main():
    # 引数の設定
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", required=True, help="Input mesh file path (PLY, OBJ, etc.)"
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output USD file path (e.g. floor_mesh.usd)",
    )
    args = parser.parse_args()

    # SimulationAppの起動
    print("Starting SimulationApp...")
    app = SimulationApp({"headless": True})

    # Isaac Simの仕様上，omni関連はSimulationApp起動後にインポートする必要があります
    import omni.kit.asset_converter as ac

    async def convert(input_path, output_path):
        ctx = ac.AssetConverterContext()
        ctx.use_meter_as_world_unit = True
        ctx.embed_textures = False
        task = ac.get_instance().create_converter_task(
            input_path, output_path, None, ctx
        )
        ok = await task.wait_until_finished()
        if not ok:
            print("FAILED:", task.get_status(), task.get_error_message())
        else:
            print("Done:", output_path)

    print(f"Converting: {args.input} -> {args.output}")
    asyncio.get_event_loop().run_until_complete(convert(args.input, args.output))

    # -Y-up → Z-up: 全 Mesh prim の頂点・法線に -90°X 回転を直接焼き込む
    # (x, y, z) → (x, z, -y)
    from pxr import Gf, Usd, UsdGeom, Vt
    R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
    stage = Usd.Stage.Open(args.output)
    count = 0
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if prim.GetTypeName() != "Mesh":
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        if pts:
            pts_np = (R @ np.array(pts).T).T
            mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts_np]))
            # BBoxCache は extent を優先するため points と合わせて更新する
            new_min = pts_np.min(axis=0)
            new_max = pts_np.max(axis=0)
            mesh.GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*new_min), Gf.Vec3f(*new_max)]))
        nrm = mesh.GetNormalsAttr().Get()
        if nrm:
            nrm_np = (R @ np.array(nrm).T).T
            mesh.GetNormalsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*n) for n in nrm_np]))
        count += 1
    stage.GetRootLayer().Save()
    print(f"Applied -90°X rotation to {count} mesh prim(s) (Z-up correction)")

    app.close()


if __name__ == "__main__":
    main()
