"""omni.kit.asset_converter によるメッシュ変換 + source_up_axis -> Z-up 回転焼き込み"""

import asyncio


def run(input: str, output: str, source_up_axis: str = "Y"):
    print("Starting SimulationApp...")
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})

    # Isaac Sim の仕様上, omni 関連は SimulationApp 起動後にインポートする必要がある
    import numpy as np
    import omni.kit.asset_converter as ac
    from pxr import Gf, Usd, UsdGeom, Vt

    from stage_generation.coords import apply_matrix_to_vec3_array, compute_extent, rotation_matrix

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

    print(f"Converting: {input} -> {output}")
    asyncio.get_event_loop().run_until_complete(convert(input, output))

    R = rotation_matrix(source_up_axis)
    stage = Usd.Stage.Open(output)
    count = 0
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if prim.GetTypeName() != "Mesh":
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        if pts:
            pts_np = apply_matrix_to_vec3_array(np.array(pts), R)
            mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts_np]))
            # BBoxCache は extent を優先するため points と合わせて更新
            new_min, new_max = compute_extent(pts_np)
            mesh.GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*new_min), Gf.Vec3f(*new_max)]))
        nrm = mesh.GetNormalsAttr().Get()
        if nrm:
            nrm_np = apply_matrix_to_vec3_array(np.array(nrm), R)
            mesh.GetNormalsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*n) for n in nrm_np]))
        count += 1
    stage.GetRootLayer().Save()
    print(f"Applied {source_up_axis}-up -> Z-up rotation to {count} mesh prim(s)")

    app.close()
