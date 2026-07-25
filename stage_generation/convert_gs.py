"""usd_convert_gsplat による GS 変換 + source_up_axis -> Z-up 回転焼き込み"""

import glob
import os
import subprocess
import sys


def _find_gsplat_paths():
    """GSPLAT_DIR/USD_LIBSを環境変数から取得し、無ければ.venv内から自動検出する"""
    gsplat_dir = os.environ.get("GSPLAT_DIR")
    usd_libs = os.environ.get("USD_LIBS")

    if gsplat_dir and usd_libs:
        return gsplat_dir, usd_libs

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    extscache = os.path.join(repo_root, ".venv", "lib", "python3.12", "site-packages", "isaacsim", "extscache")

    if not gsplat_dir:
        matches = sorted(glob.glob(os.path.join(extscache, "omni.kit.converter.gsplat-*", "pip_prebundle")))
        gsplat_dir = matches[0] if matches else None
    if not usd_libs:
        matches = sorted(glob.glob(os.path.join(extscache, "omni.usd.libs-*")))
        usd_libs = matches[0] if matches else None

    if not gsplat_dir or not usd_libs:
        raise RuntimeError(
            "GSPLAT_DIR / USD_LIBS が自動検出できませんでした。"
            "環境変数 GSPLAT_DIR / USD_LIBS を設定してください "
            f"(探索先: {extscache})"
        )
    return gsplat_dir, usd_libs


def run(input_dir: str, output_dir: str, source_up_axis: str = "Y"):
    gsplat_dir, usd_libs = _find_gsplat_paths()

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{gsplat_dir}:{usd_libs}:{env.get('PYTHONPATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{usd_libs}/bin:{env.get('LD_LIBRARY_PATH', '')}"

    usdc_path = os.path.join(output_dir, "gs.usdc")
    cmd = [
        sys.executable,
        "-m",
        "usd_convert_gsplat",
        "-i",
        input_dir,
        "-o",
        usdc_path,
        "--up-axis",
        "Y",  # usd_convert_gsplat側の中間フラグ(coords.pyと同じく"Y"のみ対応)
    ]

    print("Running usd_convert_gsplat...")
    subprocess.run(cmd, env=env, check=True)
    print(f"Done: successfully converted to {usdc_path}")

    print("Starting SimulationApp...")
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})

    import numpy as np
    from pxr import Usd

    from stage_generation.coords import rotation_matrix, rotation_quat

    stage = Usd.Stage.Open(usdc_path)
    splat_prim = stage.GetDefaultPrim()
    print(f"Default prim: {splat_prim.GetPath()} [{splat_prim.GetTypeName()}]")

    R = rotation_matrix(source_up_axis)

    pos_attr = splat_prim.GetAttribute("positions")
    positions = np.array(pos_attr.Get())
    positions = (R @ positions.T).T
    pos_attr.Set(positions)

    rot_q = rotation_quat(source_up_axis)
    orient_attr = splat_prim.GetAttribute("orientations")
    orientations = orient_attr.Get()
    rotated = [rot_q * q for q in orientations]
    orient_attr.Set(rotated)

    stage.GetRootLayer().Save()
    print(f"Done: rotation baked into USDC ({source_up_axis}-up -> Z-up)")
    app.close()
