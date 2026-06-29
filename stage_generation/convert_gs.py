import argparse
import os
import subprocess
import sys

import numpy as np
from isaacsim import SimulationApp

# 環境変数用のパス定義 (環境に合わせて変更してください)
GSPLAT_DIR = os.path.expanduser(
    "~/env_Isaac-GS/lib/python3.12/site-packages/isaacsim/extscache/omni.kit.converter.gsplat-0.1.14+110.1.0.lx64.r.cp312.u7f4/pip_prebundle"
)
USD_LIBS = os.path.expanduser(
    "~/env_Isaac-GS/lib/python3.12/site-packages/isaacsim/extscache/omni.usd.libs-1.0.3+6312fa25.lx64.r.cp312"
)


def main():
    # 引数の設定
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", required=True, help="Input directory path")
    parser.add_argument(
        "-o", "--output_dir", required=True, help="Output directory path"
    )
    args = parser.parse_args()

    # 環境変数の構築
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{GSPLAT_DIR}:{USD_LIBS}:{env.get('PYTHONPATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{USD_LIBS}/bin:{env.get('LD_LIBRARY_PATH', '')}"

    # 変換コマンドの実行
    usdc_path = os.path.join(args.output_dir, "gs.usdc")
    cmd = [
        sys.executable,
        "-m",
        "usd_convert_gsplat",
        "-i",
        args.input_dir,
        "-o",
        usdc_path,
        "--up-axis",
        "Y",
    ]

    print("Running usd_convert_gsplat...")
    subprocess.run(cmd, env=env, check=True)
    print(f"Done: successfully converted to {usdc_path}")

    # 回転処理: Y-up → Z-up (90°X 回転)
    print("Starting SimulationApp...")
    app = SimulationApp({"headless": True})

    from pxr import Gf, Usd

    stage = Usd.Stage.Open(usdc_path)
    splat_prim = stage.GetDefaultPrim()
    print(f"Default prim: {splat_prim.GetPath()} [{splat_prim.GetTypeName()}]")

    # -Y-up → Z-up: X軸 -90°回転
    # (x,y,z) → (x,z,-y): -Y が +Z へ
    rot_neg90x = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)

    # positions の回転
    pos_attr = splat_prim.GetAttribute("positions")
    positions = np.array(pos_attr.Get())
    positions = (rot_neg90x @ positions.T).T
    pos_attr.Set(positions)

    # orientations の回転（-90°X クォータニオン）
    import math
    rot_q = Gf.Quatf(math.cos(math.pi / 4), -math.sin(math.pi / 4), 0.0, 0.0)
    orient_attr = splat_prim.GetAttribute("orientations")
    orientations = orient_attr.Get()
    rotated = [rot_q * q for q in orientations]
    orient_attr.Set(rotated)

    stage.GetRootLayer().Save()
    print("Done: rotation baked into USDC (-Y-up → Z-up)")
    app.close()


if __name__ == "__main__":
    main()
