"""glTF(.glb) -> USD 変換ツール(omni.kit.asset_converter)。

実行例:
  uv run utils/convert_glb2usd.py \
      --i submodules/ai4animationpy/Demos/_ASSETS_/Geno/Model.glb \
      --o assets/avatars/Model.usd
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def convert_glb_to_usd(input_path: str, output_path: str) -> None:
    """`input_path`(.glb)を`output_path`(.usd)へ変換する(SimulationApp起動済みであること)
    Y-up→Z-up回転はここでは焼き込まない(呼び出し側のルートXformで行う想定)"""
    import omni.kit.asset_converter as ac

    async def _convert():
        ctx = ac.AssetConverterContext()
        ctx.use_meter_as_world_unit = True
        ctx.embed_textures = True
        task = ac.get_instance().create_converter_task(input_path, output_path, None, ctx)
        ok = await task.wait_until_finished()
        if not ok:
            raise RuntimeError(f"glTF -> USD変換に失敗: {task.get_status()} {task.get_error_message()}")

    asyncio.get_event_loop().run_until_complete(_convert())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True, help="入力.glbファイルパス")
    parser.add_argument("-o", "--output", required=True, help="出力.usdファイルパス")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    from utils.launch_sim import launch_sim

    app = launch_sim(headless=True)

    convert_glb_to_usd(args.input, args.output)
    print(f"Done: {args.output}")

    app.close()


if __name__ == "__main__":
    main()
