import argparse
import asyncio
import os

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
    app.close()


if __name__ == "__main__":
    main()
