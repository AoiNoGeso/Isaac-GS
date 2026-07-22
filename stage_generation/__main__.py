"""
stage_generation CLI

実行方法:
  uv run python -m stage_generation convert-gs   -i <input_dir> -o <output_dir>
  uv run python -m stage_generation convert-mesh -i <input> -o <output>
  uv run python -m stage_generation compose      -i <input_dir>

サブコマンドの実装 (isaacsim import) は argparse 解析後に読み込むため,
--help はいずれも Isaac Sim を起動せず即座に返る.
"""

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stage_generation")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gs = sub.add_parser("convert-gs", help="GSPLAT -> gs.usdc 変換 + Z-up 回転焼き込み")
    p_gs.add_argument("-i", "--input_dir", required=True, help="Input directory path")
    p_gs.add_argument("-o", "--output_dir", required=True, help="Output directory path")
    p_gs.add_argument(
        "--source-up-axis", choices=["Y"], default="Y", help="変換元データの up-axis (default: Y)"
    )

    p_mesh = sub.add_parser("convert-mesh", help="メッシュ -> USD 変換 + Z-up 回転焼き込み")
    p_mesh.add_argument("-i", "--input", required=True, help="Input mesh file path (PLY, OBJ, etc.)")
    p_mesh.add_argument("-o", "--output", required=True, help="Output USD file path (e.g. floor_mesh.usd)")
    p_mesh.add_argument(
        "--source-up-axis", choices=["Y"], default="Y", help="変換元データの up-axis (default: Y)"
    )

    p_compose = sub.add_parser("compose", help="gs.usdc/floor_mesh.usd/wall_mesh.usd を統合し stage.usda を生成")
    p_compose.add_argument(
        "-i", "--input_dir", required=True, help="Directory containing gs.usdc, floor_mesh.usd, wall_mesh.usd"
    )
    p_compose.add_argument("--gs-filename", default="gs.usdc")
    p_compose.add_argument("--floor-filename", default="floor_mesh.usd")
    p_compose.add_argument("--wall-filename", default="wall_mesh.usd")
    p_compose.add_argument("--output-filename", default="stage.usda")
    p_compose.add_argument("--margin-xy", type=float, default=3.0, help="水平方向マージン（X・Y）(default: 3.0)")
    p_compose.add_argument("--margin-z-bot", type=float, default=2.0, help="床下方向マージン (default: 2.0)")
    p_compose.add_argument("--margin-z-top", type=float, default=5.0, help="天井上方向マージン (default: 5.0)")
    p_compose.add_argument(
        "--scale", type=float, default=1.0, help="gs/floor/wall メッシュに適用する一様スケール (default: 1.0)"
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "convert-gs":
        from stage_generation import convert_gs

        convert_gs.run(args.input_dir, args.output_dir, source_up_axis=args.source_up_axis)
    elif args.command == "convert-mesh":
        from stage_generation import convert_mesh

        convert_mesh.run(args.input, args.output, source_up_axis=args.source_up_axis)
    elif args.command == "compose":
        from stage_generation import compose

        compose.run(
            args.input_dir,
            gs_filename=args.gs_filename,
            floor_filename=args.floor_filename,
            wall_filename=args.wall_filename,
            output_filename=args.output_filename,
            margin_xy=args.margin_xy,
            margin_z_bot=args.margin_z_bot,
            margin_z_top=args.margin_z_top,
            scale=args.scale,
        )


if __name__ == "__main__":
    main()
