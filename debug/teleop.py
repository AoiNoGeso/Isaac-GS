"""
デバッグ用テレオペスクリプト

W/S: 前進/後退  A/D: 左回転/右回転  P: 座標表示  R: リセット  Q: 終了

--num-humans > 0 の場合は人間アバター(ORCA + IRA)を注入する

実行:
  cd ~/Programs/Isaac-GS
  uv run debug/teleop.py
  uv run debug/teleop.py --num-humans 2
  uv run debug/teleop.py --stage corridor1
  uv run debug/teleop.py --robot cylinder  # 差動二輪の代わりにcmd_vel直接駆動の円柱で確認
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.config import stage_names

parser = argparse.ArgumentParser()
parser.add_argument("--num-humans", type=int, default=0)
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument(
    "--stage", type=str, choices=stage_names(), default="corridor2"
)
parser.add_argument(
    "--robot", type=str, choices=["jackal", "cylinder"], default="jackal",
    help="jackal(差動二輪) | cylinder(cmd_vel直接駆動の単純な円柱、車輪動力学の影響切り分け用)",
)
parser.add_argument(
    "--vis-goal", action="store_true", default=False, help="スポーン(青)・ゴール(赤)地点に半透明の円を表示"
)
parser.add_argument(
    "--reset",
    action="store_true",
    default=False,
    help="衝突/成功/タイムアウトで自動リセットする (指定しない場合は R キーのみで手動リセット)",
)
args = parser.parse_args()

from utils.launch_sim import launch_sim

app = launch_sim(headless=args.headless)

import carb
import numpy as np
import omni.appwindow
import omni.usd

from envs import PointNavIsaacEnv
from envs.config import EnvConfig

MARKER_RADIUS = 0.3  # m
MARKER_HEIGHT = 0.02  # m, xy平面に対して十分薄い円盤


def _make_marker(stage, path: str, color: tuple[float, float, float]):
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr("Z")
    cyl.CreateRadiusAttr(MARKER_RADIUS)
    cyl.CreateHeightAttr(MARKER_HEIGHT)
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    cyl.CreateDisplayOpacityAttr([0.4])
    prim = cyl.GetPrim()
    UsdGeom.Imageable(prim).MakeVisible()
    UsdGeom.Xformable(prim).AddTranslateOp()
    return prim


def _update_marker(prim, pos):
    from pxr import Gf

    prim.GetAttribute("xformOp:translate").Set(
        Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2]))
    )


def main():
    from envs.config import CYLINDER_ROBOT, JACKAL

    env_cfg = EnvConfig.from_preset(
        args.stage,
        num_humans=args.num_humans,
        human_speed_range=(0.8, 1.5),
        robot=CYLINDER_ROBOT if args.robot == "cylinder" else JACKAL,
    )
    env = PointNavIsaacEnv(env_cfg)
    env.reset()

    spawn_marker = goal_marker = None
    if args.vis_goal:
        stage = omni.usd.get_context().get_stage()
        spawn_marker = _make_marker(stage, "/World/DebugVis/SpawnMarker", (0.2, 0.4, 1.0))
        goal_marker = _make_marker(stage, "/World/DebugVis/GoalMarker", (1.0, 0.2, 0.2))

    def refresh_markers():
        if args.vis_goal:
            _update_marker(spawn_marker, env.robot_pos)
            _update_marker(goal_marker, env.goal_pos)

    refresh_markers()

    input_iface = carb.input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    keys_pressed: set = set()

    def on_key(event, *_):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            keys_pressed.add(event.input)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            keys_pressed.discard(event.input)
        return True

    input_iface.subscribe_to_keyboard_events(keyboard, on_key)
    print("[Teleop] W/S=前後  A/D=回転  P=座標表示  C=RGB観測をPNG保存  R=リセット  Q=終了")
    if args.num_humans > 0:
        print(f"[Teleop] 人物 {args.num_humans} 体, 接触センサーで衝突検知")

    snapshot_dir = Path(__file__).parent / "_snapshots"

    step = 0
    prev_human_collision = False
    collision_count = 0
    while app.is_running():
        if carb.input.KeyboardInput.Q in keys_pressed:
            print("\n[Teleop] 終了")
            break

        if carb.input.KeyboardInput.R in keys_pressed:
            env.reset()
            refresh_markers()
            keys_pressed.discard(carb.input.KeyboardInput.R)
            step = 0
            prev_human_collision = False
            print("\n[Teleop] リセット")

        if carb.input.KeyboardInput.P in keys_pressed:
            p = env.robot_pos
            print(f"\n[Pos] robot=({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})")
            for i, (hx, hy) in enumerate(env.human_positions_xy):
                print(f"      Human{i}=({hx:.2f}, {hy:.2f})")

        v_x = (
            1.0
            if carb.input.KeyboardInput.W in keys_pressed
            else -1.0
            if carb.input.KeyboardInput.S in keys_pressed
            else 0.0
        )
        omega = (
            1.0
            if carb.input.KeyboardInput.A in keys_pressed
            else -1.0
            if carb.input.KeyboardInput.D in keys_pressed
            else 0.0
        )

        obs, reward, terminated, truncated, info = env.step(
            np.array([v_x, omega], dtype=np.float32)
        )

        if carb.input.KeyboardInput.C in keys_pressed:
            keys_pressed.discard(carb.input.KeyboardInput.C)
            if "rgb" in obs:
                from PIL import Image

                # obs["rgb"]はモデル入力用にCHW・float32[0,1]へ正規化済み(RGBCameraTerm.compute()参照)
                # なので、PNG保存用にHWC・uint8[0,255]へ戻す
                rgb_hwc = (obs["rgb"].transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
                snapshot_dir.mkdir(exist_ok=True)
                out_path = snapshot_dir / f"step{step:05d}.png"
                Image.fromarray(rgb_hwc).save(out_path)
                print(f"\n[Teleop] RGB観測を保存しました: {out_path}")
            else:
                print("\n[Teleop] 観測にrgbキーがありません(observationsの設定を確認してください)")

        pos = env.robot_pos

        if args.num_humans > 0:
            dists = [
                float(np.hypot(hx - pos[0], hy - pos[1])) for hx, hy in env.human_positions_xy
            ]
            nearest_str = f"{min(dists):.2f}m" if dists else "N/A"

            human_collision = bool(info.get("human_collision", False))
            wall = bool(info.get("collision", False)) and not human_collision

            # 衝突検知の瞬間だけ警告を出す(以後は流れて消える通常ログのみ)
            if human_collision and not prev_human_collision:
                collision_count += 1
                print(
                    f"\n⚠️ : ロボットが人と衝突しました！ "
                    f"(#{collision_count}  step={step}  nearest_human={nearest_str})"
                )
            prev_human_collision = human_collision

            print(
                "\x1b[K"
                f"[step {step:5d}] "
                f"robot=({pos[0]:.2f},{pos[1]:.2f})  "
                f"nearest_human={nearest_str}  "
                f"HUMAN_COLLISION={'YES' if human_collision else 'no '}  "
                f"wall={'YES' if wall else 'no '}",
                end="\r",
            )
        else:
            goal = env.goal_pos
            dist_xy = float(np.linalg.norm(goal[[0, 1]] - pos[[0, 1]]))
            angle_rel_deg = float(env.goal_vec[1]) * 180.0
            w, qx, qy, qz = env.robot_quat
            yaw_deg = float(
                np.degrees(
                    np.arctan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))
                )
            )
            print(
                "\x1b[K"
                f"[step {step:4d}] "
                f"pos=({pos[0]:.2f},{pos[1]:.2f})  "
                f"yaw={yaw_deg:+.1f}deg  "
                f"angle_rel={angle_rel_deg:+.1f}deg  "
                f"dist_xy={dist_xy:.2f}m  "
                f"collision={int(info.get('collision', False))}",
                end="\r",
            )

        if info.get("human_locomotion_anomaly", False):
            print(f"\n[Teleop] 人物モーション異常検知によりエピソード終了(自動リセットは行いません): {info}")

        if (terminated or truncated) and args.reset:
            print()
            print(f"[Teleop] episode end — {info}")
            env.reset()
            refresh_markers()
            step = 0
        else:
            step += 1

    env.close()
    app.close()


if __name__ == "__main__":
    main()
