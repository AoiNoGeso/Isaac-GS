"""
デバッグ用テレオペスクリプト

W/S: 前進/後退  A/D: 左回転/右回転  P: 座標表示  R: リセット  Q: 終了

--num-humans > 0 の場合は IRA アバターを注入する

実行:
  cd ~/Programs/Isaac-GS
  uv run debug/teleop.py
  uv run debug/teleop.py --num-humans 2
  uv run debug/teleop.py --stage corridor1
"""

import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--num-humans", type=int, default=0)
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument(
    "--stage", type=str, choices=["room1", "corridor1", "corridor2"], default="corridor2"
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

from isaacsim import SimulationApp

app = SimulationApp({
    "headless": args.headless,
    "extra_args": ["--/rtx/scenedb/maxHistoryTransformCount=256"],
})

import omni.log

omni.log.get_log().set_channel_level(
    "omni.physx.plugin", omni.log.Level.ERROR, omni.log.SettingBehavior.OVERRIDE
)

import carb
import numpy as np
import omni.appwindow
import omni.usd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs import PointNavIsaacEnv
from envs.config import EnvConfig, STAGE_PRESETS

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
    preset = STAGE_PRESETS[args.stage]
    env_cfg = EnvConfig(
        num_humans=args.num_humans,
        human_speed_range=(0.8, 1.5),
        stage_path=preset.stage_path,
        fixed_spawn_pos=preset.fixed_spawn_pos,
        fixed_goal_pos=preset.fixed_goal_pos,
        fixed_spawn_yaw_deg=preset.fixed_spawn_yaw_deg,
    )
    env = PointNavIsaacEnv(env_cfg)
    env.reset()

    spawn_marker = goal_marker = None
    if args.vis_goal:
        stage = omni.usd.get_context().get_stage()
        spawn_marker = _make_marker(stage, "/World/DebugVis/SpawnMarker", (0.2, 0.4, 1.0))
        goal_marker = _make_marker(stage, "/World/DebugVis/GoalMarker", (1.0, 0.2, 0.2))
        _update_marker(spawn_marker, env._get_robot_pos())
        _update_marker(goal_marker, env._goal_pos)

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
    print("[Teleop] W/S=前後  A/D=回転  P=座標表示  R=リセット  Q=終了")
    if args.num_humans > 0:
        print(f"[Teleop] 人物 {len(env._ira_characters)} 体, 接触センサーで衝突検知")

    step = 0
    prev_hc = False
    collision_count = 0
    while app.is_running():
        if carb.input.KeyboardInput.Q in keys_pressed:
            print("\n[Teleop] 終了")
            break

        if carb.input.KeyboardInput.R in keys_pressed:
            env.reset()
            if args.vis_goal:
                _update_marker(spawn_marker, env._get_robot_pos())
                _update_marker(goal_marker, env._goal_pos)
            keys_pressed.discard(carb.input.KeyboardInput.R)
            step = 0
            prev_hc = False
            print("\n[Teleop] リセット")

        if carb.input.KeyboardInput.P in keys_pressed:
            p = env._get_robot_pos()
            print(f"\n[Pos] robot=({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})")
            for a in env._ira_characters:
                hp = a.get_world_position()
                if hp is not None:
                    print(f"      {a.name}=({hp.x:.2f}, {hp.y:.2f}, {hp.z:.2f})")

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

        pos = env._get_robot_pos()

        if args.num_humans > 0:
            nearest = None
            for a in env._ira_characters:
                hp = a.get_world_position()
                if hp is None:
                    continue
                d = float(np.hypot(hp.x - pos[0], hp.y - pos[1]))
                if nearest is None or d < nearest:
                    nearest = d
            nearest_str = f"{nearest:.2f}m" if nearest is not None else "N/A"

            hc = bool(info.get("human_collision", False))
            wall = bool(info.get("collision", False)) and not hc

            # 衝突検知の瞬間 (False→True) に消えない警告を残す
            if hc and not prev_hc:
                collision_count += 1
                print(
                    f"\n⚠️ : ロボットが人と衝突しました！ "
                    f"(#{collision_count}  step={step}  nearest_human={nearest_str})"
                )
            prev_hc = hc

            print(
                "\x1b[K"
                f"[step {step:5d}] "
                f"robot=({pos[0]:.2f},{pos[1]:.2f})  "
                f"nearest_human={nearest_str}  "
                f"HUMAN_COLLISION={'YES' if hc else 'no '}  "
                f"wall={'YES' if wall else 'no '}",
                end="\r",
            )
        else:
            goal = env._goal_pos
            dist_xy = float(np.linalg.norm(goal[[0, 1]] - pos[[0, 1]]))
            goal_vec = env._compute_goal_vec()
            angle_rel_deg = float(goal_vec[1]) * 180.0
            w, qx, qy, qz = env._get_robot_quat()
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

        if (terminated or truncated) and args.reset:
            print()
            print(f"[Teleop] episode end — {info}")
            env.reset()
            if args.vis_goal:
                _update_marker(spawn_marker, env._get_robot_pos())
                _update_marker(goal_marker, env._goal_pos)
            step = 0
        else:
            step += 1

    env.close()
    app.close()


if __name__ == "__main__":
    main()
