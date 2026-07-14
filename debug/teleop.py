"""
WASD テレオペスクリプト（衝突判定デバッグ用）

W/S: 前進/後退  A/D: 左回転/右回転  P: 座標表示  R: リセット  Q: 終了

--num-humans > 0 の場合は IRA 人物キャラを注入し、ロボット-人物の衝突判定
（IRA 純正 AvoidanceHandler イベント）を確認できる。衝突しても自動リセットしない。

実行:
  cd ~/Programs/Isaac-GS
  uv run debug/teleop.py
  uv run debug/teleop.py --num-humans 2
  uv run debug/teleop.py --stage corridor1_2d
"""

import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--num-humans", type=int, default=0)
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument(
    "--stage", type=str, choices=["room1", "corridor1_2d"], default="room1"
)
args = parser.parse_args()

from isaacsim import SimulationApp

app = SimulationApp({"headless": args.headless})

import omni.log

omni.log.get_log().set_channel_level(
    "omni.physx.plugin", omni.log.Level.ERROR, omni.log.SettingBehavior.OVERRIDE
)

import carb
import numpy as np
import omni.appwindow

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.isaac_env import _V_ANGULAR_MAX, _V_LINEAR_MAX, PointNavIsaacEnv
from tasks.point_navigation.config import EnvConfig, ModelConfig

_STAGE_PRESETS = {
    "room1": dict(
        stage_path="sample_data/stages/room1/stage.usda",
        fixed_spawn_pos=(0.9, -0.19, -2.6),
        fixed_goal_pos=(-3.0, 1.6, -2.6),
        fixed_spawn_yaw_deg=137,
    ),
    "corridor1_2d": dict(
        stage_path="sample_data/stages/corridor1_2d/stage.usda",
        fixed_spawn_pos=(0.4, 1.4, -1.0),
        fixed_goal_pos=(-0.1, -1.3, -0.8),
        fixed_spawn_yaw_deg=-90,
    ),
}


def main():
    preset = _STAGE_PRESETS[args.stage]
    env_cfg = EnvConfig(
        num_humans=args.num_humans,
        # 人物ありの場合はゆっくり歩かせ、毎エピソード再配置しない（狙って接近しやすくする）
        human_speed_range=(0.6, 0.6) if args.num_humans > 0 else (0.8, 1.5),
        reset_humans_each_episode=args.num_humans <= 0,
        **preset,
    )
    model_cfg = ModelConfig()
    env = PointNavIsaacEnv(env_cfg, model_cfg)
    env.reset()

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
        print(f"[Teleop] 人物 {len(env._ira_characters)} 体。IRA AvoidanceHandler イベントで衝突検知。")

    step = 0
    prev_hc = False
    collision_count = 0
    while app.is_running():
        if carb.input.KeyboardInput.Q in keys_pressed:
            print("\n[Teleop] 終了")
            break

        if carb.input.KeyboardInput.R in keys_pressed:
            env.reset()
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

            # 衝突を「検知した瞬間（False→True）」に消えない警告を残す
            if hc and not prev_hc:
                collision_count += 1
                print(
                    f"\n⚠️ : ロボットが人と衝突しました！ "
                    f"(#{collision_count}  step={step}  nearest_human={nearest_str})"
                )
            prev_hc = hc

            print(
                f"[step {step:5d}] "
                f"robot=({pos[0]:.2f},{pos[1]:.2f})  "
                f"nearest_human={nearest_str}  "
                f"HUMAN_COLLISION={'YES' if hc else 'no '}  "
                f"wall={'YES' if wall else 'no '}",
                end="\r",
            )
            # 衝突しても自動リセットしない（押し当てて観察できるように）。R で手動リセット。
            step += 1
        else:
            print(
                f"[step {step:4d}] "
                f"pos=({pos[0]:.2f},{pos[1]:.2f})  "
                f"yaw={yaw_deg:+.1f}deg  "
                f"angle_rel={angle_rel_deg:+.1f}deg  "
                f"dist_xy={dist_xy:.2f}m  "
                f"collision={int(info.get('collision', False))}",
                end="\r",
            )

            if terminated or truncated:
                print()
                print(f"[Teleop] episode end — {info}")
                env.reset()
                step = 0
            else:
                step += 1

    env.close()
    app.close()


if __name__ == "__main__":
    main()
