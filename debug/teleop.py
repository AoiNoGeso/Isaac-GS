"""
WASD テレオペスクリプト — 衝突判定デバッグ用

W/S: 前進/後退  A/D: 左回転/右回転  P: 座標表示  Q: 終了

実行:
  cd ~/Programs/Isaac-GS
  uv run debug/teleop.py
"""

import sys
from pathlib import Path

from isaacsim import SimulationApp
app = SimulationApp({"headless": False})

import omni.log
omni.log.get_log().set_channel_level(
    "omni.physx.plugin", omni.log.Level.ERROR, omni.log.SettingBehavior.OVERRIDE
)

import carb
import numpy as np
import omni.appwindow

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.isaac_env import PointNavIsaacEnv, _V_LINEAR_MAX, _V_ANGULAR_MAX
from tasks.point_navigation.config import PointNavEnvCfg


def main():
    cfg = PointNavEnvCfg()
    cfg.fixed_spawn_pos = (0.4, 1.4, -1.0)
    cfg.fixed_goal_pos = (-0.3, -3.4, -0.8)
    env = PointNavIsaacEnv(cfg)
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
    print("[Teleop] W/S=前後  A/D=回転  P=座標表示  Q=終了")

    step = 0
    while app.is_running():
        if carb.input.KeyboardInput.Q in keys_pressed:
            print("\n[Teleop] 終了")
            break

        if carb.input.KeyboardInput.P in keys_pressed:
            p = env._get_robot_pos()
            print(f"\n[Pos] ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})")

        v_x = (1.0 if carb.input.KeyboardInput.W in keys_pressed
                else -1.0 if carb.input.KeyboardInput.S in keys_pressed else 0.0) * _V_LINEAR_MAX
        omega = (1.0 if carb.input.KeyboardInput.A in keys_pressed
                 else -1.0 if carb.input.KeyboardInput.D in keys_pressed else 0.0) * _V_ANGULAR_MAX

        obs, reward, terminated, truncated, info = env.step(
            np.array([v_x / _V_LINEAR_MAX, omega / _V_ANGULAR_MAX], dtype=np.float32)
        )

        pos = env._get_robot_pos()
        goal = env._goal_pos
        dist_xy = float(np.linalg.norm(goal[[0, 1]] - pos[[0, 1]]))
        print(
            f"[step {step:4d}] "
            f"pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})  "
            f"goal=({goal[0]:.2f},{goal[1]:.2f},{goal[2]:.2f})  "
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

    app.close()


if __name__ == "__main__":
    main()
