"""
Point Navigation テストスクリプト
IsaacSim 6.0 + 学習済み SAC モデルで deterministic 評価を行う

実行方法:
  cd ~/Programs/Isaac-GS
  uv run tasks/point_navigation/test.py --model runs/PointNav-RGB+Goal/0626/sac_final.pt --stage-index 0 --headless

複数ステージを評価する場合はステージ数分だけ別プロセスで実行する:
  for i in 0 1 2; do
    uv run tasks/point_navigation/test.py --model path/to/sac_final.pt --stage-index $i --headless
  done
"""

import argparse
import sys
from pathlib import Path

from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────────
# テスト設定
# ─────────────────────────────────────────────────────────────────────────────


class TestStageCfg(BaseModel):
    stage_path: str
    fixed_spawn_pos: tuple[float, float, float]
    fixed_goal_pos: tuple[float, float, float]
    fixed_spawn_yaw_deg: float | None = None  # None でランダム


class TestCfg(BaseModel):
    model_path: str = "checkpoints/sac_final.pt"
    episodes_per_stage: int = 100
    input_rgb: bool = True
    input_goal: bool = True
    stages: list[TestStageCfg] = [
        # TestStageCfg(
        #     stage_path="sample_data/stages/corridor1_2d/stage.usda",
        #     fixed_spawn_pos=(0.4, 1.4, -1.0),
        #     fixed_goal_pos=(-0.1, -1.3, -0.8),
        #     fixed_spawn_yaw_deg=-90.0,
        # ),
        TestStageCfg(
            stage_path="sample_data/stages/room1/stage.usda",
            fixed_spawn_pos=(0.9, -0.19, -2.6),
            fixed_goal_pos=(-3.0, 1.6, -2.6),
            fixed_spawn_yaw_deg=137.0,
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# CLI 引数（SimulationApp より前に parse する）
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--model", type=str, default=None, help="チェックポイントパス")
parser.add_argument(
    "--stage-index", type=int, default=0, help="評価するステージのインデックス"
)
args = parser.parse_args()

from isaacsim import SimulationApp

app = SimulationApp({"headless": args.headless})

import omni.log

omni.log.get_log().set_channel_level(
    "omni.physx.plugin", omni.log.Level.ERROR, omni.log.SettingBehavior.OVERRIDE
)

import numpy as np
import torch
from tqdm import tqdm

_OUT = sys.stdout

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from envs.gym_wrapper import PointNavGymEnv
from tasks.point_navigation.config import PointNavEnvCfg, SACCfg
from tasks.point_navigation.policy.network import PointNavEncoder
from tasks.point_navigation.policy.policy import SACAgent

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────────────────────────


def main():
    test_cfg = TestCfg()

    if args.model is not None:
        test_cfg.model_path = args.model

    if args.stage_index >= len(test_cfg.stages):
        print(
            f"[test] Error: stage-index {args.stage_index} は範囲外です (stages={len(test_cfg.stages)})"
        )
        app.close()
        return

    stage_cfg = test_cfg.stages[args.stage_index]

    print(f"[test] Stage {args.stage_index}: {stage_cfg.stage_path}")
    print(f"[test] Model: {test_cfg.model_path}")
    print(f"[test] Episodes: {test_cfg.episodes_per_stage}")

    # ── 環境構築 ─────────────────────────────────────────────────────────────
    env_cfg = PointNavEnvCfg(
        stage_path=stage_cfg.stage_path,
        input_rgb=test_cfg.input_rgb,
        input_goal=test_cfg.input_goal,
        fixed_spawn_pos=stage_cfg.fixed_spawn_pos,
        fixed_goal_pos=stage_cfg.fixed_goal_pos,
        fixed_spawn_yaw_deg=stage_cfg.fixed_spawn_yaw_deg,
        show_camera_viewport=not args.headless,
    )

    env = PointNavGymEnv(cfg=env_cfg)
    obs, _ = env.reset()

    action_dim = env.action_space.shape[0]
    img_size = env_cfg.camera_resolution[0]

    # ── エージェント構築・ロード ───────────────────────────────────────────────
    def encoder_factory():
        return PointNavEncoder(
            input_rgb=test_cfg.input_rgb,
            input_goal=test_cfg.input_goal,
            img_size=img_size,
        )

    agent = SACAgent(
        encoder_factory=encoder_factory,
        action_dim=action_dim,
        cfg=SACCfg(),
        device=DEVICE,
    )
    agent.load(test_cfg.model_path)

    # ── 評価ループ ────────────────────────────────────────────────────────────
    successes = 0
    collisions = 0
    timeouts = 0
    total_reward = 0.0
    total_dist_final = 0.0
    total_spl = 0.0

    pbar = tqdm(range(test_cfg.episodes_per_stage), dynamic_ncols=True, file=_OUT)
    for ep in pbar:
        obs, _ = env.reset()
        init_dist = float(obs["goal"][0]) if "goal" in obs else 0.0
        ep_reward = 0.0
        path_len = 0.0
        prev_xy = None

        while True:
            action = agent.act(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            xy = info.get("robot_xz")
            if xy is not None and prev_xy is not None:
                path_len += float(np.linalg.norm(np.array(xy) - np.array(prev_xy)))
            prev_xy = xy

            if terminated or truncated:
                break

        success = bool(info.get("success", False))
        collision = bool(info.get("collision", False))
        timeout = bool(info.get("timeout", False))
        dist_final = float(info.get("dist", 0.0))
        spl = (
            float(success) * (init_dist / max(path_len, init_dist))
            if init_dist > 0
            else 0.0
        )

        successes += int(success)
        collisions += int(collision)
        timeouts += int(timeout)
        total_reward += ep_reward
        total_dist_final += dist_final
        total_spl += spl

        pbar.set_postfix(
            success=f"{successes}/{ep + 1}",
            collision=f"{collisions}/{ep + 1}",
        )

    # ── 結果表示 ──────────────────────────────────────────────────────────────
    n = test_cfg.episodes_per_stage
    tqdm.write(f"\n{'=' * 60}", file=_OUT)
    tqdm.write(f"Stage {args.stage_index}: {stage_cfg.stage_path}", file=_OUT)
    tqdm.write(f"{'=' * 60}", file=_OUT)
    tqdm.write(f"Episodes      : {n}", file=_OUT)
    tqdm.write(f"Success Rate  : {successes / n:.3f}  ({successes}/{n})", file=_OUT)
    tqdm.write(f"Collision Rate: {collisions / n:.3f}  ({collisions}/{n})", file=_OUT)
    tqdm.write(f"Timeout Rate  : {timeouts / n:.3f}  ({timeouts}/{n})", file=_OUT)
    tqdm.write(f"Avg Reward    : {total_reward / n:.2f}", file=_OUT)
    tqdm.write(f"Avg Dist Final: {total_dist_final / n:.3f} m", file=_OUT)
    tqdm.write(f"SPL           : {total_spl / n:.3f}", file=_OUT)
    tqdm.write(f"{'=' * 60}\n", file=_OUT)

    env.close()
    app.close()


if __name__ == "__main__":
    main()
