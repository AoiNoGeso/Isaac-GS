"""
Point Navigation テストスクリプト

実行方法:
  cd ~/Programs/Isaac-GS
  uv run models/sac/test.py --model runs/PointNav-RGB+Goal/0626/sac_final.pt --stage-index 0
  uv run models/sac/test.py --model path/to/sac_final.pt --stage-index 0 --num-humans 2
"""

import argparse
import sys
from pathlib import Path

from pydantic import BaseModel


class TestCfg(BaseModel):
    model_path: str = "checkpoints/sac_final.pt"
    episodes_per_stage: int = 100
    input_rgb: bool = True
    input_goal: bool = True


parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--model", type=str, default=None, help="チェックポイントパス")
parser.add_argument(
    "--stage-index", type=int, default=0, help="評価するステージのインデックス"
)
parser.add_argument("--num-humans", type=int, default=0)
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

from envs import PointNavGymEnv
from envs.config import EnvConfig, STAGE_PRESETS
from models.sac.config import ModelConfig, TrainConfig
from models.sac.network import make_encoder
from models.sac.policy import SACAgent
from utils.wandb_utils import EpisodeTracker

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    test_cfg = TestCfg()

    if args.model is not None:
        test_cfg.model_path = args.model

    stage_names = list(STAGE_PRESETS.keys())
    if args.stage_index >= len(stage_names):
        print(
            f"[test] Error: stage-index {args.stage_index} は範囲外です (stages={len(stage_names)})"
        )
        app.close()
        return

    stage_cfg = STAGE_PRESETS[stage_names[args.stage_index]]

    print(f"[test] Stage {args.stage_index}: {stage_cfg.stage_path}")
    print(f"[test] Model: {test_cfg.model_path}")
    print(f"[test] Episodes: {test_cfg.episodes_per_stage}")
    print(f"[test] num_humans: {args.num_humans}")

    model_cfg = ModelConfig(
        input_rgb=test_cfg.input_rgb,
        input_goal=test_cfg.input_goal,
    )
    env_cfg = EnvConfig(
        stage_path=stage_cfg.stage_path,
        fixed_spawn_pos=stage_cfg.fixed_spawn_pos,
        fixed_goal_pos=stage_cfg.fixed_goal_pos,
        fixed_spawn_yaw_deg=stage_cfg.fixed_spawn_yaw_deg,
        show_camera_viewport=not args.headless,
        num_humans=args.num_humans,
    )

    env = PointNavGymEnv(env_cfg=env_cfg)
    obs, _ = env.reset()

    action_dim = env.action_space.shape[0]

    agent = SACAgent(
        encoder_factory=lambda: make_encoder(model_cfg, img_size=env_cfg.camera_resolution[0]),
        action_dim=action_dim,
        cfg=TrainConfig(),
        device=DEVICE,
    )
    agent.load(test_cfg.model_path)

    successes = 0
    collisions = 0
    human_collisions = 0
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
        human_collision = bool(info.get("human_collision", False))
        timeout = bool(info.get("timeout", False))
        dist_final = float(info.get("dist", 0.0))
        spl = EpisodeTracker.compute_spl(success, init_dist, path_len)

        successes += int(success)
        collisions += int(collision)
        human_collisions += int(human_collision)
        timeouts += int(timeout)
        total_reward += ep_reward
        total_dist_final += dist_final
        total_spl += spl

        pbar.set_postfix(
            success=f"{successes}/{ep + 1}",
            collision=f"{collisions}/{ep + 1}",
        )

    n = test_cfg.episodes_per_stage
    tqdm.write(f"\n{'=' * 60}", file=_OUT)
    tqdm.write(f"Stage {args.stage_index}: {stage_cfg.stage_path}", file=_OUT)
    tqdm.write(f"{'=' * 60}", file=_OUT)
    tqdm.write(f"Episodes      : {n}", file=_OUT)
    tqdm.write(f"Success Rate  : {successes / n:.3f}  ({successes}/{n})", file=_OUT)
    tqdm.write(f"Collision Rate: {collisions / n:.3f}  ({collisions}/{n})", file=_OUT)
    tqdm.write(
        f"Human Collision Rate: {human_collisions / n:.3f}  ({human_collisions}/{n})",
        file=_OUT,
    )
    tqdm.write(f"Timeout Rate  : {timeouts / n:.3f}  ({timeouts}/{n})", file=_OUT)
    tqdm.write(f"Avg Reward    : {total_reward / n:.2f}", file=_OUT)
    tqdm.write(f"Avg Dist Final: {total_dist_final / n:.3f} m", file=_OUT)
    tqdm.write(f"SPL           : {total_spl / n:.3f}", file=_OUT)
    tqdm.write(f"{'=' * 60}\n", file=_OUT)

    env.close()
    app.close()


if __name__ == "__main__":
    main()
