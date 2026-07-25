"""Point Navigation テストスクリプト。実行例: uv run models/sac/test.py --model path/to/sac_final.pt --stage-index 0 [--num-humans N]"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--model", type=str, required=True, help="チェックポイントパス")
parser.add_argument(
    "--stage-index", type=int, default=0, help="評価するステージのインデックス"
)
parser.add_argument("--num-humans", type=int, default=0)
parser.add_argument(
    "--video", action="store_true", default=False, help="各エピソードの映像をmp4で保存する"
)
parser.add_argument(
    "--video-dir", type=str, default="videos", help="動画の出力先ディレクトリ (default: videos)"
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

import cv2
import numpy as np
import torch
from tqdm import tqdm

_OUT = sys.stdout

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from envs import PointNavGymEnv
from envs.config import STAGE_PRESETS, EnvConfig, stage_names as _stage_names
from models.sac.config import ModelConfig, TestConfig, TrainConfig
from models.sac.network import make_encoder
from models.sac.policy import SACAgent
from utils.metrics import EpisodeTracker
from utils.video import make_overhead_camera, write_frame as _write_frame, write_overhead_frame as _write_overhead_frame

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    """指定ステージでモデルを評価し、成功率等の指標を表示する"""
    test_cfg = TestConfig()

    stage_names = _stage_names()
    if args.stage_index >= len(stage_names):
        print(
            f"[test] Error: stage-index {args.stage_index} は範囲外です (stages={len(stage_names)})"
        )
        app.close()
        return

    stage_cfg = STAGE_PRESETS[stage_names[args.stage_index]]

    print(f"[test] Stage {args.stage_index}: {stage_cfg.stage_path}")
    print(f"[test] Model: {args.model}")
    print(f"[test] Episodes: {test_cfg.episodes_per_stage}")
    print(f"[test] num_humans: {args.num_humans}")

    model_cfg = ModelConfig()
    env_cfg = EnvConfig.from_preset(
        stage_names[args.stage_index],
        show_camera_viewport=not args.headless,
        num_humans=args.num_humans,
    )

    env = PointNavGymEnv(env_cfg=env_cfg)
    obs, _ = env.reset()

    action_dim = env.action_space.shape[0]

    agent = SACAgent(
        encoder_factory=lambda: make_encoder(model_cfg, img_size=env_cfg.camera_resolution[0]),
        action_dim=action_dim,
        cfg=TrainConfig(stage=stage_names[args.stage_index]),
        device=DEVICE,
    )
    agent.load(args.model)

    video_dir = None
    if args.video:
        timestamp = datetime.now().strftime("%m%d%H%M%S")
        video_dir = Path(args.video_dir) / stage_names[args.stage_index] / timestamp
        video_dir.mkdir(parents=True, exist_ok=True)
        video_fps = 1.0 / env_cfg.rendering_dt
        print(f"[test] video: enabled -> {video_dir}/")

    overhead_camera = None
    if args.video and not args.headless:
        overhead_camera = make_overhead_camera(env_cfg, stage_cfg)
        if overhead_camera is not None:
            print("[test] overhead camera: enabled")

    successes = 0
    collisions = 0
    wall_collisions = 0
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

        video_writer = None
        video_path = None
        overhead_writer = None
        overhead_path = None
        if video_dir is not None:
            H, W = env_cfg.camera_resolution[1], env_cfg.camera_resolution[0]
            video_path = video_dir / f"ep{ep:04d}.mp4"
            video_writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (W, H)
            )
            _write_frame(video_writer, obs["rgb"])

            if overhead_camera is not None:
                overhead_path = video_dir / f"ep{ep:04d}_overhead.mp4"
                overhead_writer = cv2.VideoWriter(
                    str(overhead_path), cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (W, H)
                )
                overhead_rgb, _ = overhead_camera.get_rgbd()
                _write_overhead_frame(overhead_writer, overhead_rgb)

        while True:
            action = agent.act(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            if video_writer is not None:
                _write_frame(video_writer, obs["rgb"])
            if overhead_writer is not None:
                overhead_rgb, _ = overhead_camera.get_rgbd()
                _write_overhead_frame(overhead_writer, overhead_rgb)

            xy = info.get("robot_xz")
            if xy is not None and prev_xy is not None:
                path_len += float(np.linalg.norm(np.array(xy) - np.array(prev_xy)))
            prev_xy = xy

            if terminated or truncated:
                break

        success, wall_collision, human_collision, timeout = EpisodeTracker.derive_outcome(info)
        dist_final = float(info.get("dist", 0.0))
        spl = EpisodeTracker.compute_spl(success, init_dist, path_len)

        if video_writer is not None:
            video_writer.release()
            if success:
                suffix = "_s"
            elif human_collision:
                suffix = "_h"
            elif wall_collision:
                suffix = "_w"
            else:
                suffix = "_t"
            video_path.rename(video_path.with_stem(video_path.stem + suffix))
            if overhead_writer is not None:
                overhead_writer.release()
                overhead_path.rename(overhead_path.with_stem(overhead_path.stem + suffix))

        successes += int(success)
        collisions += int(wall_collision or human_collision)
        wall_collisions += int(wall_collision)
        human_collisions += int(human_collision)
        timeouts += int(timeout)
        total_reward += ep_reward
        total_dist_final += dist_final
        total_spl += spl

        pbar.set_postfix(
            success=f"{successes}/{ep + 1}",
            wall=f"{wall_collisions}/{ep + 1}",
            human=f"{human_collisions}/{ep + 1}",
        )

    n = test_cfg.episodes_per_stage
    tqdm.write(f"\n{'=' * 60}", file=_OUT)
    tqdm.write(f"Stage {args.stage_index}: {stage_cfg.stage_path}", file=_OUT)
    tqdm.write(f"{'=' * 60}", file=_OUT)
    tqdm.write(f"Episodes      : {n}", file=_OUT)
    tqdm.write(f"Success Rate  : {successes / n:.3f}  ({successes}/{n})", file=_OUT)
    tqdm.write(f"Collision Rate: {collisions / n:.3f}  ({collisions}/{n})", file=_OUT)
    tqdm.write(
        f"Wall Collision Rate : {wall_collisions / n:.3f}  ({wall_collisions}/{n})",
        file=_OUT,
    )
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
