"""
Point Navigation 学習スクリプト
IsaacSim 6.0 + 自前 SAC + wandb ログ

実行方法:
  cd ~/Programs/Isaac-GS
  uv run models/sac/train.py --headless
  uv run models/sac/train.py --headless --run-name my_run
  uv run models/sac/train.py --headless --checkpoint runs/point_nav/checkpoints/sac_10000.pt
  uv run models/sac/train.py --headless --num-humans 2
"""

import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--run-name", type=str, default=None)
parser.add_argument("--no-wandb", action="store_true", default=False)
parser.add_argument("--num-humans", type=int, default=None)
args = parser.parse_args()

from isaacsim import SimulationApp

app = SimulationApp({"headless": args.headless})

import omni.log

omni.log.get_log().set_channel_level(
    "omni.physx.plugin", omni.log.Level.ERROR, omni.log.SettingBehavior.OVERRIDE
)

import torch
import wandb
from tqdm import tqdm

# Isaac Sim が stderr を横取りするため stdout に固定
_OUT = sys.stdout

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from envs import PointNavGymEnv
from envs.config import EnvConfig
from models.sac.config import ModelConfig, TrainConfig
from models.sac.network import make_encoder
from models.sac.policy import ReplayBuffer, SACAgent
from utils.wandb_utils import EpisodeTracker

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    train_cfg = TrainConfig()
    if args.run_name is not None:
        train_cfg.run_name = args.run_name

    model_cfg = ModelConfig()
    env_cfg = EnvConfig()
    if args.num_humans is not None:
        env_cfg.num_humans = args.num_humans

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project="Isaac-GS_PointNav",
            name=train_cfg.run_name,
            config={
                "total_timesteps": train_cfg.total_timesteps,
                "input_rgb": model_cfg.input_rgb,
                "input_goal": model_cfg.input_goal,
                "num_humans": env_cfg.num_humans,
                **{
                    f"sac/{k}": v
                    for k, v in train_cfg.model_dump().items()
                    if k
                    in (
                        "buffer_size",
                        "batch_size",
                        "gamma",
                        "tau",
                        "learning_rate",
                        "learning_starts",
                        "train_freq",
                        "gradient_steps",
                        "target_entropy",
                    )
                },
            },
            dir=train_cfg.log_dir,
        )

    env = PointNavGymEnv(env_cfg=env_cfg)
    obs, _ = env.reset()

    obs_spec = {k: v.shape for k, v in env.observation_space.spaces.items()}
    action_dim = env.action_space.shape[0]

    buffer = ReplayBuffer(
        capacity=train_cfg.buffer_size,
        obs_spec=obs_spec,
        action_dim=action_dim,
        device=DEVICE,
    )

    agent = SACAgent(
        encoder_factory=lambda: make_encoder(model_cfg, img_size=env_cfg.camera_resolution[0]),
        action_dim=action_dim,
        cfg=train_cfg,
        device=DEVICE,
    )

    if args.checkpoint:
        agent.load(args.checkpoint)

    log_dir = train_cfg.log_dir
    ckpt_dir = f"{log_dir}/checkpoints"
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    tracker = EpisodeTracker(window=100)
    tracker.reset(obs)

    modality_str = (
        f"rgb={model_cfg.input_rgb}, goal={model_cfg.input_goal}, "
        f"humans={env_cfg.num_humans}"
    )
    print(
        f"[train] modality=({modality_str})  device={DEVICE}  total={train_cfg.total_timesteps:,}"
    )

    metrics: dict = {}
    pbar = tqdm(range(1, train_cfg.total_timesteps + 1), dynamic_ncols=True, file=_OUT)
    for step in pbar:
        if len(buffer) < train_cfg.learning_starts:
            action = env.action_space.sample()
        else:
            action = agent.act(obs, deterministic=False)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # timeout による truncation では done=0 とする
        buffer.add(obs, action, reward, next_obs, float(terminated))
        tracker.step(reward, info)
        obs = next_obs

        if done:
            ep_metrics = tracker.finish(info)
            tqdm.write(
                f"episode={ep_metrics['episode/count']:4d}"
                f"  reward={ep_metrics['episode/reward']:+7.1f}"
                f"  success={ep_metrics['episode/success_rate']:.2f}"
                f"  collision={ep_metrics['episode/collision_rate']:.2f}"
                f"  human_col={ep_metrics['episode/human_collision_rate']:.2f}",
                file=_OUT,
            )
            if use_wandb:
                wandb.log(ep_metrics, step=step)
            obs, reset_info = env.reset()
            tracker.reset(obs, reset_info)

        if (
            len(buffer) >= train_cfg.learning_starts
            and step % train_cfg.train_freq == 0
        ):
            metrics = agent.update(buffer)
            if use_wandb and step % train_cfg.log_interval == 0:
                wandb.log(metrics, step=step)

        if step % train_cfg.checkpoint_interval == 0:
            ckpt_path = f"{ckpt_dir}/sac_{step}.pt"
            agent.save(ckpt_path)
            tqdm.write(f"[train] Checkpoint saved: {ckpt_path}", file=_OUT)

    final_path = f"{log_dir}/sac_final.pt"
    agent.save(final_path)
    print(f"[train] Final model saved: {final_path}")

    if use_wandb:
        wandb.finish()
    env.close()
    app.close()


if __name__ == "__main__":
    main()
