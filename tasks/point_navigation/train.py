"""
Point Navigation 学習スクリプト
IsaacSim 6.0 + skrl 2.x PPO + CNN policy

実行方法:
  cd ~/Programs/Isaac-GS
  uv run tasks/point_navigation/train.py
  uv run tasks/point_navigation/train.py --headless
  uv run tasks/point_navigation/train.py --headless --checkpoint runs/point_nav/checkpoints/best.pt
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--checkpoint", type=str, default=None)
args = parser.parse_args()

from isaacsim import SimulationApp
app = SimulationApp({"headless": args.headless})

import torch
sys.path.insert(0, os.path.expanduser("~/Programs/Isaac-GS"))

from tasks.point_navigation.env.isaac_env import PointNavEnvCfg
from tasks.point_navigation.env.gym_wrapper import PointNavGymEnv
from tasks.point_navigation.policy.cnn_encoder import PointNavActor, PointNavCritic

from skrl.agents.torch.ppo import PPO
from skrl.agents.torch.ppo.ppo_cfg import PPO_CFG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.trainers.torch.sequential import SequentialTrainerCfg
from skrl.envs.wrappers.torch import wrap_env
from skrl.utils import set_seed

set_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 128


def main():
    # ── 環境 ──────────────────────────────────────────────────────────────────
    cfg = PointNavEnvCfg()
    gym_env = PointNavGymEnv(cfg=cfg)
    gym_env.reset()
    env = wrap_env(gym_env)

    # ── モデル ────────────────────────────────────────────────────────────────
    models = {
        "policy": PointNavActor(
            observation_space=gym_env.observation_space,
            action_space=gym_env.action_space,
            device=DEVICE,
            img_size=IMG_SIZE,
        ),
        "value": PointNavCritic(
            observation_space=gym_env.observation_space,
            action_space=gym_env.action_space,
            device=DEVICE,
            img_size=IMG_SIZE,
        ),
    }

    if args.checkpoint:
        models["policy"].load(args.checkpoint)
        print(f"[train] Loaded checkpoint: {args.checkpoint}")

    # ── メモリ ────────────────────────────────────────────────────────────────
    memory = RandomMemory(memory_size=512, num_envs=1, device=DEVICE)

    # ── PPO 設定 ──────────────────────────────────────────────────────────────
    ppo_cfg = PPO_CFG(
        rollouts=512,
        learning_epochs=5,
        mini_batches=4,
        discount_factor=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        grad_norm_clip=1.0,
        ratio_clip=0.2,
        value_loss_scale=1.0,
        entropy_loss_scale=0.005,
        kl_threshold=0.01,
    )

    agent = PPO(
        models=models,
        memory=memory,
        cfg=ppo_cfg,
        observation_space=gym_env.observation_space,
        action_space=gym_env.action_space,
        device=DEVICE,
    )

    # ── 学習 ──────────────────────────────────────────────────────────────────
    trainer = SequentialTrainer(
        cfg=SequentialTrainerCfg(
            timesteps=500_000,
            headless=args.headless,
        ),
        env=env,
        agents=agent,
    )

    print("[train] Start training...")
    trainer.train()

    gym_env.close()
    app.close()


if __name__ == "__main__":
    main()
