"""
Point Navigation 学習スクリプト
IsaacSim 6.0 + skrl 2.x PPO + CNN policy + wandb ログ

実行方法:
  cd ~/Programs/Isaac-GS
  uv run tasks/point_navigation/train.py --headless
  uv run tasks/point_navigation/train.py --headless --run-name my_run
  uv run tasks/point_navigation/train.py --headless --checkpoint runs/point_nav/checkpoints/best.pt
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--run-name", type=str, default=None, help="wandb run name")
parser.add_argument("--no-wandb", action="store_true", default=False)
args = parser.parse_args()

from isaacsim import SimulationApp
app = SimulationApp({"headless": args.headless})

import numpy as np
import torch
import gymnasium as gym
import wandb
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

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 84
LOG_DIR  = "runs/point_nav"


class WandbEpisodeLogger(gym.Wrapper):
    """エピソード終了時に wandb へメトリクスを送る（1エピソード = 1ログ）．"""

    def __init__(self, env, window: int = 100):
        super().__init__(env)
        self._ep_reward      = 0.0
        self._ep_count       = 0
        self._total_steps    = 0
        self._ep_initial_dist = 0.0
        self._ep_path_len    = 0.0
        self._prev_xz        = None
        self._success_buf    = []
        self._collision_buf  = []
        self._window         = window

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._ep_reward = 0.0
        self._ep_path_len = 0.0
        self._prev_xz = None
        # goal_vec[0] = clip(dist/10, 0, 1) → 初期距離を復元（最大10mでクリップ）
        if isinstance(obs, dict):
            self._ep_initial_dist = float(obs["goal"][0]) * 10.0
        else:
            self._ep_initial_dist = float(obs[0]) * 10.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._ep_reward   += float(reward)
        self._total_steps += 1

        # 移動経路長の積算
        xz = info.get("robot_xz")
        if xz is not None:
            xz = np.array(xz, dtype=np.float32)
            if self._prev_xz is not None:
                self._ep_path_len += float(np.linalg.norm(xz - self._prev_xz))
            self._prev_xz = xz

        if terminated or truncated:
            self._ep_count += 1

            success   = bool(info.get("success",   False))
            collision = bool(info.get("collision", False))
            timeout   = bool(truncated and not terminated)

            self._success_buf.append(float(success))
            self._collision_buf.append(float(collision))
            if len(self._success_buf)  > self._window:
                self._success_buf.pop(0)
            if len(self._collision_buf) > self._window:
                self._collision_buf.pop(0)

            # SPL = success × (l / max(p, l))
            l = self._ep_initial_dist
            p = self._ep_path_len
            spl = float(success) * (l / max(p, l)) if l > 0 else 0.0

            wandb.log({
                "episode/reward":         self._ep_reward,
                "episode/success":        float(success),
                "episode/collision":      float(collision),
                "episode/timeout":        float(timeout),
                "episode/spl":            spl,
                "episode/dist_final":     float(info.get("dist", 0.0)),
                "episode/success_rate":   sum(self._success_buf)   / len(self._success_buf),
                "episode/collision_rate": sum(self._collision_buf) / len(self._collision_buf),
                "episode/count":          self._ep_count,
            }, step=self._total_steps)

        return obs, reward, terminated, truncated, info


def main():
    use_wandb = not args.no_wandb

    # ── wandb 初期化 ──────────────────────────────────────────────────────────
    if use_wandb:
        wandb.init(
            project="isaac-gs-point-nav",
            name=args.run_name,
            config={
                "algorithm":      "PPO",
                "encoder":        "CNN-scratch",
                "obs":            ["rgb(3,84,84)", "goal(2,)"],
                "action":         "[v_x_norm, omega_z_norm]",
                "rollouts":       512,
                "learning_epochs": 5,
                "mini_batches":   4,
                "gamma":          0.99,
                "gae_lambda":     0.95,
                "lr":             3e-4,
                "total_timesteps": 500_000,
                "device":         DEVICE,
            },
            dir=LOG_DIR,
        )

    # ── 環境 ──────────────────────────────────────────────────────────────────
    cfg     = PointNavEnvCfg()
    gym_env = PointNavGymEnv(cfg=cfg)
    if use_wandb:
        gym_env = WandbEpisodeLogger(gym_env)
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

    # ── PPO 設定（skrl の loss ログは wandb run に同乗させる）────────────────
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
        experiment={
            "directory":           LOG_DIR,
            "experiment_name":     args.run_name or "",
            "write_interval":      1000,
            "checkpoint_interval": 10_000,
            "wandb":               False,  # 手動 wandb.init() と2重にならないよう無効化
        },
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

    if use_wandb:
        wandb.finish()

    gym_env.close()
    app.close()


if __name__ == "__main__":
    main()
