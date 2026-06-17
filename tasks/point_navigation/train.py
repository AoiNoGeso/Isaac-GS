"""
Point Navigation 学習スクリプト
IsaacSim 6.0 + skrl 2.x + CNN policy + wandb ログ

実行方法:
  cd ~/Programs/Isaac-GS
  uv run tasks/point_navigation/train.py --headless
  uv run tasks/point_navigation/train.py --headless --algo sac
  uv run tasks/point_navigation/train.py --headless --run-name my_run
  uv run tasks/point_navigation/train.py --headless --checkpoint runs/point_nav/checkpoints/best.pt
"""

import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument(
    "--algo",
    type=str,
    default=None,
    choices=["ppo", "sac"],
    help="アルゴリズム選択(PPO / SAC)",
)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--run-name", type=str, default=None)
parser.add_argument("--no-wandb", action="store_true", default=False)
args = parser.parse_args()

from isaacsim import SimulationApp

app = SimulationApp({"headless": args.headless})

import omni.log

omni.log.get_log().set_channel_level(
    "omni.physx.plugin", omni.log.Level.ERROR, omni.log.SettingBehavior.OVERRIDE
)

import gymnasium as gym
import numpy as np
import torch
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from skrl.agents.torch.ppo import PPO
from skrl.agents.torch.ppo.ppo_cfg import PPO_CFG
from skrl.agents.torch.sac import SAC
from skrl.agents.torch.sac.sac_cfg import SAC_CFG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.trainers.torch.sequential import SequentialTrainerCfg

from envs.gym_wrapper import PointNavGymEnv
from tasks.point_navigation.config import PointNavEnvCfg, PointNavTrainCfg
from tasks.point_navigation.policy.network import (
    PointNavActor,
    PointNavCritic,
    SACPointNavActor,
    SACPointNavCritic,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 84
LOG_DIR = "runs/point_nav"


def _grad_norm(param) -> float | None:
    if param is not None and param.grad is not None:
        return float(param.grad.norm().item())
    return None


class WandbEpisodeLogger(gym.Wrapper):
    def __init__(self, env, window: int = 100, models: dict | None = None):
        super().__init__(env)
        self._ep_reward = 0.0
        self._ep_count = 0
        self._total_steps = 0
        self._ep_initial_dist = 0.0
        self._ep_path_len = 0.0
        self._prev_xy = None
        self._success_buf = []
        self._collision_buf = []
        self._window = window
        self._models = models or {}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._ep_reward = 0.0
        self._ep_path_len = 0.0
        self._prev_xy = None
        if isinstance(obs, dict):
            self._ep_initial_dist = float(obs["goal"][0]) * 10.0
        else:
            self._ep_initial_dist = float(obs[0]) * 10.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._ep_reward += float(reward)
        self._total_steps += 1

        xy = info.get("robot_xz")
        if xy is not None:
            xy = np.array(xy, dtype=np.float32)
            if self._prev_xy is not None:
                self._ep_path_len += float(np.linalg.norm(xy - self._prev_xy))
            self._prev_xy = xy

        log = {}
        policy = self._models.get("policy")
        if policy is not None:
            try:
                entropy = policy.get_entropy()
                if entropy is not None:
                    log["train/entropy"] = float(entropy.mean())
            except Exception:
                pass
            for name, attr in [("mean_layer", "mean_layer"), ("log_std_layer", "log_std_layer"),
                                ("log_std_param", "log_std_param")]:
                layer = getattr(policy, attr, None)
                if layer is None:
                    continue
                param = layer.weight if hasattr(layer, "weight") else layer
                g = _grad_norm(param)
                if g is not None:
                    log[f"grad/policy/{name}"] = g

        for role in ("critic_1", "critic_2", "value"):
            m = self._models.get(role)
            if m is None:
                continue
            last = None
            for mod in m.modules():
                if isinstance(mod, torch.nn.Linear):
                    last = mod
            if last is not None:
                g = _grad_norm(last.weight)
                if g is not None:
                    log[f"grad/{role}/output_layer"] = g

        if log:
            wandb.log(log, step=self._total_steps)

        if terminated or truncated:
            self._ep_count += 1
            success = bool(info.get("success", False))
            collision = bool(info.get("collision", False))
            timeout = bool(truncated and not terminated)

            self._success_buf.append(float(success))
            self._collision_buf.append(float(collision))
            if len(self._success_buf) > self._window:
                self._success_buf.pop(0)
            if len(self._collision_buf) > self._window:
                self._collision_buf.pop(0)

            l = self._ep_initial_dist
            p = self._ep_path_len
            spl = float(success) * (l / max(p, l)) if l > 0 else 0.0

            wandb.log(
                {
                    "episode/reward": self._ep_reward,
                    "episode/success": float(success),
                    "episode/collision": float(collision),
                    "episode/timeout": float(timeout),
                    "episode/spl": spl,
                    "episode/dist_final": float(info.get("dist", 0.0)),
                    "episode/success_rate": sum(self._success_buf)
                    / len(self._success_buf),
                    "episode/collision_rate": sum(self._collision_buf)
                    / len(self._collision_buf),
                    "episode/count": self._ep_count,
                },
                step=self._total_steps,
            )

        return obs, reward, terminated, truncated, info


def build_ppo(obs_space, act_space, cfg, train_cfg):
    models = {
        "policy": PointNavActor(obs_space, act_space, DEVICE, IMG_SIZE),
        "value": PointNavCritic(obs_space, act_space, DEVICE, IMG_SIZE),
    }
    memory = RandomMemory(memory_size=cfg.rollouts, num_envs=1, device=DEVICE)
    agent_cfg = PPO_CFG(
        rollouts=cfg.rollouts,
        learning_epochs=cfg.learning_epochs,
        mini_batches=cfg.mini_batches,
        discount_factor=cfg.discount_factor,
        gae_lambda=cfg.gae_lambda,
        learning_rate=cfg.learning_rate,
        grad_norm_clip=cfg.grad_norm_clip,
        ratio_clip=cfg.ratio_clip,
        value_loss_scale=cfg.value_loss_scale,
        entropy_loss_scale=cfg.entropy_loss_scale,
        kl_threshold=cfg.kl_threshold,
        experiment={
            "directory": LOG_DIR,
            "experiment_name": "",
            "write_interval": train_cfg.log_interval,
            "checkpoint_interval": train_cfg.checkpoint_interval,
            "wandb": False,
        },
    )
    return models, PPO(
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=obs_space,
        action_space=act_space,
        device=DEVICE,
    )


def build_sac(obs_space, act_space, cfg, train_cfg):
    def make_critic():
        return SACPointNavCritic(obs_space, act_space, DEVICE, IMG_SIZE)

    models = {
        "policy": SACPointNavActor(obs_space, act_space, DEVICE, IMG_SIZE),
        "critic_1": make_critic(),
        "critic_2": make_critic(),
        "target_critic_1": make_critic(),
        "target_critic_2": make_critic(),
    }
    models["target_critic_1"].load_state_dict(models["critic_1"].state_dict())
    models["target_critic_2"].load_state_dict(models["critic_2"].state_dict())

    memory = RandomMemory(memory_size=cfg.memory_size, num_envs=1, device=DEVICE)
    agent_cfg = SAC_CFG(
        gradient_steps=1,
        batch_size=cfg.batch_size,
        discount_factor=cfg.discount_factor,
        polyak=cfg.polyak,
        learning_rate=cfg.learning_rate,
        random_timesteps=cfg.random_timesteps,
        learning_starts=cfg.learning_starts,
        grad_norm_clip=cfg.grad_norm_clip,
        learn_entropy=cfg.learn_entropy,
        initial_entropy_value=cfg.initial_entropy_value,
        experiment={
            "directory": LOG_DIR,
            "experiment_name": "",
            "write_interval": train_cfg.log_interval,
            "checkpoint_interval": train_cfg.checkpoint_interval,
            "wandb": False,
        },
    )
    return models, SAC(
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=obs_space,
        action_space=act_space,
        device=DEVICE,
    )


def main():
    train_cfg = PointNavTrainCfg()
    if args.algo is not None:
        train_cfg.algo = args.algo
    if args.run_name is not None:
        train_cfg.run_name = args.run_name

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project="isaac-gs-point-nav",
            name=train_cfg.run_name,
            config={
                "algorithm": train_cfg.algo.upper(),
                "total_timesteps": train_cfg.total_timesteps,
            },
            dir=LOG_DIR,
        )

    env_cfg = PointNavEnvCfg()
    env_cfg.fixed_spawn_pos = train_cfg.fixed_spawn_pos
    env_cfg.fixed_goal_pos = train_cfg.fixed_goal_pos

    gym_env = PointNavGymEnv(cfg=env_cfg)
    gym_env.reset()
    env = wrap_env(gym_env)

    obs_space = gym_env.observation_space
    act_space = gym_env.action_space

    if train_cfg.algo == "ppo":
        models, agent = build_ppo(obs_space, act_space, train_cfg.ppo, train_cfg)
    else:
        models, agent = build_sac(obs_space, act_space, train_cfg.sac, train_cfg)

    if use_wandb:
        gym_env = WandbEpisodeLogger(gym_env, models=models)
        env = wrap_env(gym_env)

    if args.checkpoint:
        models["policy"].load(args.checkpoint)
        print(f"[train] Loaded checkpoint: {args.checkpoint}")

    trainer = SequentialTrainer(
        cfg=SequentialTrainerCfg(
            timesteps=train_cfg.total_timesteps, headless=args.headless
        ),
        env=env,
        agents=agent,
    )

    print(f"[train] Start training with {train_cfg.algo.upper()}...")
    trainer.train()

    if use_wandb:
        wandb.finish()
    gym_env.close()
    app.close()


if __name__ == "__main__":
    main()
