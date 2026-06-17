"""
Point Navigation 学習スクリプト
IsaacSim 6.0 + Stable-Baselines3 + CNN policy + wandb ログ

実行方法:
  cd ~/Programs/Isaac-GS
  uv run tasks/point_navigation/train.py --headless
  uv run tasks/point_navigation/train.py --headless --algo ppo
  uv run tasks/point_navigation/train.py --headless --run-name my_run
  uv run tasks/point_navigation/train.py --headless --checkpoint runs/point_nav/checkpoints/best_model
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

import numpy as np
import torch
import wandb
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from envs.gym_wrapper import PointNavGymEnv
from tasks.point_navigation.config import PointNavEnvCfg, PointNavTrainCfg
from tasks.point_navigation.policy.sb3_policy import PointNavFeaturesExtractor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_DIR = "runs/point_nav"
CHECKPOINT_DIR = f"{LOG_DIR}/checkpoints"


# ---------------------------------------------------------------------------
# wandb コールバック
# ---------------------------------------------------------------------------

class WandbEpisodeCallback(BaseCallback):
    """エピソード終了時に報酬・成功率・SPL を wandb に記録する"""

    def __init__(self, window: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self._ep_reward = 0.0
        self._ep_count = 0
        self._ep_initial_dist = 0.0
        self._ep_path_len = 0.0
        self._prev_xy = None
        self._success_buf: list[float] = []
        self._collision_buf: list[float] = []
        self._window = window

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        reward = self.locals["rewards"][0]
        done = self.locals["dones"][0]

        self._ep_reward += float(reward)

        xy = info.get("robot_xz")
        if xy is not None:
            xy = np.array(xy, dtype=np.float32)
            if self._prev_xy is not None:
                self._ep_path_len += float(np.linalg.norm(xy - self._prev_xy))
            self._prev_xy = xy

        if done:
            self._ep_count += 1
            success = bool(info.get("success", False))
            collision = bool(info.get("collision", False))
            timeout = bool(info.get("timeout", False))

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
                    "episode/success_rate": sum(self._success_buf) / len(self._success_buf),
                    "episode/collision_rate": sum(self._collision_buf) / len(self._collision_buf),
                    "episode/count": self._ep_count,
                },
                step=self.num_timesteps,
            )

            self._ep_reward = 0.0
            self._ep_path_len = 0.0
            self._prev_xy = None
            obs = self.locals.get("new_obs") or self.locals.get("obs")
            if obs is not None:
                goal = obs["goal"][0] if isinstance(obs, dict) else obs[0]
                self._ep_initial_dist = float(goal[0]) * 10.0

        return True


class WandbTrainLogCallback(BaseCallback):
    """SB3 のログ変数（エントロピー等）を wandb に転送する"""

    def __init__(self, log_interval: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self._log_interval = log_interval

    def _on_step(self) -> bool:
        if self.num_timesteps % self._log_interval == 0:
            log = {}
            for key in ("train/entropy_loss", "train/actor_loss", "train/critic_loss",
                        "train/ent_coef", "train/policy_gradient_loss", "train/value_loss"):
                val = self.logger.name_to_value.get(key)
                if val is not None:
                    log[key] = val
            if log:
                wandb.log(log, step=self.num_timesteps)
        return True


# ---------------------------------------------------------------------------
# モデル構築
# ---------------------------------------------------------------------------

_POLICY_KWARGS = {
    "features_extractor_class": PointNavFeaturesExtractor,
    "features_extractor_kwargs": {"features_dim": 288},
    "net_arch": [128, 64],
}


def build_sac(env: PointNavGymEnv, cfg, checkpoint: str | None) -> SAC:
    if checkpoint:
        model = SAC.load(checkpoint, env=env, device=DEVICE)
        print(f"[train] Loaded SAC checkpoint: {checkpoint}")
        return model

    return SAC(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=cfg.learning_rate,
        buffer_size=cfg.buffer_size,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        tau=cfg.tau,
        learning_starts=cfg.learning_starts,
        train_freq=cfg.train_freq,
        gradient_steps=cfg.gradient_steps,
        ent_coef=cfg.ent_coef,
        target_entropy=cfg.target_entropy,
        policy_kwargs=_POLICY_KWARGS,
        device=DEVICE,
        verbose=1,
    )


def build_ppo(env: PointNavGymEnv, cfg, checkpoint: str | None) -> PPO:
    if checkpoint:
        model = PPO.load(checkpoint, env=env, device=DEVICE)
        print(f"[train] Loaded PPO checkpoint: {checkpoint}")
        return model

    return PPO(
        policy="MultiInputPolicy",
        env=env,
        n_steps=cfg.n_steps,
        n_epochs=cfg.n_epochs,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        learning_rate=cfg.learning_rate,
        max_grad_norm=cfg.max_grad_norm,
        clip_range=cfg.clip_range,
        vf_coef=cfg.vf_coef,
        ent_coef=cfg.ent_coef,
        policy_kwargs=_POLICY_KWARGS,
        device=DEVICE,
        verbose=1,
    )


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------

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

    env = PointNavGymEnv(cfg=env_cfg)
    env.reset()

    if train_cfg.algo == "sac":
        model = build_sac(env, train_cfg.sac, args.checkpoint)
    else:
        model = build_ppo(env, train_cfg.ppo, args.checkpoint)

    callbacks: list[BaseCallback] = [
        CheckpointCallback(
            save_freq=train_cfg.checkpoint_interval,
            save_path=CHECKPOINT_DIR,
            name_prefix=train_cfg.algo,
        ),
    ]
    if use_wandb:
        callbacks += [
            WandbEpisodeCallback(window=100),
            WandbTrainLogCallback(log_interval=train_cfg.log_interval),
        ]

    print(f"[train] Start training with {train_cfg.algo.upper()}...")
    model.learn(
        total_timesteps=train_cfg.total_timesteps,
        callback=CallbackList(callbacks),
        reset_num_timesteps=args.checkpoint is None,
    )

    model.save(f"{LOG_DIR}/{train_cfg.algo}_final")
    print(f"[train] Saved final model to {LOG_DIR}/{train_cfg.algo}_final")

    if use_wandb:
        wandb.finish()
    env.close()
    app.close()


if __name__ == "__main__":
    main()
