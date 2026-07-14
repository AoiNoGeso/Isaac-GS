"""
Point Navigation 学習スクリプト
IsaacSim 6.0 + 自前 SAC + wandb ログ

実行方法:
  cd ~/Programs/Isaac-GS
  uv run tasks/point_navigation/train.py --headless
  uv run tasks/point_navigation/train.py --headless --run-name my_run
  uv run tasks/point_navigation/train.py --headless --checkpoint runs/point_nav/checkpoints/sac_10000.pt
  uv run tasks/point_navigation/train.py --headless --num-humans 2
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

import numpy as np
import torch
import wandb
from tqdm import tqdm

# Isaac Sim が stderr を横取りするため stdout に固定
_OUT = sys.stdout

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from envs.isaac_env import PointNavGymEnv
from tasks.point_navigation.config import EnvConfig, ModelConfig, TrainConfig
from tasks.point_navigation.policy.network import PointNavEncoder
from tasks.point_navigation.policy.policy import ReplayBuffer, SACAgent

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# エピソード統計トラッカー
# ---------------------------------------------------------------------------


class EpisodeTracker:
    def __init__(self, window: int = 100):
        self._window = window
        self._ep_reward = 0.0
        self._ep_steps = 0
        self._ep_path_len = 0.0
        self._ep_init_dist = 0.0
        self._prev_xy = None
        self._ep_count = 0
        self._total_success = 0
        self._total_collision = 0
        self._total_human_collision = 0
        self._success_buf: list[float] = []
        self._collision_buf: list[float] = []

    def step(self, reward: float, info: dict):
        self._ep_reward += reward
        self._ep_steps += 1
        xy = info.get("robot_xz")
        if xy is not None:
            xy = np.array(xy, dtype=np.float32)
            if self._prev_xy is not None:
                self._ep_path_len += float(np.linalg.norm(xy - self._prev_xy))
            self._prev_xy = xy

    def reset(self, obs: dict, info: dict | None = None):
        self._ep_reward = 0.0
        self._ep_steps = 0
        self._ep_path_len = 0.0
        self._prev_xy = None
        if "goal" in obs:
            self._ep_init_dist = float(obs["goal"][0])  # dist [m]
        elif info is not None and "dist" in info:
            self._ep_init_dist = float(
                info["dist"]
            )  # input_goal=False 時のフォールバック

    def finish(self, info: dict) -> dict:
        self._ep_count += 1
        success = bool(info.get("success", False))
        collision = bool(info.get("collision", False))
        human_collision = bool(info.get("human_collision", False))
        timeout = bool(info.get("timeout", False))

        self._total_success += int(success)
        self._total_collision += int(collision)
        self._total_human_collision += int(human_collision)
        self._success_buf.append(float(success))
        self._collision_buf.append(float(collision))
        if len(self._success_buf) > self._window:
            self._success_buf.pop(0)
        if len(self._collision_buf) > self._window:
            self._collision_buf.pop(0)

        l = self._ep_init_dist
        p = self._ep_path_len
        spl = float(success) * (l / max(p, l)) if l > 0 else 0.0

        return {
            "episode/reward": self._ep_reward,
            "episode/steps": self._ep_steps,
            "episode/success": float(success),
            "episode/collision": float(collision),
            "episode/human_collision": float(human_collision),
            "episode/timeout": float(timeout),
            "episode/spl": spl,
            "episode/dist_final": float(info.get("dist", 0.0)),
            "episode/success_rate": self._total_success / self._ep_count,
            "episode/collision_rate": self._total_collision / self._ep_count,
            "episode/human_collision_rate": self._total_human_collision / self._ep_count,
            "episode/count": self._ep_count,
        }


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------


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

    # ── 環境 ─────────────────────────────────────────────────────────────
    env = PointNavGymEnv(env_cfg=env_cfg, model_cfg=model_cfg)
    obs, _ = env.reset()

    # ── obs_spec（バッファ構築用）────────────────────────────────────────
    obs_spec = {k: v.shape for k, v in env.observation_space.spaces.items()}
    action_dim = env.action_space.shape[0]

    # ── リプレイバッファ ──────────────────────────────────────────────────
    buffer = ReplayBuffer(
        capacity=train_cfg.buffer_size,
        obs_spec=obs_spec,
        action_dim=action_dim,
        device=DEVICE,
    )

    # ── SAC エージェント ──────────────────────────────────────────────────
    img_size = model_cfg.camera_resolution[0]

    def encoder_factory():
        return PointNavEncoder(
            input_rgb=model_cfg.input_rgb,
            input_goal=model_cfg.input_goal,
            img_size=img_size,
        )

    agent = SACAgent(
        encoder_factory=encoder_factory,
        action_dim=action_dim,
        cfg=train_cfg,
        device=DEVICE,
    )

    if args.checkpoint:
        agent.load(args.checkpoint)

    # ── 学習ループ ────────────────────────────────────────────────────────
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
        # 行動選択（warmup 中はランダム）
        if len(buffer) < train_cfg.learning_starts:
            action = env.action_space.sample()
        else:
            action = agent.act(obs, deterministic=False)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # バッファに格納（timeout による truncation では done=0 とする）
        buffer.add(obs, action, reward, next_obs, float(terminated))
        tracker.step(reward, info)
        obs = next_obs

        # エピソード終了
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

        # 学習ステップ
        if (
            len(buffer) >= train_cfg.learning_starts
            and step % train_cfg.train_freq == 0
        ):
            metrics = agent.update(buffer)
            if use_wandb and step % train_cfg.log_interval == 0:
                wandb.log(metrics, step=step)

        # チェックポイント保存
        if step % train_cfg.checkpoint_interval == 0:
            ckpt_path = f"{ckpt_dir}/sac_{step}.pt"
            agent.save(ckpt_path)
            tqdm.write(f"[train] Checkpoint saved: {ckpt_path}", file=_OUT)

    # ── 最終保存 ─────────────────────────────────────────────────────────
    final_path = f"{log_dir}/sac_final.pt"
    agent.save(final_path)
    print(f"[train] Final model saved: {final_path}")

    if use_wandb:
        wandb.finish()
    env.close()
    app.close()


if __name__ == "__main__":
    main()
