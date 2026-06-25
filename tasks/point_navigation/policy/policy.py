"""
SAC (Soft Actor-Critic)

構成:
  ReplayBuffer  - 経験再生バッファ
  Actor         - tanh squashing + 対数確率
  Critic        - Clipped Double-Q (Q1, Q2)
  SACAgent      - 学習・推論・保存/ロード
"""

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_LOG_STD_MIN = -5
_LOG_STD_MAX = 2


# ---------------------------------------------------------------------------
# ReplayBuffer
# ---------------------------------------------------------------------------


class ReplayBuffer:
    """観測対応のオフポリシーバッファ"""

    def __init__(
        self,
        capacity: int,
        obs_spec: dict[str, tuple[int, ...]],
        action_dim: int,
        device: str,
    ):
        self._cap = capacity
        self._ptr = 0
        self._size = 0
        self._dev = device

        self._obs = {
            k: np.zeros((capacity, *s), dtype=np.float32) for k, s in obs_spec.items()
        }
        self._next_obs = {
            k: np.zeros((capacity, *s), dtype=np.float32) for k, s in obs_spec.items()
        }
        self._actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rewards = np.zeros((capacity, 1), dtype=np.float32)
        self._dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(
        self,
        obs: dict,
        action: np.ndarray,
        reward: float,
        next_obs: dict,
        done: float,
    ):
        for k in self._obs:
            self._obs[k][self._ptr] = obs[k]
            self._next_obs[k][self._ptr] = next_obs[k]
        self._actions[self._ptr] = action
        self._rewards[self._ptr] = reward
        self._dones[self._ptr] = done
        self._ptr = (self._ptr + 1) % self._cap
        self._size = min(self._size + 1, self._cap)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self._size, size=batch_size)
        to_t = lambda x: torch.FloatTensor(x).to(self._dev)
        obs = {k: to_t(self._obs[k][idx]) for k in self._obs}
        next_obs = {k: to_t(self._next_obs[k][idx]) for k in self._next_obs}
        return (
            obs,
            to_t(self._actions[idx]),
            to_t(self._rewards[idx]),
            next_obs,
            to_t(self._dones[idx]),
        )

    def __len__(self) -> int:
        return self._size


# ---------------------------------------------------------------------------
# Actor / Critic
# ---------------------------------------------------------------------------


class Actor(nn.Module):
    """ガウス方策 + tanh squashing"""

    def __init__(self, feat_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)

    def _dist(self, feat: torch.Tensor):
        h = self.net(feat)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(_LOG_STD_MIN, _LOG_STD_MAX)
        return mean, log_std

    def sample(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """reparameterization sample + log_prob"""
        mean, log_std = self._dist(feat)
        std = log_std.exp()
        u = mean + std * torch.randn_like(mean)
        action = torch.tanh(u)
        log_prob = (
            -0.5 * ((u - mean) / std).pow(2)
            - log_std
            - math.log(math.sqrt(2 * math.pi))
        ).sum(-1, keepdim=True)
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-6).sum(-1, keepdim=True)
        return action, log_prob

    def mean_action(self, feat: torch.Tensor) -> torch.Tensor:
        """決定論的推論用（tanh(mean)）"""
        mean, _ = self._dist(feat)
        return torch.tanh(mean)


class Critic(nn.Module):
    """Clipped Double-Q ネットワーク"""

    def __init__(self, feat_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()

        def _mlp():
            return nn.Sequential(
                nn.Linear(feat_dim + action_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        self.q1 = _mlp()
        self.q2 = _mlp()

    def forward(self, feat: torch.Tensor, action: torch.Tensor):
        x = torch.cat([feat, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q_min(self, feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self(feat, action)
        return torch.min(q1, q2)


# ---------------------------------------------------------------------------
# SACAgent
# ---------------------------------------------------------------------------


class SACAgent:
    """
    SAC Agent

    Args:
        encoder_factory: PointNavEncoder を返す
        action_dim:      行動次元数
        cfg:             SACCfg
        device:          "cuda" or "cpu"
    """

    def __init__(self, encoder_factory, action_dim: int, cfg: Any, device: str):
        self.device = device
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.batch_size = cfg.batch_size
        self.gradient_steps = cfg.gradient_steps

        # ── エンコーダ（Actor / Critic 独立）──────────────────────────
        self.actor_enc = encoder_factory().to(device)
        self.critic_enc = encoder_factory().to(device)
        feat_dim = self.actor_enc.out_dim

        # ── ポリシー / Q ネットワーク ──────────────────────────────────
        self.actor = Actor(feat_dim, action_dim).to(device)
        self.critic = Critic(feat_dim, action_dim).to(device)

        # ── ターゲットネットワーク（Critic のみ）──────────────────────
        self.target_critic_enc = copy.deepcopy(self.critic_enc)
        self.target_critic = copy.deepcopy(self.critic)
        for p in (
            *self.target_critic_enc.parameters(),
            *self.target_critic.parameters(),
        ):
            p.requires_grad_(False)

        # ── オプティマイザ ────────────────────────────────────────────
        self.actor_opt = torch.optim.Adam(
            [*self.actor_enc.parameters(), *self.actor.parameters()],
            lr=cfg.learning_rate,
        )
        self.critic_opt = torch.optim.Adam(
            [*self.critic_enc.parameters(), *self.critic.parameters()],
            lr=cfg.learning_rate,
        )

        # ── 自動エントロピー調整（α）────────────────────────────────
        self.target_entropy = (
            float(-action_dim)
            if cfg.target_entropy == "auto"
            else float(cfg.target_entropy)
        )
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.learning_rate)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # ── 推論 ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def act(self, obs: dict, deterministic: bool = False) -> np.ndarray:
        obs_t = {k: torch.FloatTensor(v[None]).to(self.device) for k, v in obs.items()}
        feat = self.actor_enc(obs_t)
        if deterministic:
            action = self.actor.mean_action(feat)
        else:
            action, _ = self.actor.sample(feat)
        return action.cpu().numpy()[0]

    # ── 学習ステップ ─────────────────────────────────────────────────────

    def update(self, buffer: ReplayBuffer) -> dict:
        metrics: dict[str, float] = {}
        for _ in range(self.gradient_steps):
            obs, actions, rewards, next_obs, dones = buffer.sample(self.batch_size)

            # ── Critic 更新 ───────────────────────────────────────────
            with torch.no_grad():
                next_feat_a = self.actor_enc(next_obs)
                next_actions, next_lp = self.actor.sample(next_feat_a)
                next_feat_c = self.target_critic_enc(next_obs)
                target_q = self.target_critic.q_min(next_feat_c, next_actions)
                target_q = rewards + (1.0 - dones) * self.gamma * (
                    target_q - self.alpha * next_lp
                )

            feat_c = self.critic_enc(obs)
            q1, q2 = self.critic(feat_c, actions)
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

            # ── Actor 更新（Critic パラメータを一時凍結）──────────────
            self._set_critic_grad(False)

            feat_a = self.actor_enc(obs)
            new_actions, log_p = self.actor.sample(feat_a)
            with torch.no_grad():
                feat_c_d = self.critic_enc(obs)
            actor_q = self.critic.q_min(feat_c_d, new_actions)
            actor_loss = (self.alpha.detach() * log_p - actor_q).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self._set_critic_grad(True)

            # ── α 更新 ────────────────────────────────────────────────
            alpha_loss = -(
                self.log_alpha * (log_p.detach() + self.target_entropy)
            ).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

            # ── ソフトターゲット更新 ──────────────────────────────────
            self._soft_update(self.critic_enc, self.target_critic_enc)
            self._soft_update(self.critic, self.target_critic)

            metrics = {
                "train/critic_loss": critic_loss.item(),
                "train/actor_loss": actor_loss.item(),
                "train/alpha_loss": alpha_loss.item(),
                "train/alpha": self.alpha.item(),
                "train/log_prob": log_p.mean().item(),
                "train/q_mean": target_q.mean().item(),
            }
        return metrics

    # ── 保存 / ロード ────────────────────────────────────────────────────

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor_enc": self.actor_enc.state_dict(),
                "actor": self.actor.state_dict(),
                "critic_enc": self.critic_enc.state_dict(),
                "critic": self.critic.state_dict(),
                "log_alpha": self.log_alpha.data,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor_enc.load_state_dict(ckpt["actor_enc"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic_enc.load_state_dict(ckpt["critic_enc"])
        self.critic.load_state_dict(ckpt["critic"])
        self.log_alpha.data = ckpt["log_alpha"]
        # ターゲットネットワークをリセット
        self.target_critic_enc = copy.deepcopy(self.critic_enc)
        self.target_critic = copy.deepcopy(self.critic)
        for p in (
            *self.target_critic_enc.parameters(),
            *self.target_critic.parameters(),
        ):
            p.requires_grad_(False)
        print(f"[SACAgent] Loaded checkpoint: {path}")

    # ── 内部ユーティリティ ───────────────────────────────────────────────

    def _soft_update(self, src: nn.Module, tgt: nn.Module):
        for ps, pt in zip(src.parameters(), tgt.parameters()):
            pt.data.mul_(1.0 - self.tau).add_(ps.data, alpha=self.tau)

    def _set_critic_grad(self, requires: bool):
        for p in (*self.critic.parameters(), *self.critic_enc.parameters()):
            p.requires_grad_(requires)
