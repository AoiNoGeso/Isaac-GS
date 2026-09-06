"""SAC (Soft Actor-Critic): ReplayBuffer / Actor(tanh squashing) / Critic(Clipped Double-Q) / SACAgent"""

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


_NP_TO_TORCH_DTYPE = {
    np.dtype(np.float32): torch.float32,
    np.dtype(np.float16): torch.float16,
    np.dtype(np.uint8): torch.uint8,
}


class ReplayBuffer:
    """遅延スタッキング対応のオフポリシーバッファ。

    1フレームだけを保存し、stack_size>1のときはsample()/encode_recent_observation()時に
    過去フレームを遅延結合する(next_obsもobsと同じ配列をidx+1で参照するだけで複製しない)。
    stack_size=1なら結合処理を経由せず、従来と全く同じ観測を返す。

    バッファ本体はpinned(ページロック)CPUメモリ上のtorch.Tensorとして確保し、sample()時に
    .to(device, non_blocking=True)で転送する(tests/buffer_style_test.py参照。従来のnumpy配列
    +torch.FloatTensor(x).to(device)方式よりsample()が高速)。

    典型的な使い方(train_loop.py参照):
        idx = buffer.store_frame(obs)             # 1フレーム保存、行動選択用にidxを控える
        action = agent.act(buffer.encode_recent_observation())
        next_obs, reward, terminated, truncated, info = env.step(action)
        buffer.store_effect(idx, action, reward, float(terminated))
        if truncated and not terminated:
            # Bellman targetがnext_obsを使う(done=0)ため、reset前の真の継続フレームを
            # 行動選択とは無関係に保存しておく(store_effectは呼ばない=sample()の対象外)
            buffer.store_frame(next_obs)
        if terminated or truncated:
            obs, _ = env.reset()
            buffer.reset_episode()
    """

    _STACK_KEY = "dino_feat"  # このキーだけstack_size倍にスタックする

    def __init__(
        self,
        capacity: int,
        obs_spec: dict[str, tuple[int, ...]],
        action_dim: int,
        device: str,
        obs_dtypes: dict[str, type] | None = None,
        stack_size: int = 1,
    ):
        """obs_specは1フレームぶんの形状(スタック前)。obs_dtypesでnp.uint8を指定したキーは
        [0,1]のfloatとして受け取り、内部ではuint8で保持してメモリを節約する"""
        self._cap = capacity
        self._stack = stack_size
        self._ptr = 0
        self._size = 0
        self._dev = device
        self._pin = torch.device(device).type == "cuda"
        obs_dtypes = obs_dtypes or {}
        self._obs_dtype = {k: obs_dtypes.get(k, np.float32) for k in obs_spec}
        self._obs_spec = obs_spec
        self._cur_episode_step = 0

        def _zeros(shape: tuple[int, ...], np_dtype) -> torch.Tensor:
            torch_dtype = _NP_TO_TORCH_DTYPE[np.dtype(np_dtype)]
            return torch.zeros(shape, dtype=torch_dtype, pin_memory=self._pin)

        self._frames = {k: _zeros((capacity, *s), self._obs_dtype[k]) for k, s in obs_spec.items()}
        self._episode_step = torch.zeros(capacity, dtype=torch.int32)
        self._valid = torch.zeros(capacity, dtype=torch.bool)  # store_effect済みのスロットのみTrue
        self._actions = _zeros((capacity, action_dim), np.float32)
        self._rewards = _zeros((capacity, 1), np.float32)
        self._dones = _zeros((capacity, 1), np.float32)

    def _encode(self, k: str, value: np.ndarray) -> np.ndarray:
        if self._obs_dtype[k] == np.uint8:
            return np.round(value * 255.0).astype(np.uint8)
        return value

    def store_frame(self, obs: dict) -> int:
        """1step分の観測(1フレーム)だけを保存し、書き込んだスロットのインデックスを返す"""
        i = self._ptr
        for k, arr in self._frames.items():
            arr[i] = torch.from_numpy(np.asarray(self._encode(k, obs[k])))
        self._episode_step[i] = self._cur_episode_step
        self._valid[i] = False
        self._cur_episode_step += 1
        self._ptr = (i + 1) % self._cap
        self._size = min(self._size + 1, self._cap)
        return i

    def store_effect(self, idx: int, action, reward: float, done: float) -> None:
        """store_frame()が返したidxへ行動・報酬・doneを書き込み、サンプル対象として有効化する"""
        self._actions[idx] = torch.from_numpy(np.asarray(action, dtype=np.float32))
        self._rewards[idx] = float(reward)
        self._dones[idx] = float(done)
        self._valid[idx] = True

    def reset_episode(self) -> None:
        """env.reset()の直後に呼ぶ。次のstore_frame()をエピソード先頭(episode_step=0)として扱う"""
        self._cur_episode_step = 0

    def _gather(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        """idx(各サンプルの現在フレーム位置)のスタック観測をdevice上のfloatテンソルで返す"""
        ep = self._episode_step[idx]
        out = {}
        for k, frames in self._frames.items():
            if k == self._STACK_KEY and self._stack > 1:
                parts = []
                for back_k in range(self._stack - 1, -1, -1):  # 古い→新しいの順に積む
                    back = torch.minimum(torch.full_like(ep, back_k), ep)
                    src = (idx - back) % self._cap
                    parts.append(frames.index_select(0, src))
                x = torch.cat(parts, dim=1)
            else:
                x = frames.index_select(0, idx)
            x = x.to(self._dev, dtype=torch.float32, non_blocking=self._pin)
            if self._obs_dtype[k] == np.uint8:
                x = x / 255.0
            out[k] = x
        return out

    def encode_recent_observation(self) -> dict[str, np.ndarray]:
        """直近のstore_frame()呼び出し分のスタック観測をnumpyで返す(agent.act()用)"""
        idx = torch.tensor([(self._ptr - 1) % self._cap])
        return {k: v[0].cpu().numpy() for k, v in self._gather(idx).items()}

    def sample(self, batch_size: int):
        idx = torch.randint(0, self._size, (batch_size,))
        invalid = ~self._valid[idx]
        while invalid.any():
            idx[invalid] = torch.randint(0, self._size, (int(invalid.sum()),))
            invalid = ~self._valid[idx]

        obs = self._gather(idx)
        next_obs = self._gather((idx + 1) % self._cap)

        def to_t_plain(t: torch.Tensor) -> torch.Tensor:
            return t.index_select(0, idx).to(self._dev, dtype=torch.float32, non_blocking=self._pin)

        return obs, to_t_plain(self._actions), to_t_plain(self._rewards), next_obs, to_t_plain(self._dones)

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
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
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
    """Actor/Criticの学習・推論・保存を統括するSACエージェント"""

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
