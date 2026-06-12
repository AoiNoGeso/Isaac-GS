"""
Step 1〜5 個別動作確認スクリプト

実行方法:
  cd ~/Programs/Isaac-GS

  # Step 1: World 起動 + stage ロード + Carter スポーン
  uv run tasks/point_navigation/check_env_v2.py --step 1

  # Step 2: RGB-D カメラ画像を PNG 保存
  uv run tasks/point_navigation/check_env_v2.py --step 2

  # Step 3: gymnasium API バリデーション
  uv run tasks/point_navigation/check_env_v2.py --step 3

  # Step 4: CNN policy フォワードパス
  uv run tasks/point_navigation/check_env_v2.py --step 4 --headless

  # Step 5: PPO 10イテレーション動作確認
  uv run tasks/point_navigation/check_env_v2.py --step 5 --headless
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--step", type=int, required=True, choices=[1, 2, 3, 4, 5])
parser.add_argument("--headless", action="store_true", default=False)
args = parser.parse_args()

from isaacsim import SimulationApp
app = SimulationApp({"headless": args.headless})

import torch
import numpy as np
sys.path.insert(0, os.path.expanduser("~/Programs/Isaac-GS"))


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: World 起動 + Carter スポーン + 10ステップ
# ─────────────────────────────────────────────────────────────────────────────
def check_step1():
    print("\n[Step 1] World 起動 + stage ロード + Carter スポーン")
    from tasks.point_navigation.env.isaac_env import PointNavIsaacEnv, PointNavEnvCfg

    env = PointNavIsaacEnv(PointNavEnvCfg())
    obs = env.reset()

    print(f"  obs keys: {list(obs.keys())}")
    print(f"  rgb shape:   {obs['rgb'].shape}")
    print(f"  depth shape: {obs['depth'].shape}")

    for i in range(10):
        action = np.random.uniform(-1, 1, size=(2,)).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  step={i:2d} | reward={reward:.4f} | dist={info['dist']:.2f}m | done={terminated or truncated}")

    env.close()
    print("✅ Step 1 OK\n")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: カメラ画像を PNG 保存して目視確認
# ─────────────────────────────────────────────────────────────────────────────
def check_step2():
    print("\n[Step 2] RGB-D カメラ画像を保存")
    import imageio
    from tasks.point_navigation.env.isaac_env import PointNavIsaacEnv, PointNavEnvCfg

    env = PointNavIsaacEnv(PointNavEnvCfg())
    obs = env.reset()

    rgb, depth = env._camera.get_rgbd()
    print(f"  RGB  shape={rgb.shape}, dtype={rgb.dtype}, min={rgb.min()}, max={rgb.max()}, mean={rgb.mean():.1f}")
    print(f"  Depth shape={depth.shape}, dtype={depth.dtype}, min={depth.min():.2f}, max={depth.max():.2f}, mean={depth.mean():.2f}")
    print(f"  Depth non-zero pixels: {(depth > 0).sum()} / {depth.size}")

    out_dir = os.path.expanduser("~/Programs/Isaac-GS/debug_images")
    os.makedirs(out_dir, exist_ok=True)
    imageio.imwrite(os.path.join(out_dir, "rgb.png"), rgb)

    # inf を最大値に置換してから可視化
    depth_finite = np.where(np.isfinite(depth), depth, 10.0)
    depth_vis = np.clip(depth_finite / 10.0 * 255, 0, 255).astype(np.uint8)
    imageio.imwrite(os.path.join(out_dir, "depth.png"), depth_vis)

    print(f"  保存先: {out_dir}/rgb.png, depth.png")
    env.close()
    print("✅ Step 2 OK\n")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: gymnasium API バリデーション
# ─────────────────────────────────────────────────────────────────────────────
def check_step3():
    """
    gymnasium の check_env は物理シミュレータの非決定性でfailするため
    API 準拠を手動で検証する．
    """
    print("\n[Step 3] gymnasium API 手動バリデーション")
    import gymnasium as gym
    from gymnasium import spaces
    from tasks.point_navigation.env.gym_wrapper import PointNavGymEnv, PointNavEnvCfg

    env = PointNavGymEnv(PointNavEnvCfg())

    # observation_space / action_space の型確認
    assert isinstance(env.observation_space, spaces.Dict), "obs space must be Dict"
    assert isinstance(env.action_space, spaces.Box), "act space must be Box"
    print(f"  observation_space: {env.observation_space}")
    print(f"  action_space:      {env.action_space}")

    # reset() の戻り値確認
    obs, info = env.reset()
    assert isinstance(obs, dict), "reset obs must be dict"
    assert "rgb"   in obs and "depth" in obs
    assert env.observation_space.contains(obs), f"obs out of space: rgb={obs['rgb'].min():.2f}~{obs['rgb'].max():.2f}"
    print(f"  reset() OK: rgb={obs['rgb'].shape}, depth={obs['depth'].shape}")

    # step() の戻り値確認（ランダムアクション 3 回）
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs), f"step obs out of space at step {i}"
        assert isinstance(reward, float), "reward must be float"
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
        assert isinstance(info, dict)
        print(f"  step {i}: reward={reward:.4f}, terminated={terminated}, truncated={truncated}, dist={info.get('dist', '?'):.2f}m")

    env.close()
    print("✅ Step 3 OK\n")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: CNN policy フォワードパス（IsaacSim 不要）
# ─────────────────────────────────────────────────────────────────────────────
def check_step4():
    print("\n[Step 4] CNN policy フォワードパス確認")
    import gymnasium as gym
    from gymnasium import spaces
    import numpy as np
    from tasks.point_navigation.policy.cnn_encoder import PointNavActor, PointNavCritic

    device = "cuda" if torch.cuda.is_available() else "cpu"
    obs_space = spaces.Dict({
        "rgb":   spaces.Box(0.0, 1.0, shape=(3, 128, 128), dtype=np.float32),
        "depth": spaces.Box(0.0, 1.0, shape=(1, 128, 128), dtype=np.float32),
    })
    act_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    actor  = PointNavActor(obs_space, act_space, device).to(device)
    critic = PointNavCritic(obs_space, act_space, device).to(device)

    B = 4
    # skrl 2.x: inputs["observations"] に dict を渡す
    dummy_inputs = {
        "observations": {
            "rgb":   torch.rand(B, 3, 128, 128, device=device),
            "depth": torch.rand(B, 1, 128, 128, device=device),
        }
    }

    mean, actor_out = actor.compute(dummy_inputs, role="policy")
    value, _        = critic.compute(dummy_inputs, role="value")
    log_std = actor_out["log_std"]

    print(f"  Actor  output: mean={mean.shape}, log_std={log_std.shape}")
    print(f"  Critic output: value={value.shape}")
    assert mean.shape == (B, 2)
    assert value.shape == (B, 1)
    print("✅ Step 4 OK\n")


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: PPO 10イテレーション動作確認
# ─────────────────────────────────────────────────────────────────────────────
def check_step5():
    print("\n[Step 5] skrl PPO 10イテレーション動作確認")
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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gym_env = PointNavGymEnv(PointNavEnvCfg())
    gym_env.reset()
    env = wrap_env(gym_env)  # skrl 2.x: gymnasium env を Wrapper でラップ

    models = {
        "policy": PointNavActor(gym_env.observation_space, gym_env.action_space, device),
        "value":  PointNavCritic(gym_env.observation_space, gym_env.action_space, device),
    }
    memory = RandomMemory(memory_size=64, num_envs=1, device=device)

    ppo_cfg = PPO_CFG(
        rollouts=64,
        learning_epochs=2,
        mini_batches=2,
        learning_rate=3e-4,
        grad_norm_clip=1.0,
        ratio_clip=0.2,
        entropy_loss_scale=0.005,
        kl_threshold=0.01,
    )

    agent = PPO(
        models=models,
        memory=memory,
        cfg=ppo_cfg,
        observation_space=gym_env.observation_space,
        action_space=gym_env.action_space,
        device=device,
    )

    trainer = SequentialTrainer(
        cfg=SequentialTrainerCfg(timesteps=128, headless=True),
        env=env,
        agents=agent,
    )
    trainer.train()

    env.close()
    print("✅ Step 5 OK\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    steps = {
        1: check_step1,
        2: check_step2,
        3: check_step3,
        4: check_step4,
        5: check_step5,
    }
    steps[args.step]()
    app.close()
