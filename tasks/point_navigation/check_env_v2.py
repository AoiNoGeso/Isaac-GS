"""
Step 1〜5 個別動作確認スクリプト

実行方法:
  cd ~/Programs/Isaac-GS

  # Step 4: CNN shape 確認（IsaacSim 不要，最速）
  uv run tasks/point_navigation/check_env_v2.py --step 4 --headless

  # Step 1: World 起動 + stage ロード + Carter スポーン + NavMesh bake 確認
  uv run tasks/point_navigation/check_env_v2.py --step 1

  # Step 2: RGB カメラ画像を debug_images/ に PNG 保存
  uv run tasks/point_navigation/check_env_v2.py --step 2

  # Step 3: gymnasium API バリデーション（goal_vec が obs に含まれるか確認）
  uv run tasks/point_navigation/check_env_v2.py --step 3 --headless

  # Step 5: PPO 128 ステップ動作確認（loss が NaN でないか確認）
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

import omni.log
omni.log.set_level(omni.log.Level.ERROR, channel="omni.physx.plugin")

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/Programs/Isaac-GS"))


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: World 起動 + Carter スポーン + 10ステップ
# ─────────────────────────────────────────────────────────────────────────────
def check_step1():
    print("\n[Step 1] World 起動 + stage ロード + Carter スポーン")
    from tasks.point_navigation.env.isaac_env import PointNavEnvCfg, PointNavIsaacEnv

    env = PointNavIsaacEnv(PointNavEnvCfg())
    obs = env.reset()

    print(f"  obs keys: {list(obs.keys())}")
    print(f"  rgb  shape: {obs['rgb'].shape}")
    print(f"  goal shape: {obs['goal'].shape}  val={obs['goal']}")

    for i in range(1000):
        action = np.random.uniform(-1, 1, size=(2,)).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"  step={i:2d} | reward={reward:.4f} | dist={info['dist']:.2f}m | goal={obs['goal']} | done={terminated or truncated}"
        )

    env.close()
    print("✅ Step 1 OK\n")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: カメラ画像を PNG 保存して目視確認
# ─────────────────────────────────────────────────────────────────────────────
def check_step2():
    print("\n[Step 2] RGB カメラ画像を保存")
    import imageio
    from tasks.point_navigation.env.isaac_env import PointNavEnvCfg, PointNavIsaacEnv

    env = PointNavIsaacEnv(PointNavEnvCfg())
    env.reset()

    rgb, _ = env._camera.get_rgbd()
    print(
        f"  RGB  shape={rgb.shape}, dtype={rgb.dtype}, min={rgb.min()}, max={rgb.max()}, mean={rgb.mean():.1f}"
    )

    out_dir = os.path.expanduser("~/Programs/Isaac-GS/debug_images")
    os.makedirs(out_dir, exist_ok=True)
    imageio.imwrite(os.path.join(out_dir, "rgb.png"), rgb)
    print(f"  保存先: {out_dir}/rgb.png")

    env.close()
    print("✅ Step 2 OK\n")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: gymnasium API バリデーション
# ─────────────────────────────────────────────────────────────────────────────
def check_step3():
    print("\n[Step 3] gymnasium API 手動バリデーション")
    from gymnasium import spaces
    from tasks.point_navigation.env.gym_wrapper import PointNavEnvCfg, PointNavGymEnv

    env = PointNavGymEnv(PointNavEnvCfg())

    assert isinstance(env.observation_space, spaces.Dict)
    assert isinstance(env.action_space, spaces.Box)
    assert "rgb" in env.observation_space.spaces
    assert "goal" in env.observation_space.spaces
    print(f"  observation_space: {env.observation_space}")
    print(f"  action_space:      {env.action_space}")

    obs, info = env.reset()
    assert isinstance(obs, dict)
    assert "rgb" in obs and "goal" in obs
    assert env.observation_space.contains(obs), (
        f"obs out of space: rgb={obs['rgb'].min():.2f}~{obs['rgb'].max():.2f}, goal={obs['goal']}"
    )
    print(f"  reset() OK: rgb={obs['rgb'].shape}, goal={obs['goal']}")

    for i in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs), f"step obs out of space at step {i}"
        assert isinstance(reward, float)
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
        print(
            f"  step {i}: reward={reward:.4f}, goal={obs['goal']}, dist={info.get('dist', '?'):.2f}m"
        )

    env.close()
    print("✅ Step 3 OK\n")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: CNN policy フォワードパス（IsaacSim 不要）
# ─────────────────────────────────────────────────────────────────────────────
def check_step4():
    print("\n[Step 4] CNN policy フォワードパス確認")
    import numpy as np
    from gymnasium import spaces
    from tasks.point_navigation.policy.cnn_encoder import PointNavActor, PointNavCritic

    device = "cuda" if torch.cuda.is_available() else "cpu"
    obs_space = spaces.Dict(
        {
            "rgb": spaces.Box(0.0, 1.0, shape=(3, 84, 84), dtype=np.float32),
            "goal": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
        }
    )
    act_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    actor = PointNavActor(obs_space, act_space, device).to(device)
    critic = PointNavCritic(obs_space, act_space, device).to(device)

    B = 4
    dummy_inputs = {
        "observations": {
            "rgb": torch.rand(B, 3, 84, 84, device=device),
            "goal": torch.rand(B, 2, device=device),
        }
    }

    mean, actor_out = actor.compute(dummy_inputs, role="policy")
    value, _ = critic.compute(dummy_inputs, role="value")
    log_std = actor_out["log_std"]

    print(f"  Actor  output: mean={mean.shape}, log_std={log_std.shape}")
    print(f"  Critic output: value={value.shape}")
    assert mean.shape == (B, 2)
    assert value.shape == (B, 1)
    print("✅ Step 4 OK\n")


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: PPO 数イテレーション動作確認
# ─────────────────────────────────────────────────────────────────────────────
def check_step5():
    print("\n[Step 5] skrl PPO 数イテレーション動作確認")
    from skrl.agents.torch.ppo import PPO
    from skrl.agents.torch.ppo.ppo_cfg import PPO_CFG
    from skrl.envs.wrappers.torch import wrap_env
    from skrl.memories.torch import RandomMemory
    from skrl.trainers.torch import SequentialTrainer
    from skrl.trainers.torch.sequential import SequentialTrainerCfg
    from skrl.utils import set_seed
    from tasks.point_navigation.env.gym_wrapper import PointNavGymEnv
    from tasks.point_navigation.env.isaac_env import PointNavEnvCfg
    from tasks.point_navigation.policy.cnn_encoder import PointNavActor, PointNavCritic

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gym_env = PointNavGymEnv(PointNavEnvCfg())
    gym_env.reset()
    env = wrap_env(gym_env)

    models = {
        "policy": PointNavActor(
            gym_env.observation_space, gym_env.action_space, device
        ),
        "value": PointNavCritic(
            gym_env.observation_space, gym_env.action_space, device
        ),
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
