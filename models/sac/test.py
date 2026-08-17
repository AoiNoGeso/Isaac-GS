"""Point Navigation テストスクリプト。実行例: uv run models/sac/test.py --model path/to/sac_final.pt --stage-index 0 [--num-humans N]"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# envs/models配下はtorchを巻き込むためimportしない
# (launch_sim()がCUDA_VISIBLE_DEVICESを設定するより前にCUDAが初期化されるのを防ぐ)
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--model", type=str, required=True, help="チェックポイントパス")
parser.add_argument(
    "--stage-index", type=int, default=0, help="評価するステージのインデックス"
)
parser.add_argument("--num-humans", type=int, default=0)
parser.add_argument(
    "--video", action="store_true", default=False, help="各エピソードの映像をmp4で保存する"
)
parser.add_argument(
    "--video-dir", type=str, default="videos", help="動画の出力先ディレクトリ (default: videos)"
)
parser.add_argument("--gpu", type=int, default=0, help="共用マシンでの複数GPU分散を防ぐため、使用するGPU番号を固定する")
args = parser.parse_args()

from utils.launch_sim import launch_sim

app = launch_sim(headless=args.headless, gpu=args.gpu)

import torch
from tqdm import tqdm

_OUT = sys.stdout

from envs import PointNavGymEnv
from envs.config import EnvConfig, get_preset, stage_names
from models.sac.config import ModelConfig, TestConfig, TrainConfig
from models.sac.network import make_encoder
from models.sac.policy import SACAgent
from utils.recorder import EpisodeRecorder, make_overhead_camera
from utils.rollout import evaluate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    """指定ステージでモデルを評価し、成功率等の指標を表示する"""
    test_cfg = TestConfig()

    names = stage_names()
    if args.stage_index >= len(names):
        print(f"[test] Error: stage-index {args.stage_index} は範囲外です (stages={len(names)})")
        app.close()
        return

    stage_name = names[args.stage_index]
    stage_cfg = get_preset(stage_name)

    print(f"[test] Stage {args.stage_index}: {stage_cfg.stage_path}")
    print(f"[test] Model: {args.model}")
    print(f"[test] Episodes: {test_cfg.episodes_per_stage}")
    print(f"[test] num_humans: {args.num_humans}")

    model_cfg = ModelConfig()
    env_cfg = EnvConfig.from_preset(
        stage_name,
        show_camera_viewport=not args.headless,
        num_humans=args.num_humans,
    )

    env = PointNavGymEnv(env_cfg=env_cfg)
    env.reset()  # 俯瞰カメラ生成にはステージのロードが済んでいる必要がある
    action_dim = env.action_space.shape[0]

    agent = SACAgent(
        encoder_factory=lambda: make_encoder(model_cfg, img_size=env_cfg.camera_resolution[0]),
        action_dim=action_dim,
        cfg=TrainConfig(stage=stage_name),
        device=DEVICE,
    )
    agent.load(args.model)

    recorder = None
    if args.video:
        video_dir = Path(args.video_dir) / stage_name / datetime.now().strftime("%m%d%H%M%S")
        overhead_camera = make_overhead_camera(stage_cfg)
        recorder = EpisodeRecorder(
            out_dir=video_dir,
            # 動画1フレーム=env.step()1回(decimation物理サブステップぶんの時間経過)なので、
            # fpsはrendering_dtではなくenv.step()の呼び出し頻度に合わせる
            fps=1.0 / (env_cfg.physics_dt * env_cfg.decimation),
            robot_resolution=env_cfg.camera_resolution,
            overhead_camera=overhead_camera,
        )
        print(f"[test] video: enabled -> {video_dir}/")
        if overhead_camera is not None:
            print("[test] overhead camera: enabled")

    stats = evaluate(
        env,
        agent,
        test_cfg.episodes_per_stage,
        recorder=recorder,
        video_episodes=test_cfg.episodes_per_stage if args.video else 0,
    )

    separator = "=" * 60
    tqdm.write(f"\n{separator}", file=_OUT)
    tqdm.write(f"Stage {args.stage_index}: {stage_cfg.stage_path}", file=_OUT)
    tqdm.write(separator, file=_OUT)
    for line in stats.report_lines():
        tqdm.write(line, file=_OUT)
    tqdm.write(f"{separator}\n", file=_OUT)

    env.close()
    app.close()


if __name__ == "__main__":
    main()
