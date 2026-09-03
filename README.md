# Isaac-GS

[Recon-GS](https://github.com/AoiNoGeso/Recon-GS) によって生成した 3D Gaussian Splatting (.ply) と床・壁メッシュ (.ply) を Isaac Sim 6.0.1 で使用し，ビジョンベース強化学習 (Point Navigation / Social Navigation) を実行するためのパイプラインです

## 概要

```
Recon-GS による 3D 再構成
        ↓
gaussian.ply + floor.ply + wall.ply
        ↓
stage_generation convert-gs     →  gs.usdc        (視覚: 3DGS スプラット)
stage_generation convert-mesh   →  floor_mesh.usd (物理: 床コライダ)
stage_generation convert-mesh   →  wall_mesh.usd  (物理: 壁コライダ)
        ↓
stage_generation compose        →  stage.usda  (CollisionAPI + NavMeshVolume)
        ↓
RL 学習 (scripts/train.py)  ← 起動時に NavMesh をランタイム bake
```

生成される `stage.usda` は 3DGS を視覚表現とし，床・壁メッシュを不可視コライダとして重ねることで，リアルな見た目と正確な物理コリジョンを両立します

`--num-humans` を指定すると ORCA(衝突回避)+ ai4animationpy(歩行モーション生成)による歩行者を注入し，Social Navigation(人物回避を伴うナビゲーション)としても学習できます

## 前提条件

- uv がインストールされていること
- CUDA 対応の GPU 環境(RTX 4090 で動作確認)
- [Recon-GS](https://github.com/AoiNoGeso/Recon-GS) で生成した `gaussian.ply`・`floor.ply`・`wall.ply`

## ディレクトリ構成

```
Isaac-GS/
├── setup.sh                    # 環境構築スクリプト
├── pyproject.toml              # Isaac-GS 追加依存パッケージ管理
├── submodules/
│   └── ai4animationpy/          # git submodule 歩行モデル本体・重み・Guidancesの取得元
├── stage_generation/            # 統合 CLI: python -m stage_generation {convert-gs,convert-mesh,compose}
│   ├── __main__.py              # argparse エントリポイント(サブコマンド定義)
│   ├── convert_gs.py            # GS (.ply) → .usdc 変換
│   ├── convert_mesh.py          # メッシュ (.ply) → .usd 変換
│   ├── compose.py                # gs.usdc + floor/wall → stage.usda 合成
│   └── coords.py                 # source-up-axis → Z-up 座標変換(共通)
├── assets/
│   ├── stages/                    # ステージファイル置き場
│   │   ├── room1/stage.usda
│   │   ├── corridor1/stage.usda
│   │   └── corridor2/stage.usda
│   └── avatars/                   # 人物アバターUSD置き場(utils/convert_glb2usd.pyの出力先)
├── envs/
│   ├── config.py                 # EnvConfig / RobotConfig(+JACKAL)
│   ├── stage_config.py           # StagePreset / STAGE_PRESETS / get_preset / stage_names
│   ├── geometry.py               # quat_to_yaw / goal_vec(isaacsim 非依存、env と deploy で共用)
│   ├── isaac_env.py              # IsaacSim 環境コア (PointNavIsaacEnv) + gymnasium ラッパー (PointNavGymEnv)
│   ├── human_controller/         # 人物キャラの衝突回避・歩行モーション・USD配置
│   │   ├── base.py               # CrowdController(Protocol) — 衝突回避アルゴリズムの共通インターフェース
│   │   ├── human_manager.py      # HumanManager — CrowdController + LocomotionPool を接続するマネージャ
│   │   ├── locomotion.py         # LocomotionPool/LocomotionAgent — ai4animationpyによる歩行モーション生成
│   │   └── controllers/orca/     # ORCASimulator(RVO2ラッパー)
│   └── sensors/
│       └── camera_sensor.py      # RGB カメラセンサ (RGBCamera)
├── models/
│   └── sac/                      # 自前実装 SAC(参照実装)
│       ├── config.py             # ModelConfig / TrainConfig / TestConfig / replay_buffer_spec
│       ├── network.py            # エンコーダ (make_encoder)
│       ├── policy.py             # ReplayBuffer / Actor / Critic / SACAgent
│       ├── train.py              # 学習スクリプト
│       └── test.py               # 評価スクリプト
│   # 外部モデル(NoMaD, NavDP 等)は models/<Name>/ に git clone して配置する
├── utils/
│   ├── metrics.py                 # EpisodeTracker / EvalStats(SPL 計算・wandb ログ、train/test 共用)
│   ├── rollout.py                 # evaluate() — 評価ロールアウトの唯一の実装(train/test共用)
│   ├── recorder.py                # EpisodeRecorder / make_overhead_camera — 動画収録
│   ├── launch_sim.py              # launch_sim() — SimulationApp起動の共通処理
│   └── convert_glb2usd.py         # glTF(.glb) → USD 変換ツール(人物アバター用)
├── deploy/
│   └── deploy.py                 # 実機 policy 推論ノード (ROS2)
├── debug/
│   └── teleop.py                  # WASD テレオペ
└── runs/                        # 学習ログ・チェックポイント
```

## 環境構築

```bash
git clone --recursive https://github.com/AoiNoGeso/Isaac-GS.git
cd Isaac-GS
```

### 1. システム依存ライブラリ

```bash
sudo apt install python3.12-dev libgl1-mesa-dev libx11-dev \
    libxcursor-dev libxi-dev libxinerama-dev libxrandr-dev libportaudio2
```

### 2. Python 環境のセットアップ

`setup.sh` でインストールします

```bash
zsh setup.sh
```

`setup.sh` は以下を順番に実行します：

0. `git submodule update --init --recursive` で [ai4animationpy](https://github.com/facebookresearch/ai4animationpy.git) submodule(`submodules/ai4animationpy`)を取得
1. リポジトリ直下に `.venv` (Python 3.12) 仮想環境を作成
2. IsaacSim 6.0.1 をインストール(CUDA対応のtorch/torchvisionが同梱される)
3. `pyproject.toml` のパッケージ (gymnasium / wandb / Pillow / pydantic / opencv-python) をインストール
4. `submodules/ai4animationpy`(歩行モーション生成)をインストール
5. [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2.git)をCython経由でソースビルドインストール

歩行モデルの重み(`Network.pt`/`PostProcessor.pt`)・アバターメッシュ(`Model.glb`)・ガイダンステンプレートは、いずれもIsaac-GS側に複製せず `submodules/ai4animationpy/Demos/` 配下を直接参照します(`envs/human_controller/locomotion.py`)

## パイプライン実行手順

### 1. 3DGS (.ply) → .usdc

```bash
uv run python -m stage_generation convert-gs \
    -i path/to/gaussian.ply \
    -o assets/stages/corridor1
```

### 2. 床・壁メッシュ (.ply) → .usd

```bash
uv run python -m stage_generation convert-mesh \
    -i path/to/floor.ply \
    -o assets/stages/corridor1/floor_mesh.usd

uv run python -m stage_generation convert-mesh \
    -i path/to/wall.ply \
    -o assets/stages/corridor1/wall_mesh.usd
```

### 3. stage.usda の合成

```bash
uv run python -m stage_generation compose -i assets/stages/corridor1
```

出力: `assets/stages/corridor1/stage.usda`(NavMesh自体は学習/テスト実行時にランタイムで自動bakeされる)

### 人物アバター (.glb) → .usd

Social Navigationで使う人物アバターのUSDは`utils/convert_glb2usd.py`で変換します

```bash
uv run utils/convert_glb2usd.py \
    --i submodules/ai4animationpy/Demos/_ASSETS_/Geno/Model.glb \
    --o assets/avatars/Model.usd
```

`envs/human_controller/human_manager.py`は`assets/avatars/Model.usd`を読み込む前提のため、`--num-humans > 0`で学習/テスト/デバッグを行う前に一度だけ実行しておく必要があります(未変換の場合はエラーメッセージに変換コマンドが表示されます)。`--i`/`--o`は任意の`.glb`/`.usd`パスを指定できる汎用ツールなので、別のアバターメッシュに差し替える場合も同様に使えます

### 4. ステージ登録

新しいステージを使う場合は `envs/stage_config.py` の `STAGE_PRESETS` にスポーン/ゴール座標を登録します

### 5. ナビゲーションタスク学習

学習設定は `models/sac/config.py` の `TrainConfig`(`stage`/`run_name`/`log_dir` は実験ごとに書き換える3点セット)、エンコーダ入力設定は同ファイルの `ModelConfig` で編集します

```bash
# 学習開始
uv run scripts/train.py --model sac --headless

# 人物キャラの数を指定(Social Navigation)
uv run scripts/train.py --model sac --headless --num-humans 2

# チェックポイントから再開
uv run scripts/train.py --model sac --headless \
    --checkpoint runs/SocialNav-RGB+Goal/corridor2/checkpoints/sac_10000.pt

# wandb なし
uv run scripts/train.py --model sac --headless --no-wandb

# DINOv3視覚エンコーダ版
uv run scripts/train.py --model sac_DINOv3 --headless
```

学習中は `TrainConfig.val_interval` ごとにgreedy方策でのバリデーションを実行し、成功率/衝突率/タイムアウト率をログします。`val_video_episodes`を設定すると、先頭N件のバリデーションエピソードの動画をheadlessでも `{log_dir}/val_videos/robot/`(ロボット視点)・`{log_dir}/val_videos/overhead/`(俯瞰視点、ステージに`overhead_camera_translation`が設定されている場合のみ)にローカル保存します(wandbへは送信しません)。ファイル名は `{step}_{episode}_{終了理由タグ}.mp4`(タグは成功=`s`／人物衝突=`h`／壁衝突=`w`／タイムアウト=`t`)です

## テスト

学習済みモデルを評価します。評価エピソード数は `models/sac/config.py` の `TestConfig` で定義します

```bash
# 単一ステージ(index 0)
uv run scripts/test.py --model sac \
    --checkpoint runs/SocialNav-RGB+Goal/corridor2/sac_final.pt \
    --log-dir runs/SocialNav-RGB+Goal/corridor2/test \
    --stage-index 0

# 各エピソードの映像を保存(俯瞰カメラも設定されていれば同時保存)
uv run scripts/test.py --model sac --checkpoint <path> --log-dir <dir> --stage-index 2 --video

# チェックポイントディレクトリを丸ごと掃引(DINOv3視覚エンコーダ版)
uv run scripts/test.py --model sac_DINOv3 --checkpoint <dir> --log-dir <dir> --headless
```

| 引数 | 説明 |
|---|---|
| `--model` | チェックポイントパス(必須) |
| `--stage-index` | 評価するステージのインデックス(デフォルト 0) |
| `--num-humans` | 人物キャラの数(デフォルト 0) |
| `--headless` | ヘッドレス実行 |
| `--video` | 各エピソードの映像をmp4で保存する |
| `--video-dir` | 動画の出力先ディレクトリ(デフォルト `videos`) |

出力例:
```
============================================================
Stage 0: assets/stages/room1/stage.usda
============================================================
Episodes      : 100
Success Rate  : 0.720  (72/100)
Collision Rate: 0.150  (15/100)
Wall Collision Rate : 0.100  (10/100)
Human Collision Rate: 0.050  (5/100)
Timeout Rate  : 0.130  (13/100)
Avg Reward    : 45.32
Avg Dist Final: 0.381 m
SPL           : 0.613
============================================================
```

### 俯瞰カメラの設定

`--video` 時にロボット視点に加えて俯瞰視点も撮影したい場合は、`envs/stage_config.py` の該当ステージに `overhead_camera_translation`/`overhead_camera_orientation` を設定してください。座標はIsaac SimのGUIでPerspectiveビューポートを希望のアングルに操作し、Script Editorで以下を実行して取得できます。

```python
from omni.kit.viewport.utility import get_active_viewport
from omni.kit.viewport.utility.camera_state import ViewportCameraState

viewport = get_active_viewport()
cam_state = ViewportCameraState(viewport.camera_path, viewport)
pos = cam_state.position_world
quat = cam_state.usd_camera.ComputeLocalToWorldTransform(0).ExtractRotationQuat()
print(pos, quat.GetReal(), quat.GetImaginary())
```

## デバッグ

```bash
uv run debug/teleop.py
uv run debug/teleop.py --num-humans 2             # ORCA+ai4animationpyの人物キャラを注入して衝突判定を確認
uv run debug/teleop.py --stage corridor1 --reset  # 衝突/成功/タイムアウトで自動リセット
uv run debug/teleop.py --vis-goal                 # スポーン(青)・ゴール(赤)地点に半透明の円を表示
```

| キー | 動作 |
|---|---|
| W / S | 前進 / 後退 |
| A / D | 左回転 / 右回転 |
| P | 現在のワールド座標を表示(人物ありの場合は各人物の座標も) |
| R | 手動リセット |
| Q | 終了 |

## システム構成

| 項目 | 内容 |
|---|---|
| シミュレータ | Isaac Sim 6.0.1 |
| RL アルゴリズム | 自前実装 SAC (Soft Actor-Critic) |
| ロボット | Clearpath Jackal (4輪スキッドステア) |
| 観測 | 環境は常に RGB 84×84 px・ゴールベクトル (2,) を発行。どれをエンコーダ入力に使うかは `models/sac/config.py` の `ModelConfig` で選択 |
| 行動 | [v_x_norm, ω_z_norm](`RobotConfig` の `wheel_base`/`v_linear_max`/`v_angular_max` を介しロボット非依存に左右輪速度へ変換) |
| 人物の衝突回避 | ORCA(RVO2)。ロボットもagentとして登録され人物と1:1のreciprocal avoidance対象になる(`RobotConfig.footprint_radius`) |
| 壁衝突判定 | ContactSensor (PhysX) による wall_mesh 接触検出 |
| 人物衝突判定 | ロボット-人物間の距離ベース判定(`human_collision_dist`) |
| 衝突優先順位 | 壁衝突と人物衝突が同時発生した場合は人物衝突を優先(ゴール到達時のみ上書きしない) |
| 座標系 | Z-up、ロボット前方 = +X |

外部のナビゲーションモデル(NoMaD, NavDP 等)を試す場合は `models/<Name>/` に git clone して配置してください

## 注意事項

- `stage_generation/convert_gs.py` の GSPLAT 変換パス(`GSPLAT_DIR` / `USD_LIBS`)は `.venv` 内から自動検出されます。自動検出できない場合は環境変数で上書きしてください (以下のコマンドで検索できます)
  ```bash
  find .venv -type d -name "omni.kit.converter.gsplat-*"
  find .venv -type d -name "omni.usd.libs-*"
  export GSPLAT_DIR=...
  export USD_LIBS=...
  ```
