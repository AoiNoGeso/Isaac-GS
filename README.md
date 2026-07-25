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
RL 学習 (models/sac/train.py)  ← 起動時に NavMesh をランタイム bake
```

生成される `stage.usda` は 3DGS を視覚表現とし，床・壁メッシュを不可視コライダとして重ねることで，リアルな見た目と正確な物理コリジョンを両立します

`--num-humans` を指定すると IRA (isaacsim.replicator.agent) による歩行者を注入し，Social Navigation（人物回避を伴うナビゲーション）としても学習できます

## 前提条件

- uv がインストールされていること
- CUDA 対応の GPU 環境（RTX 4090 で動作確認）
- [Recon-GS](https://github.com/AoiNoGeso/Recon-GS) で生成した `gaussian.ply`・`floor.ply`・`wall.ply`

## ディレクトリ構成

```
Isaac-GS/
├── setup.sh                    # 環境構築スクリプト
├── pyproject.toml              # Isaac-GS 追加依存パッケージ管理
├── stage_generation/            # 統合 CLI: python -m stage_generation {convert-gs,convert-mesh,compose}
│   ├── __main__.py              # argparse エントリポイント（サブコマンド定義）
│   ├── convert_gs.py            # GS (.ply) → .usdc 変換
│   ├── convert_mesh.py          # メッシュ (.ply) → .usd 変換
│   ├── compose.py                # gs.usdc + floor/wall → stage.usda 合成
│   └── coords.py                 # source-up-axis → Z-up 座標変換（共通）
├── stages/                       # ステージファイル置き場
│   ├── room1/stage.usda
│   ├── corridor1/stage.usda
│   └── corridor2/stage.usda
├── envs/
│   ├── config.py                 # EnvConfig / RobotConfig(+JACKAL)
│   ├── stage_config.py           # StagePreset / STAGE_PRESETS / get_preset / stage_names
│   ├── geometry.py               # quat_to_yaw / goal_vec（isaacsim 非依存、env と deploy で共用）
│   ├── isaac_env.py              # IsaacSim 環境コア (PointNavIsaacEnv) + gymnasium ラッパー (PointNavGymEnv)
│   ├── humans/
│   │   └── ira.py                # IRA人物キャラの注入・NavMeshサンプリング・再配置 (IRAHumanManager)
│   └── sensors/
│       └── camera_sensor.py      # RGB+Depth カメラセンサ (RGBDCamera)
├── models/
│   └── sac/                      # 自前実装 SAC（参照実装）
│       ├── config.py             # ModelConfig / TrainConfig / TestConfig / replay_buffer_spec
│       ├── network.py            # エンコーダ (make_encoder)
│       ├── policy.py             # ReplayBuffer / Actor / Critic / SACAgent
│       ├── train.py              # 学習スクリプト
│       └── test.py               # 評価スクリプト
│   # 外部モデル（NoMaD, NavDP 等）は models/<Name>/ に git clone して配置する
├── utils/
│   ├── metrics.py                 # EpisodeTracker（SPL 計算・wandb ログ、train/test 共用）
│   └── video.py                   # 動画書き出し・俯瞰カメラ生成の共用ヘルパー
├── tests/                          # pytestテスト (isaacsim非依存の部分のみ)
├── deploy/
│   └── deploy.py                 # 実機 policy 推論ノード (ROS2)
├── debug/
│   └── teleop.py                 # WASD テレオペ
└── runs/                        # 学習ログ・チェックポイント
```

## 環境構築

```bash
git clone https://github.com/AoiNoGeso/Isaac-GS.git
cd Isaac-GS
```

### 1. システム依存ライブラリ

```bash
sudo apt install python3.12-dev libgl1-mesa-dev libx11-dev \
    libxcursor-dev libxi-dev libxinerama-dev libxrandr-dev
```

### 2. Python 環境のセットアップ

`setup.sh` で順番にインストールします

```bash
zsh setup.sh
```

`setup.sh` は以下を順番に実行します：

1. リポジトリ直下に `.venv` (Python 3.12) 仮想環境を作成
2. IsaacSim 6.0.1 をインストール（CUDA対応のtorch/torchvisionが同梱される）
3. `pyproject.toml` のパッケージ (gymnasium / wandb / Pillow / pydantic / opencv-python) をインストール

## パイプライン実行手順

### 1. 3DGS (.ply) → .usdc

```bash
uv run python -m stage_generation convert-gs \
    -i path/to/gaussian.ply \
    -o stages/corridor1
```

`--source-up-axis {Y}`（既定 `Y` = Recon-GS の -Y-up 出力、現状 `Y` のみ対応）で変換元データの up-axis を指定できます

### 2. 床・壁メッシュ (.ply) → .usd

```bash
uv run python -m stage_generation convert-mesh \
    -i path/to/floor.ply \
    -o stages/corridor1/floor_mesh.usd

uv run python -m stage_generation convert-mesh \
    -i path/to/wall.ply \
    -o stages/corridor1/wall_mesh.usd
```

### 3. stage.usda の合成

```bash
uv run python -m stage_generation compose -i stages/corridor1
```

出力: `stages/corridor1/stage.usda`（NavMesh自体は学習/テスト実行時にランタイムで自動bakeされる）

### 4. ステージ登録

新しいステージを使う場合は `envs/stage_config.py` の `STAGE_PRESETS` にスポーン/ゴール座標を登録します

### 5. ナビゲーションタスク学習

学習設定は `models/sac/config.py` の `TrainConfig`（`stage`/`run_name`/`log_dir` は実験ごとに書き換える3点セット）、エンコーダ入力設定は同ファイルの `ModelConfig` で編集します

```bash
# 学習開始
uv run models/sac/train.py

# 人物キャラの数を指定（Social Navigation）
uv run models/sac/train.py --num-humans 2

# チェックポイントから再開
uv run models/sac/train.py \
    --checkpoint runs/SocialNav-RGB+Goal/corridor2/checkpoints/sac_10000.pt

# wandb なし
uv run models/sac/train.py --no-wandb
```

学習中は `TrainConfig.val_interval` ごとにgreedy方策でのバリデーションを実行し、成功率/衝突率/タイムアウト率をログします。`val_video_episodes`（既定1）を設定すると、先頭N件のバリデーションエピソードの動画（ロボット視点・俯瞰視点）を `{log_dir}/val_videos/` にローカル保存します（wandbへは送信しません）

## テスト

学習済みモデルを評価します。評価エピソード数は `models/sac/config.py` の `TestConfig` で定義します

```bash
# 単一ステージ（index 0）
uv run models/sac/test.py \
    --model runs/SocialNav-RGB+Goal/corridor2/sac_final.pt \
    --stage-index 0

# 各エピソードの映像を保存（俯瞰カメラも設定されていれば同時保存）
uv run models/sac/test.py --model <path> --stage-index 2 --video
```

| 引数 | 説明 |
|---|---|
| `--model` | チェックポイントパス（必須） |
| `--stage-index` | 評価するステージのインデックス（デフォルト 0） |
| `--num-humans` | 人物キャラの数（デフォルト 0） |
| `--headless` | ヘッドレス実行 |
| `--video` | 各エピソードの映像をmp4で保存する |
| `--video-dir` | 動画の出力先ディレクトリ（デフォルト `videos`） |

出力例:
```
============================================================
Stage 0: stages/room1/stage.usda
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
uv run debug/teleop.py --num-humans 2            # IRA 人物キャラを注入して衝突判定を確認
uv run debug/teleop.py --stage corridor1 --reset  # 衝突/成功/タイムアウトで自動リセット
```

| キー | 動作 |
|---|---|
| W / S | 前進 / 後退 |
| A / D | 左回転 / 右回転 |
| P | 現在のワールド座標を表示（人物ありの場合は各人物の座標も） |
| R | 手動リセット |
| Q | 終了 |

## テストコード

isaacsim非依存の部分（`envs.geometry`/`EpisodeTracker`/`ReplayBuffer`/`SACAgent`のsave-load等）は`tests/`にpytestを用意しています

```bash
uv run python -m pytest tests -q
```

## システム構成

| 項目 | 内容 |
|---|---|
| シミュレータ | Isaac Sim 6.0.1 |
| RL アルゴリズム | 自前実装 SAC (Soft Actor-Critic) |
| ロボット | Clearpath Jackal (4輪スキッドステア) |
| 観測 | 環境は常に RGB 84×84 px・Depth・ゴールベクトル (2,) を発行。どれをエンコーダ入力に使うかは `models/sac/config.py` の `ModelConfig` で選択 |
| 行動 | [v_x_norm, ω_z_norm]（`RobotConfig` の `wheel_base`/`v_linear_max`/`v_angular_max` を介しロボット非依存に左右輪速度へ変換） |
| 壁衝突判定 | ContactSensor (PhysX) による wall_mesh 接触検出 |
| 人物衝突判定 | ContactSensorに加え、ロボット-人物間の距離ベース判定を併用（`human_collision_dist`） |
| 衝突優先順位 | 壁衝突と人物衝突が同時発生した場合は人物衝突を優先（ゴール到達時のみ上書きしない） |
| 座標系 | Z-up、ロボット前方 = +X |

外部のナビゲーションモデル（NoMaD, NavDP 等）を試す場合は `models/<Name>/` に git clone して配置してください

## 注意事項

- `stage_generation/convert_gs.py` の GSPLAT 変換パス（`GSPLAT_DIR` / `USD_LIBS`）は `.venv` 内から自動検出されます。自動検出できない場合は環境変数で上書きしてください (以下のコマンドで検索できます)
  ```bash
  find .venv -type d -name "omni.kit.converter.gsplat-*"
  find .venv -type d -name "omni.usd.libs-*"
  export GSPLAT_DIR=...
  export USD_LIBS=...
  ```
