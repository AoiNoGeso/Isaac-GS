# Isaac-GS

[Recon-GS](https://github.com/AoiNoGeso/Recon-GS) によって生成した 3D Gaussian Splatting (.ply) と床・壁メッシュ (.ply) を Isaac Sim 6.0 で使用し，ビジョンベース強化学習(Point Navigation)を実行するためのパイプラインです

## 概要

```
Recon-GS による 3D 再構成
        ↓
gaussian.ply + floor.ply + wall.ply
        ↓
convert_gs.py     →  gs.usdc        (視覚: 3DGS スプラット)
convert_mesh.py   →  floor_mesh.usd (物理: 床コライダ)
convert_mesh.py   →  wall_mesh.usd  (物理: 壁コライダ)
        ↓
compose_stage.py  →  stage.usda  (CollisionAPI + NavMeshVolume)
        ↓
RL 学習 (tasks/point_navigation/train.py)  ← 起動時に NavMesh をランタイム bake
```

生成される `stage.usda` は 3DGS を視覚表現とし，床・壁メッシュを不可視コライダとして重ねることで，リアルな見た目と正確な物理コリジョンを両立します

## 前提条件

  - Isaac Sim 6.0 は [公式ガイド](https://isaac-sim.github.io/IsaacLab/develop/source/setup/installation/pip_installation.html) に従って `~/env_isaaclab` にインストールしておいてください (uv 推奨/IsaacLabは未インストールでもOK)
  - [Recon-GS](https://github.com/AoiNoGeso/Recon-GS) で生成した `gaussian.ply`・`floor.ply`・`wall.ply`

## ディレクトリ構成

```
Isaac-GS/
├── stage_generation/
│   ├── convert_gs.py       # GS (.ply) → .usdc 変換
│   ├── convert_mesh.py     # メッシュ (.ply) → .usd 変換
│   └── compose_stage.py    # gs.usdc + floor/wall → stage.usda 合成
├── stages/                 # ステージファイル置き場
│   └── corridor1/
│       └── stage.usda
├── envs/                   # 環境スクリプト
│   ├── isaac_env.py        # IsaacSim 環境コア
│   ├── gym_wrapper.py      # gymnasium.Env ラッパー
│   └── sensors/            # センサースクリプト
│       └── camera_sensor.py# RGB カメラセンサ
├── tasks/
│   └── point_navigation/
│       ├── config.py       # 環境・報酬・アルゴリズム設定
│       ├── train.py        # 学習スクリプト
│       └── policy/
│           └── network.py  # CNN エンコーダ + Actor/Critic
├── debug/                  # デバッグ用スクリプト
│   └── teleop.py           # WASD テレオペ
└── runs/                   # 学習ログ・チェックポイント
    └── point_nav/
        └── wandb/
```
## 環境構築

```bash
git clone https://github.com/AoiNoGeso/Isaac-GS.git
cd Isaac-GS
```

IsaacSim 環境の有効化
```bash
source ~/env_isaaclab/bin/activate
```

## パイプライン実行手順

### 1. 3DGS (.ply) → .usdc

```bash
uv run stage_generation/convert_gs.py \
    -i path/to/gaussian.ply \
    -o stages/corridor1
```

### 2. 床・壁メッシュ (.ply) → .usd

```bash
uv run stage_generation/convert_mesh.py \
    -i path/to/floor.ply \
    -o stages/corridor1/floor_mesh.usd

uv run stage_generation/convert_mesh.py \
    -i path/to/wall.ply \
    -o stages/corridor1/wall_mesh.usd
```

### 3. stage.usda の合成

```bash
uv run stage_generation/compose_stage.py -i stages/corridor1
```

出力: `stages/corridor1/stage.usda`

### 4. RL 学習

学習設定は `tasks/point_navigation/config.py` で編集します

```bash
# PPO (デフォルト)
uv run tasks/point_navigation/train.py --headless

# SAC
uv run tasks/point_navigation/train.py --headless --algo sac

# 実験名を指定
uv run tasks/point_navigation/train.py --headless --run-name my_run

# チェックポイントから再開
uv run tasks/point_navigation/train.py \
    --headless --checkpoint runs/point_nav/checkpoints/best.pt

# wandb なし
uv run tasks/point_navigation/train.py --headless --no-wandb
```

## デバッグ

```bash
# WASD テレオペ
uv run debug/teleop.py
```

| キー | 動作 |
|---|---|
| W / S | 前進 / 後退 |
| A / D | 左回転 / 右回転 |
| P | 現在のワールド座標を表示 |
| Q | 終了 |

## システム構成

| 項目 | 内容 |
|---|---|
| シミュレータ | Isaac Sim 6.0 |
| RL フレームワーク | skrl 2.x |
| ロボット | Carter V1 (差動二輪) |
| 観測 | RGB 84×84 px + goal ベクトル (2,) |
| 行動 | [v_x_norm, ω_z_norm](ユニサイクルモデル) |
| 衝突判定 | ContactSensor(Newtonエンジン)による wall_mesh 接触検出 |
| 座標系 | Z-up|

## 注意事項

- `convert_gs.py` 内の `GSPLAT_DIR` / `USD_LIBS` パスは環境に合わせて修正してください (以下のコマンドで検索できます)
  ```bash
  find ~/env_isaaclab -type d -name "omni.kit.converter.gsplat-*"
  find ~/env_isaaclab -type d -name "omni.usd.libs-*"
  ```
