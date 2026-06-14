# Isaac-GS

[Recon-GS](https://github.com/AoiNoGeso/Recon-GS) によって生成した 3D Gaussian Splatting (.ply) と床・壁メッシュ (.ply) を Isaac Sim 6.0 で使用し，ビジョンベース強化学習（Point Navigation）を実行するためのパイプラインです．

## 概要

```
Recon-GS による 3D 再構成
        ↓
gaussian.ply + floor.ply + wall.ply
        ↓
convert_gs.py     →  gs.usdc        （視覚: 3DGS スプラット）
convert_mesh.py   →  floor_mesh.usd （物理: 床コライダ）
convert_mesh.py   →  wall_mesh.usd  （物理: 壁コライダ・衝突判定対象）
        ↓
compose_stage.py  →  stage.usda  （CollisionAPI + NavMeshVolume 配置済み）
        ↓
Isaac Sim GUI で NavMesh Bake → stage.usda に保存
        ↓
RL 学習 (tasks/point_navigation/train.py)
```

生成される `stage.usda` は 3DGS を視覚表現とし，床・壁メッシュを不可視コライダとして重ねることで，リアルな見た目と正確な物理コリジョンを両立します．  
強化学習タスクでは RGB-D カメラ観測と PPO を用いて，壁を避けながらゴールへ到達するエージェントを学習します．

## 前提条件

- Isaac Sim 6.0（`~/env_isaaclab` に uv で管理）
- [Recon-GS](https://github.com/AoiNoGeso/Recon-GS) で生成した `gaussian.ply`・`floor.ply`・`wall.ply`
- `uv` がインストール済みであること

## ディレクトリ構成

```
Isaac-GS/
├── stage_generation/
│   ├── convert_gs.py       # GS (.ply) → .usdc 変換・回転補正
│   ├── convert_mesh.py     # メッシュ (.ply) → .usd 変換（汎用）
│   └── compose_stage.py    # gs.usdc + floor/wall → stage.usda 合成・NavMeshVolume 配置
├── tasks/
│   └── point_navigation/
│       ├── __init__.py
│       ├── train.py            # PPO 学習スクリプト
│       ├── check_env_v2.py     # Step 1〜5 個別動作確認
│       ├── env/
│       │   ├── isaac_env.py    # IsaacSim 環境コア（World / Articulation / PhysX）
│       │   ├── camera_sensor.py# RGB-D カメラセンサ（isaacsim.sensors.camera）
│       │   └── gym_wrapper.py  # gymnasium.Env ラッパー（skrl 用）
│       └── policy/
│           └── cnn_encoder.py  # CNN エンコーダ + Actor/Critic（skrl 2.x）
├── docs/
│   └── spec.md             # 詳細仕様書
└── sample_data/            # サンプルデータ（.gitignore 対象）
    └── stages/
        └── corridor1/
            ├── gs.usdc
            ├── floor_mesh.usd
            ├── wall_mesh.usd
            └── stage.usda
```

## パイプライン実行手順

> **すべてのスクリプトは `uv run` で実行します．事前に以下で環境を有効化してください．**
>
> ```bash
> source ~/env_isaaclab/bin/activate
> ```

### 1. 3DGS (.ply) → .usdc

```bash
uv run stage_generation/convert_gs.py \
    -i path/to/gaussian.ply \
    -o sample_data/stages/corridor1
```

出力: `sample_data/stages/corridor1/gs.usdc`

### 2. 床メッシュ (.ply) → .usd

```bash
uv run stage_generation/convert_mesh.py \
    -i path/to/floor.ply \
    -o sample_data/stages/corridor1/floor_mesh.usd
```

### 3. 壁メッシュ (.ply) → .usd

```bash
uv run stage_generation/convert_mesh.py \
    -i path/to/wall.ply \
    -o sample_data/stages/corridor1/wall_mesh.usd
```

### 4. stage.usda の合成（CollisionAPI + NavMeshVolume 自動配置）

```bash
uv run stage_generation/compose_stage.py \
    -i sample_data/stages/corridor1
```

出力: `sample_data/stages/corridor1/stage.usda`

### 5. NavMesh Bake（GUI で手動実施）

> NavMesh Bake は Isaac Sim 6.0 のスタンドアロン API からは実行できないため，GUI で行います．

1. Isaac Sim を起動し `sample_data/stages/corridor1/stage.usda` を開く
2. `Window > Navigation > NavMesh` パネルを開く
3. **Bake** ボタンを押す（NavMeshVolume は自動配置済み）
4. 床面上に青いオーバーレイが表示されることを確認
5. `File > Save` で保存

## 動作確認（Step 1〜5）

各 Step を順番に確認します．

```bash
# Step 1: World 起動 + stage ロード + Carter スポーン
uv run tasks/point_navigation/check_env_v2.py --step 1

# Step 2: RGB-D カメラ画像を debug_images/ に PNG 保存
uv run tasks/point_navigation/check_env_v2.py --step 2

# Step 3: gymnasium API バリデーション（manual）
uv run tasks/point_navigation/check_env_v2.py --step 3

# Step 4: CNN policy フォワードパス確認
uv run tasks/point_navigation/check_env_v2.py --step 4 --headless

# Step 5: PPO 10 イテレーション動作確認
uv run tasks/point_navigation/check_env_v2.py --step 5 --headless
```

## RL 学習

```bash
# ヘッドレス学習（推奨）
uv run tasks/point_navigation/train.py --headless

# GUI 付き学習
uv run tasks/point_navigation/train.py

# チェックポイントから再開
uv run tasks/point_navigation/train.py \
    --headless --checkpoint runs/point_nav/checkpoints/best.pt
```

## 出力ステージの Prim 構成

```
/World                   （Xform, defaultPrim）
└── /World/env
    ├── /World/env/gs         （3DGS スプラット参照, visible）
    ├── /World/env/floor_mesh （床コライダ, invisible）
    └── /World/env/wall_mesh  （壁コライダ, invisible, 衝突判定対象）
/World/PhysicsScene      （PhysicsScene, gravity Y-down）
/World/NavMeshVolume     （floor_mesh AABB から自動配置）
```

| Prim | 役割 | Visible | コリジョン |
|---|---|---|---|
| `env/gs` | 視覚（3DGS スプラット） | ✓ | なし |
| `env/floor_mesh` | 床コライダ | ✗ | CollisionAPI / 近似: none |
| `env/wall_mesh` | 壁コライダ・衝突検知 | ✗ | CollisionAPI / 近似: none |

ステージ設定: `upAxis = Y`，`metersPerUnit = 1.0`

## システム構成

| 項目 | 内容 |
|---|---|
| シミュレータ | Isaac Sim 6.0 |
| RL フレームワーク | skrl 2.x（PPO） |
| ロボット | Carter V1（差動二輪） |
| 観測 | RGB 84×84 px + goal ベクトル (2,) |
| CNN エンコーダ | Conv(3→32)→Conv(32→64)→Conv(64→64)→FC(256) + GoalEnc FC(2→32) |
| 衝突判定 | PhysX contact event callback（wall_mesh のみ） |
| NavMesh | omni.anim.navigation.core（Bake は GUI で実施） |

## 注意事項

- `convert_gs.py` 内の `GSPLAT_DIR` / `USD_LIBS` パスは環境に合わせて修正してください．
- `compose_stage.py` は `gs.usdc`・`floor_mesh.usd`・`wall_mesh.usd` の 3 ファイルが揃っていない場合エラーで終了します．
- NavMesh Bake はスタンドアロン API からは動作しないため，必ず GUI で実施してください．
- `PointNavEnvCfg.stage_path` のデフォルトは絶対パスです．環境に合わせて変更してください．
