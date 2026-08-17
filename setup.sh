#!/usr/bin/env bash
# Isaac-GS 環境セットアップスクリプト

set -e

ENV_DIR=".venv"
AI4ANIM_DIR="submodules/ai4animationpy"

# 0. submodule取得（ai4animationpy本体・歩行モデル重み・Guidancesをここから直接参照する）
git submodule update --init --recursive

# 1. 仮想環境作成
uv venv --python 3.12 --seed "$ENV_DIR"
source "$ENV_DIR/bin/activate"

# 2. IsaacSim インストール
uv pip install \
    "isaacsim[all,extscache]==6.0.1.0" \
    --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match \
    --prerelease=allow

# 3. Isaac-GS 追加依存インストール
uv pip install -e .

# 4. ai4animationpy インストール（歩行アニメーション生成、docs/ai4animation-orca-plan.md参照）
uv pip install -e "$AI4ANIM_DIR" --no-deps

# pygltflib は ai4animationpy の必須依存
uv pip install pygltflib==1.16.5

# 5. rvo2 (Python-RVO2) インストール（ORCAによる人物衝突回避、--num-humans使用時に必要）
# PyPI配布が無くソースビルドが必要なため、事前にCythonをインストールしておく
uv pip install Cython
uv run python -m pip install --no-build-isolation "git+https://github.com/sybrenstuvel/Python-RVO2.git"

echo ""
echo "セットアップ完了！"
