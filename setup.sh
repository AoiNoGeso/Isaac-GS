#!/usr/bin/env bash
# Isaac-GS 環境セットアップスクリプト

set -e

ENV_DIR=".venv"

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
uv pip install -e ~/Programs/ai4animationpy --no-deps

# pygltflib は ai4animationpy の必須依存
uv pip install pygltflib==1.16.5

echo ""
echo "セットアップ完了！"
