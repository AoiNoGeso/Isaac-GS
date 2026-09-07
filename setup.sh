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

# 4. rvo2 (Python-RVO2) インストール（ORCAによる人物衝突回避、--num-humans使用時に必要）
# PyPI配布が無くソースビルドが必要なため、事前にCythonをインストールしておく
uv pip install Cython
uv run python -m pip install --no-build-isolation "git+https://github.com/sybrenstuvel/Python-RVO2.git"

echo ""
echo "セットアップ完了！"
