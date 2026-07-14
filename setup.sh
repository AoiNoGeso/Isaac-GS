#!/usr/bin/env bash
# Isaac-GS 環境セットアップスクリプト
# 実行: bash setup.sh
set -e

# 仮想環境はリポジトリ直下 .venv に集約する。
# uv run は既定で .venv を使うため、pyproject の [tool.uv] managed=false（自動sync無効）と
# 併せて `source activate` なしに `uv run xxx.py` が実行できる。
ENV_DIR=".venv"

# 1. 仮想環境作成
uv venv --python 3.12 --seed "$ENV_DIR"
source "$ENV_DIR/bin/activate"

# 2. IsaacSim インストール
uv pip install \
    "isaacsim[all,extscache]==6.0.0.1" \
    --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match \
    --prerelease=allow

# 3. PyTorch (CUDA 12.8) インストール
#    isaacsim のメタデータ宣言より新しいバージョンを上書きインストール
uv pip install -U \
    "torch==2.10.0" \
    "torchvision==0.25.0" \
    --index-url https://download.pytorch.org/whl/cu128

# 4. Isaac-GS 追加依存インストール（gymnasium / wandb / Pillow / pydantic）
uv pip install -e .

echo ""
echo "セットアップ完了！"
echo "実行方法（source activate 不要）:"
echo "  uv run tasks/point_navigation/train.py --headless"
echo "  （従来通り source .venv/bin/activate → python でも可）"
