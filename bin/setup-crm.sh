#!/usr/bin/env bash
# Cài đặt thin-ml/CRM (Zhengyi/CRM) để chuyển ảnh -> GLB.
# Dùng: bash bin/setup-crm.sh   (cần GPU NVIDIA + CUDA toolkit/nvcc)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# 1. Clone mã nguồn CRM
[ -d CRM ] || git clone https://github.com/thu-ml/CRM.git CRM

# 2. Môi trường Python 3.10 riêng (stack gốc của CRM quá cũ cho python 3.13 mặc định)
command -v uv >/dev/null || pip install -q uv
[ -d .venv ] || uv venv --python 3.10 .venv
# shellcheck disable=SC1091
PY="$ROOT/.venv/bin/python"
PIP="uv pip install --python $PY"

# 3. PyTorch cu130 (khớp nvcc 13.0 của máy, cần để biên dịch nvdiffrast; T4 vẫn được hỗ trợ)
$PIP torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130

# 4. Thư viện của CRM
$PIP "numpy<2" "setuptools<81" wheel ninja
$PIP -r CRM/requirements.txt

# 5. nvdiffrast (biên dịch CUDA khi chạy lần đầu)
$PIP --no-build-isolation git+https://github.com/NVlabs/nvdiffrast

# Patch run.py: xuất thêm file .glb (bản gốc chỉ lưu zip obj)
grep -q output3d.glb CRM/run.py || sed -i 's|^\(\s*\)shutil.copy(obj_path, args.outdir+"output3d.zip")|&\n\1shutil.copy(glb_path, args.outdir+"output3d.glb")|' CRM/run.py

# 6. Tải trọng số model từ HuggingFace (Zhengyi/CRM)
"$PY" - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["CRM.pth", "ccm-diffusion.pth", "pixel-diffusion.pth"]:
    print(hf_hub_download(repo_id="Zhengyi/CRM", filename=f))
PY
echo "Xong. Chạy: bash bin/run-crm.sh IMG.webp"
