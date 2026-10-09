#!/usr/bin/env bash
# Cài đặt Tencent Hunyuan3D-2.1 (shape + texture PBR) để chuyển ảnh -> GLB.
# Dùng: bash bin/setup-hunyuan.sh   (cần GPU NVIDIA ~24GB+ VRAM + CUDA toolkit/nvcc)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
HY=Hunyuan3D-2.1

# 1. Clone mã nguồn (pin commit đã test)
if [ ! -e "$HY/.git" ]; then
  rm -rf "$HY"
  git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git "$HY"
  git -C "$HY" checkout -q 82920d6
fi

# 2. Môi trường Python 3.11 riêng (bpy trên PyPI chỉ còn bản cp311 từ 4.2)
command -v uv >/dev/null || pip install -q uv
[ -d .venv-hy ] || uv venv --python 3.11 .venv-hy
PY="$ROOT/.venv-hy/bin/python"
PIP="uv pip install --python $PY"

# 3. PyTorch cu130 (khớp nvcc 13.0 để biên dịch custom_rasterizer)
$PIP torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130

# 4. Thư viện: bỏ mirror Trung Quốc và các gói chỉ dùng cho train/demo web/cupy
grep -vE '^(--extra-index-url|cupy|deepspeed|gradio|fastapi|uvicorn)' "$HY/requirements.txt" | sed 's/^bpy==4.0$/bpy==4.2.0/' > /tmp/hy-req.txt
$PIP "setuptools<81" wheel -r /tmp/hy-req.txt

# 5. Biên dịch extension CUDA/C++ cho bước texture
$PIP --no-build-isolation -e "$HY/hy3dpaint/custom_rasterizer"
# (thay compile_mesh_painter.sh: venv của uv không có python3-config để lấy đuôi .so)
DR="$HY/hy3dpaint/DifferentiableRenderer"
c++ -O3 -Wall -shared -std=c++11 -fPIC $("$PY" -m pybind11 --includes) "$DR/mesh_inpaint_processor.cpp" \
  -o "$DR/mesh_inpaint_processor$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"

# 6. Trọng số RealESRGAN (upscale texture); trọng số Hunyuan tự tải từ HF khi chạy lần đầu
mkdir -p "$HY/hy3dpaint/ckpt"
[ -f "$HY/hy3dpaint/ckpt/RealESRGAN_x4plus.pth" ] || \
  wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P "$HY/hy3dpaint/ckpt"

echo "Xong. Chạy: bash bin/run-hunyuan.sh IMG.webp"
