#!/usr/bin/env bash
# Dùng: bash bin/run-hunyuan.sh [ảnh] [thư mục out] [tuỳ chọn...]  -> out/hunyuan/textured.glb
# Tuỳ chọn: --steps 50 --guidance 5.0 --octree 384 --views 8 --tex-res 768 --faces 150000 --seed 1234 --no-texture
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMG="$(realpath "${1:-$ROOT/IMG.webp}")"
OUT="$(realpath -m "${2:-$ROOT/out/hunyuan}")"
shift $(( $# < 2 ? $# : 2 ))
cd "$ROOT/Hunyuan3D-2.1"
"$ROOT/.venv-hy/bin/python" "$ROOT/bin/hunyuan_gen.py" "$IMG" "$OUT" "$@"
