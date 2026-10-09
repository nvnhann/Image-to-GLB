#!/usr/bin/env bash
# Dùng: bash bin/run-crm.sh [ảnh] [thư mục out]   -> out/output3d.glb
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMG="$(realpath "${1:-$ROOT/IMG.webp}")"
OUT="$(realpath -m "${2:-$ROOT/out}")"
export PATH="$ROOT/.venv/bin:$PATH"
mkdir -p "$OUT"
cd "$ROOT/CRM"
python run.py --inputdir "$IMG" --outdir "$OUT/"
