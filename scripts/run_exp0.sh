#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build/exp0}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/results/exp0}"

DEVICE="${DEVICE:-0}"
BYTES_MIN="${BYTES_MIN:-1MB}"
BYTES_MAX="${BYTES_MAX:-8GB}"
BYTES_MULT="${BYTES_MULT:-2}"
BLOCK="${BLOCK:-256}"
GRID_MUL="${GRID_MUL:-1}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-200}"
CUDA_ARCH="${CUDA_ARCH:-}"

MASK_STRIDE="${MASK_STRIDE:-2}"
MASK_ACTIVE="${MASK_ACTIVE:-1}"
GATHER_SPAN="${GATHER_SPAN:-0}"
GATHER_SEED="${GATHER_SEED:-1}"

cmake_args=(
  -S "$ROOT_DIR/benchmarks/experiment0"
  -B "$BUILD_DIR"
  -DCMAKE_BUILD_TYPE=Release
)
if [[ -n "$CUDA_ARCH" ]]; then
  cmake_args+=(-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH")
fi

cmake "${cmake_args[@]}"
cmake --build "$BUILD_DIR" -j

mkdir -p "$OUT_DIR"

bin="$BUILD_DIR/bench_hbm_bw"
common_args=(
  --device "$DEVICE"
  --bytes_min "$BYTES_MIN" --bytes_max "$BYTES_MAX" --bytes_mult "$BYTES_MULT"
  --block "$BLOCK" --grid_mul "$GRID_MUL"
  --warmup "$WARMUP" --iters "$ITERS"
)

"$bin" --mode seq "${common_args[@]}" --csv "$OUT_DIR/exp0_seq.csv"
"$bin" --mode masked "${common_args[@]}" --mask_stride "$MASK_STRIDE" --mask_active "$MASK_ACTIVE" \
  --csv "$OUT_DIR/exp0_masked.csv"
"$bin" --mode gather "${common_args[@]}" --gather_span "$GATHER_SPAN" --gather_seed "$GATHER_SEED" \
  --csv "$OUT_DIR/exp0_gather.csv"

echo "exp0 outputs in: $OUT_DIR"
