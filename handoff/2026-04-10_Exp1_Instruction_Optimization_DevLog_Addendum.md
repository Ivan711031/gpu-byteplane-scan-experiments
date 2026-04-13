# Dev Log Addendum: Exp1 Instruction-Level Reimplementation

Author: Codex (handoff addendum for Nick)  
Date: 2026-04-10 (CST, UTC+8)  
Last Updated: 2026-04-10 22:22 CST  
Base Reference: `handoff/2026-04-10_Project_Spec_and_Log.md`  
Scope: **Delta only** (this file records only newly implemented work and validation in this session)

## 0. Current Status Gate (Must Read)

Although this reimplementation improves `k=1..6`, the current `exp1` byte-path behavior is **not accepted as final** and should be treated as a regression against intended scaling behavior:

1. `k=7..8` throughput regressed versus the previous run.
2. The curve still does not match the expected near-memory-bound trend for higher `k`.
3. Next handoff should prioritize fixing high-`k` behavior before claiming optimization success.

## 1. What Was Reimplemented (Exp1)

### 1.1 `benchmarks/experiment1/bench_byteplane_scan.cu`
Reimplemented `byte` strategy from runtime loop control to compile-time template dispatch:

1. Replaced runtime kernel:
   - old: `scan_planes_u8_byte(..., int k, ...)`
   - new: `template <int K> scan_planes_u8_byte_unrolled(...)`
2. Added compile-time unrolling in kernel body:
   - `#pragma unroll` over `p = 0..K-1`
3. Added host-side dispatch helpers:
   - `scan_planes_u8_byte_unrolled_kernel_ptr(k)` for occupancy estimation
   - `launch_scan_planes_u8_byte_unrolled(k, ...)` for kernel launch switch-case (`k=1..8`)
4. Updated `run_one()`:
   - warmup loop and timed loop now launch template-specialized kernel for `strategy=byte`
5. Updated occupancy kernel selection for `strategy=byte`:
   - uses `k_for_occupancy = opt.single_k ? opt.k_single : opt.k_max`
6. Fixed pre-existing compile blocker found during rebuild:
   - `pstatic_cast<uint64_t>` -> `static_cast<uint64_t>` in shared128 path

### 1.2 `benchmarks/experiment1/CMakeLists.txt`
Added CUDA compile flag:

- `-lineinfo`

Purpose: preserve source mapping metadata for Nsight Compute source-level analysis.

### 1.3 `scripts/run_exp1.sh`
Updated generated Nsight template command:

- added `--import-source yes`
- added `--source-folders <repo_root>`

Purpose: make `.ncu-rep` portable for cross-machine source viewing.

### 1.4 New Slurm wrapper
Added root-level wrapper:

- `run_exp1.sh`

Purpose: align exp1 submission path with existing exp0 wrapper style (`sbatch run_exp1.sh`).

### 1.5 Design doc saved
Added spec file:

- `spec/2026-04-10_Instruction-Level_Optimization_for_Progressive_Byte-Plane_Scan_Kernel.md`

## 2. Commands Executed (Key Runs)

## 2.1 Build / compile validation
```bash
cmake -S benchmarks/experiment1 -B build/exp1 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build/exp1 -j
```

## 2.2 GPU smoke test (byte + packed32)
```bash
srun -p dev -A gov108018 -N 1 --gres=gpu:1 --cpus-per-task=2 --time=00:05:00 bash -lc '
  ml load jupyter/miniconda3
  ml load cuda/12.6
  source activate gpu-byteplane-scan
  cd /home/ccres1995/workspace_hanyin/gpu-byteplane-scan-experiments
  ./build/exp1/bench_byteplane_scan --device 0 --n 1024 --plane_bytes 1 --strategy byte --k_min 1 --k_max 2 --warmup 0 --iters 1 --csv /tmp/exp1_byte_smoke.csv
  ./build/exp1/bench_byteplane_scan --device 0 --n 1024 --plane_bytes 1 --strategy packed32 --k_min 1 --k_max 2 --warmup 0 --iters 1 --csv /tmp/exp1_packed32_smoke.csv
'
```
Result: success on job `16665`.

## 2.3 Full exp1 runner on GPU node (store in `results/`)
```bash
srun -p dev -A gov108018 -N 1 --gres=gpu:1 --cpus-per-task=4 --time=00:30:00 bash -lc '
  ml load jupyter/miniconda3
  ml load cuda/12.6
  source activate gpu-byteplane-scan
  cd /home/ccres1995/workspace_hanyin/gpu-byteplane-scan-experiments
  ./scripts/run_exp1.sh
'
```
Result: success on job `16675`.

## 2.4 Nsight Compute (source-importable report)
```bash
srun -p dev -A gov108018 -N 1 --gres=gpu:1 --cpus-per-task=4 --time=00:20:00 bash -lc '
  ml load jupyter/miniconda3
  ml load cuda/12.6
  source activate gpu-byteplane-scan
  cd /home/ccres1995/workspace_hanyin/gpu-byteplane-scan-experiments
  ts=$(date +%Y%m%d_%H%M%S)
  out_base="/home/ccres1995/workspace_hanyin/gpu-byteplane-scan-experiments/results/exp1/ncu_exp1_view_${ts}"
  ncu --set full --target-processes all --import-source yes --source-folders "$PWD" --export "$out_base" \
      ./build/exp1/bench_byteplane_scan --device 0 --n 1000000 --plane_bytes 1 --strategy byte \
      --k_min 1 --k_max 1 --block 256 --grid_mul 1 --warmup 0 --iters 1 --csv "${out_base}_bench.csv"
'
```
Result: success on job `16677`, kernel name shown as `scan_planes_u8_byte_unrolled`.

## 3. Test Results (New Data Only)

### 3.1 Full run artifact (post-reimplementation)
- Run directory:
  - `results/exp1/run_20260410_203754_job16675_H200/`
- Metadata highlights:
  - GPU: `NVIDIA H200`
  - Commit: `0e01db50e4aa0540fe6bf2e2123cc003901fa04b`
  - Worktree: `clean`

### 3.2 Byte strategy old/new comparison (same runner settings, `k=1..8`)
Source:
- old: `run_20260410_144202_job16611_H200/exp1.csv`
- new: `run_20260410_203754_job16675_H200/exp1.csv`

| k | old ms | new ms | speedup (old/new) | old GBps | new GBps | ratio (new/old) |
|---|-------:|-------:|------------------:|---------:|---------:|----------------:|
| 1 | 0.165192 | 0.140218 | 1.178 | 605.356 | 713.176 | 1.178 |
| 2 | 0.301759 | 0.162470 | 1.857 | 662.780 | 1230.993 | 1.857 |
| 3 | 0.325461 | 0.176530 | 1.844 | 921.771 | 1699.430 | 1.844 |
| 4 | 0.241899 | 0.194309 | 1.245 | 1653.586 | 2058.577 | 1.245 |
| 5 | 0.377459 | 0.214490 | 1.760 | 1324.647 | 2331.110 | 1.760 |
| 6 | 0.513687 | 0.339181 | 1.514 | 1168.026 | 1768.965 | 1.514 |
| 7 | 0.539680 | 0.603681 | 0.894 | 1297.064 | 1159.553 | 0.894 |
| 8 | 0.404947 | 0.630561 | 0.642 | 1975.568 | 1268.712 | 0.642 |

Observation:
1. `k=1..6` improved.
2. `k=7..8` regressed vs previous run.
3. Scaling remains non-ideal for strict memory-bound expectation; further analysis still needed.
4. **Project decision**: treat this state as "needs improvement", not "optimization completed."

### 3.3 New Nsight artifacts generated
1. `results/exp1/ncu_exp1_linesrc_20260410_172704.ncu-rep` (2026-04-10 17:27)
2. `results/exp1/ncu_exp1_view_20260410_204124.ncu-rep` (2026-04-10 20:41)
3. Corresponding `*_bench.csv` files generated with each report.

## 4. Git Record (This Session)

1. Commit:
   - `0e01db5 Optimize exp1 byte strategy and add source-importable ncu workflow`
2. Changed files in commit:
   - `benchmarks/experiment1/CMakeLists.txt`
   - `benchmarks/experiment1/bench_byteplane_scan.cu`
   - `run_exp1.sh`
   - `scripts/run_exp1.sh`
   - `spec/2026-04-10_Instruction-Level_Optimization_for_Progressive_Byte-Plane_Scan_Kernel.md`
3. Push:
   - `origin/main` updated (`718c080..0e01db5`)

## 5. Outstanding Follow-Ups (Not Yet Done Here)

1. Run full sweep for `strategy=packed32` under same `N=1e8, k=1..8` to validate Action B impact directly.
2. Re-profile hot spots at `k=6..8` with Nsight Compute (separate reports) to explain high-k regression.
3. Add a side-by-side CSV comparison script into repo (`results/exp1/compare_*.sh` or similar) for repeatable reporting.
4. Gate condition for next checkpoint:
   - No regression at `k=7..8` compared to `run_20260410_144202_job16611_H200/exp1.csv`.
   - Throughput trend should not collapse in high `k` range.
