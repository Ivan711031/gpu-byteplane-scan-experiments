# Project Spec & Log: gpu-byteplane-scan-experiments

## PRD
### High-Level Objective
在 TWCC 的 H100/H200 混合環境下，建立可重現、可追蹤、可交接的 GPU benchmark workflow，用於驗證：
- Experiment 0: HBM bandwidth baseline（seq/masked/gather）
- Experiment 1: byte-plane scan throughput scaling

### Product Scope
- 提供穩定 runner 與 Slurm wrapper，避免手動流程造成結果不可比。
- 每次執行必須自動記錄 GPU 型號、Job ID、執行參數、重跑指令。
- 產出可直接延伸到 Nsight Compute profiling 的 command template。

### Out of Scope (Current Phase)
- 不修改 `exp0/exp1` CUDA kernel 核心演算法。
- 不做 encoding/decoding pipeline 與資料庫層整合。
- 不在本階段整合自動 `ncu` 執行流程（僅提供 template）。

### Success Criteria
- `scripts/run_exp0.sh` 能一鍵跑完 `seq/masked/gather` 並產生完整 metadata。
- `scripts/run_exp1.sh` 能一鍵跑完預設 `k=1..8` 並產生 setup/metadata。
- 每次 run 目錄命名帶 timestamp/job/gpu，避免覆寫。
- 在 GPU 計算節點可實際成功執行（已驗證 H200）。

## SRS
### Tech Stack & Environment
- Environment: Linux + Slurm + NVIDIA CUDA（TWCC）
- CUDA Toolkit: 12.6
- Compiler: GCC 11.5.0
- CMake: 4.3.1
- Conda env: `gpu-byteplane-scan`

### Project Structure
- `benchmarks/experiment0/`: `bench_hbm_bw.cu` + experiment0 README
- `benchmarks/experiment1/`: `bench_byteplane_scan.cu` + experiment1 README
- `scripts/run_exp0.sh`: 正式 exp0 runner
- `scripts/run_exp1.sh`: 正式 exp1 runner
- `run_exp0.sh`: Slurm wrapper（exp0）
- `results/exp0/`, `results/exp1/`: run artifacts
- `spec/`: 人類寫給 agent 的規劃文檔（目前單層）
- `handoff/`: agent 接手用 context spec（本檔）

### Functional Requirements (Implemented)
1. 每次 run 建立唯一目錄：
   - `results/exp0/run_<timestamp>_job<id_or_nojob>_<gpu_tag>/`
   - `results/exp1/run_<timestamp>_job<id_or_nojob>_<gpu_tag>/`
2. 每次 run 產出：
   - `run_meta.txt`
   - `repro_command.txt`
   - `ncu_command_template.txt`
3. exp0 runner 預設執行三模式：`seq`, `masked`, `gather`
4. exp1 runner 額外產出 `setup_estimate.txt`（memory 初始化估算）
5. `CUDA_ARCH` 預設 90（Hopper），可被環境變數覆寫

### Interfaces & Contracts
- `scripts/run_exp0.sh` env vars:
  - `DEVICE`, `CUDA_ARCH`, `BYTES_MIN`, `BYTES_MAX`, `BYTES_MULT`, `BLOCK`, `GRID_MUL`, `WARMUP`, `ITERS`, `MASK_STRIDE`, `MASK_ACTIVE`, `GATHER_SPAN`, `GATHER_SEED`
- `scripts/run_exp1.sh` env vars:
  - `DEVICE`, `CUDA_ARCH`, `N`, `PLANE_BYTES`, `STRATEGY`, `K_MIN`, `K_MAX`, `BLOCK`, `GRID_MUL`, `WARMUP`, `ITERS`
- `run_exp0.sh`:
  - 載入 `jupyter/miniconda3`, `cuda/12.6`
  - activate `gpu-byteplane-scan`
  - 呼叫 `scripts/run_exp0.sh`

### Commands & Testing
#### Environment bootstrap
```bash
ml load jupyter/miniconda3
ml load cuda/12.6
source activate gpu-byteplane-scan
which nvcc
nvcc --version
cmake --version
gcc --version
```

#### Compile/Run commands
```bash
# Exp0 via runner
./scripts/run_exp0.sh

# Exp1 via runner
./scripts/run_exp1.sh

# Exp0 via Slurm wrapper
sbatch run_exp0.sh
```

#### Direct Slurm verification commands (used)
```bash
# Exp0 minimal runtime check
srun --immediate=60 -p dev -A gov108018 -N 1 --gres=gpu:1 --cpus-per-task=2 --time=00:05:00 \
  bash -lc 'ml load jupyter/miniconda3 && ml load cuda/12.6 && source activate gpu-byteplane-scan && \
            cd /home/ccres1995/workspace_hanyin/gpu-byteplane-scan-experiments && \
            ./build/exp0/bench_hbm_bw --mode seq --bytes_min 1MB --bytes_max 1MB --warmup 0 --iters 1 --csv /tmp/exp0_env_check.csv'

# Exp1 minimal runtime check
srun --immediate=60 -p dev -A gov108018 -N 1 --gres=gpu:1 --cpus-per-task=2 --time=00:05:00 \
  bash -lc 'ml load jupyter/miniconda3 && ml load cuda/12.6 && source activate gpu-byteplane-scan && \
            cd /home/ccres1995/workspace_hanyin/gpu-byteplane-scan-experiments && \
            N=1024 K_MIN=1 K_MAX=1 WARMUP=0 ITERS=1 ./scripts/run_exp1.sh'
```

### Test Evidence (Latest Known Good Runs)
- Exp0 full runner:
  - Job: `16608`
  - Node: `25a-hgpn069`
  - GPU: `NVIDIA H200`
  - Output: `results/exp0/run_20260410_143905_job16608_H200/`
- Exp1 full runner:
  - Job: `16611`
  - Node: `25a-hgpn069`
  - GPU: `NVIDIA H200`
  - Output: `results/exp1/run_20260410_144202_job16611_H200/`

### Boundaries (Crucial)
#### Architecture boundaries (do not change without explicit approval)
- 禁止修改核心 benchmark kernel 行為與資料路徑：
  - `benchmarks/experiment0/bench_hbm_bw.cu`
  - `benchmarks/experiment1/bench_byteplane_scan.cu`
- 禁止改動 CSV schema（`device/sm/cc` 欄位必須保留）。
- 禁止移除 runner 產生的追蹤檔：`run_meta.txt`, `repro_command.txt`, `ncu_command_template.txt`。
- 禁止把 runner 輸出改回 repo 根目錄（必須維持 `results/exp*/run_*` 結構）。

#### Process boundaries
- 在登入節點（如 `25a-lgn04`）不要宣稱 GPU runtime 可用；必須透過 `srun/sbatch` 到計算節點驗證。
- 禁止使用破壞性 git 指令（`reset --hard`, 強制覆蓋歷史）。

### Completed Tasks (Context)
- [x] 建立混合 H100/H200 workflow 的 runner 架構
- [x] 新增 `scripts/run_exp1.sh`
- [x] 新增根目錄 `run_exp0.sh` Slurm wrapper
- [x] README 重寫（資料夾狀態、環境、使用方式）
- [x] `spec/` 改為單層 `.md`，檔名含日期
- [x] 成功 push commits:
  - `5b64d08` align runners for mixed H100/H200 workflow
  - `17718b7` reorganize spec plans and expand README guide

### Current Task & Next Steps
#### Current bottlenecks
- 目前可見/可用節點以 H200 為主，H100 尚未在當前查詢中出現可用資源。
- 歷史 results 內有早期錯誤命名 run 目錄（GPU 偵測修正前留下），需要是否清理的決策。
- 尚未提供 exp1 專用 Slurm wrapper（目前可用 `srun` 或手動 sbatch 腳本）。

#### Next actionable items
1. 補 `run_exp1.sh` 對應 Slurm wrapper（命名與 `run_exp0.sh` 對齊）。
2. 取得 H100 節點後，重跑 exp0/exp1 建立 H100 vs H200 對照 baseline。
3. 規劃是否保留或清理早期 `jobnojob_UNKNOWN_GPU/invalid_gpu_tag` 結果目錄。
4. 若進入 profiling 階段，根據 `ncu_command_template.txt` 增加固定 metrics set 與輸出命名規範。
