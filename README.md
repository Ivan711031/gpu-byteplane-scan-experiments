# GPU Byte-Plane Scan Experiments

## 1) 專案在做什麼
這個專案用 GPU benchmark 驗證 progressive byte-plane 掃描是否能有效節省 HBM 頻寬，並建立後續 precision-throughput 分析的基準。

目前重點是兩個基礎實驗：
- `Experiment 0`: HBM bandwidth microbenchmark（`seq/masked/gather`）
- `Experiment 1`: Byte-plane scan throughput scaling（SOA layout）

## 2) 目前資料夾狀態
`benchmarks/`
- 放實驗程式本體。
- `benchmarks/experiment0/`: `bench_hbm_bw.cu` 與 exp0 說明。
- `benchmarks/experiment1/`: `bench_byteplane_scan.cu` 與 exp1 說明。

`scripts/`
- 正式 benchmark runner。
- `scripts/run_exp0.sh`: 會 build + 跑 exp0 三模式，並寫 metadata / repro / ncu template。
- `scripts/run_exp1.sh`: 會 build + 跑 exp1，並寫 setup summary + metadata / repro / ncu template。

`run_exp0.sh`
- Slurm wrapper（範例）。
- 用 `sbatch run_exp0.sh` 可在排程系統上呼叫 `scripts/run_exp0.sh`。

`results/`
- 每次執行 runner 的結果輸出位置。
- `results/exp0/run_<timestamp>_job<id_or_nojob>_<gpu_tag>/`
- `results/exp1/run_<timestamp>_job<id_or_nojob>_<gpu_tag>/`

`spec/`
- 放「你寫給 agent 的計劃」與該計劃下的變更脈絡。
- 目前整理為單層放置 `.md`，不再用 `spec/h100/` 子資料夾。
- 命名規則建議：`YYYY-MM-DD_主題.md`。
- 目前檔案：`spec/2026-04-10_H100-H200_混合環境對齊計畫.md`

`build/`
- CMake build 輸出（中間產物）。

## 3) 虛擬環境與工具鏈（你現在這台機器）
先載入 module 並啟用 conda：

```bash
ml load jupyter/miniconda3
ml load cuda/12.6
source activate gpu-byteplane-scan
```

建議每次先確認：

```bash
which nvcc
nvcc --version
cmake --version
gcc --version
```

補充：
- 在登入節點通常可以編譯，但不一定能直接使用 GPU。
- 若遇到 `no CUDA-capable device is detected`，代表你在沒有 GPU 的節點，請改用 `srun` 或 `sbatch` 到計算節點執行。

## 4) 如何執行
### 4.1 Experiment 0（正式 runner）

```bash
./scripts/run_exp0.sh
```

常用覆寫參數（環境變數）：

```bash
DEVICE=0 CUDA_ARCH=90 BYTES_MIN=1MB BYTES_MAX=8GB BYTES_MULT=2 WARMUP=10 ITERS=200 ./scripts/run_exp0.sh
```

### 4.2 Experiment 1（正式 runner）

```bash
./scripts/run_exp1.sh
```

常用覆寫參數（環境變數）：

```bash
DEVICE=0 CUDA_ARCH=90 N=100000000 PLANE_BYTES=1 STRATEGY=byte K_MIN=1 K_MAX=8 WARMUP=10 ITERS=200 ./scripts/run_exp1.sh
```

### 4.3 用 Slurm 跑 exp0（wrapper）

```bash
sbatch run_exp0.sh
```

## 5) 每次執行會產生哪些檔案
exp0 run 目錄內：
- `exp0_seq.csv`
- `exp0_masked.csv`
- `exp0_gather.csv`
- `run_meta.txt`
- `repro_command.txt`
- `ncu_command_template.txt`

exp1 run 目錄內：
- `exp1.csv`
- `setup_estimate.txt`
- `run_meta.txt`
- `repro_command.txt`
- `ncu_command_template.txt`

`run_meta.txt` 會記錄：
- job id / partition / host
- 實際 GPU 名稱與 tag（可分辨 H100/H200）
- command line
- git branch / commit / dirty 狀態

## 6) H100/H200 混合環境注意事項
- 這份 repo 不強綁 nano4 或 nano5；重點是每次執行都要可追蹤「實際跑在哪種 GPU」。
- runner 會用執行當下偵測到的 GPU 寫入 metadata 與 run 目錄名稱。
- `CUDA_ARCH=90` 針對 Hopper（H100/H200）可用；如需其他架構可自行覆寫。

## 7) 延伸：Nsight Compute
每個 run 目錄都會附 `ncu_command_template.txt`。
你可以直接複製後微調 metrics / output 名稱，再重跑做 profiling。
