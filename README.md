# GPU Byte-Plane Scan Experiments

## 計劃目標
這個專案驗證 progressive byte-plane 掃描在 GPU 上是否能節省 HBM 頻寬，並建立 precision-throughput 曲線。

## 專案範圍
目前聚焦兩個基礎實驗：

- Experiment 0: HBM bandwidth microbenchmark
- Experiment 1: Byte-plane scan throughput scaling

程式只做 benchmark，不含資料庫層整合。

## 執行環境
本專案針對 Linux + NVIDIA CUDA 環境設計，TWCC 環境可能會落到 H100 或 H200。

需求：

- NVIDIA Driver
- CUDA Toolkit (`nvcc`)
- CMake >= 3.24
- C++17 編譯器

`-DCMAKE_CUDA_ARCHITECTURES=90` 適用 Hopper 家族（H100/H200）。目前 runner 預設 `CUDA_ARCH=90`，可用環境變數覆寫。

## 正式入口
正式 benchmark runner：

- `scripts/run_exp0.sh`
- `scripts/run_exp1.sh`

Slurm wrapper（exp0 範例）：

- `run_exp0.sh`（用 `sbatch run_exp0.sh` 送出）

## 快速開始

### Experiment 0（runner）

```bash
./scripts/run_exp0.sh
```

常用覆寫參數（環境變數）：

```bash
DEVICE=0 CUDA_ARCH=90 BYTES_MIN=1MB BYTES_MAX=8GB ITERS=200 ./scripts/run_exp0.sh
```

### Experiment 1（runner）

```bash
./scripts/run_exp1.sh
```

常用覆寫參數（環境變數）：

```bash
DEVICE=0 CUDA_ARCH=90 N=100000000 PLANE_BYTES=1 STRATEGY=byte K_MIN=1 K_MAX=8 ./scripts/run_exp1.sh
```

## 輸出與可追蹤性
每次 runner 執行都會建立獨立 run directory，不會覆寫前一次結果：

- `results/exp0/run_<timestamp>_job<jobid_or_nojob>_<gpu_tag>/`
- `results/exp1/run_<timestamp>_job<jobid_or_nojob>_<gpu_tag>/`

每個 run directory 會包含：

- CSV 結果
- `run_meta.txt`
- `repro_command.txt`
- `ncu_command_template.txt`

`gpu_tag` 由執行當下偵測 GPU 名稱產生，至少會區分 H100/H200。

## 額外說明

- `exp0` runner 預設會跑 `seq`、`masked`、`gather` 三種 mode。
- `exp1` 目前不依賴外部 dataset，程式會在 runtime 直接建立並初始化 SOA planes。
- 若要看各實驗細節，請讀：
  - `benchmarks/experiment0/README.md`
  - `benchmarks/experiment1/README.md`
