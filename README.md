# GPU Byte-Plane Scan Experiments

## 計劃目標
這個專案要驗證「progressive byte-plane 掃描」在 GPU 上是否能有效節省 HBM 頻寬，並建立 precision-throughput 曲線。核心想法是把 FP64 拆成 byte-planes，先讀高位、逐層補精度，縮小每次掃描的資料量。

## 目前工作階段
目前完成的內容聚焦在實驗 0 與實驗 1 的基礎 benchmark。

- Experiment 0：HBM bandwidth microbenchmark
- Experiment 1：Byte-plane 掃描 throughput 的 scaling 行為

這兩個實驗是後續所有結果的基準與前提，因此優先完成。

## 專案程式在做什麼
專案中的程式用來量測 GPU HBM 讀取頻寬與 byte-plane scan 的吞吐量，不包含 encoding/decoding 或資料庫層的功能。

目前包含：
- `benchmarks/experiment0/bench_hbm_bw.cu`
  用 dummy reduction 測量 GPU global memory 的有效讀取頻寬，支援三種存取模式：
  `seq`（coalesced sequential read）、`masked`（部分 lane 閒置）、`gather`（index array 隨機讀取）。
- `benchmarks/experiment1/bench_byteplane_scan.cu`
  量測 byte-plane layout 的掃描 throughput，並比較不同讀取策略。
- `scripts/run_exp0.sh`
  一鍵跑完 Experiment 0 的三種模式輸出 CSV。

## 環境需求
必須在 Linux + NVIDIA GPU 環境執行（例如國網中心的 nano4）。

需求：
- NVIDIA Driver
- CUDA Toolkit（需要 `nvcc`）
- CMake >= 3.24
- C++ 編譯器（gcc/g++ 或 clang，支援 C++17）

建議：
- Ninja（加速 build）

## 快速開始

### Experiment 0：HBM bandwidth
建置與執行：

```bash
cmake -S benchmarks/experiment0 -B build/exp0 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build/exp0 -j
./build/exp0/bench_hbm_bw --mode seq --csv exp0_hbm_bw.csv
```

一鍵腳本（跑 seq/masked/gather 三種模式）：

```bash
./scripts/run_exp0.sh
```

輸出 CSV 預設在 `results/exp0/`。

常用參數：

```bash
./build/exp0/bench_hbm_bw \
  --mode masked \
  --mask_stride 2 --mask_active 1 \
  --bytes_min 1MB --bytes_max 8GB --bytes_mult 2 \
  --block 256 --grid_mul 1 \
  --warmup 10 --iters 200 \
  --csv exp0_masked.csv
```

### Experiment 1：Byte-plane scaling
建置與執行：

```bash
cmake -S benchmarks/experiment1 -B build/exp1 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build/exp1 -j
./build/exp1/bench_byteplane_scan --csv exp1.csv
```

## 常見問題

Q：為什麼一定要 Linux + NVIDIA GPU？
因為這些 benchmark 都用 CUDA，macOS 不支援 NVIDIA CUDA 開發。

Q：`-DCMAKE_CUDA_ARCHITECTURES=90` 一定要嗎？
不一定。這是 H100 的 SM 版本。請依實際 GPU 調整，例如 A100 用 80，V100 用 70。

## 後續計劃
後續會進行：
- Experiment 2：截斷誤差分析
- Experiment 3：Progressive aggregation kernel
- Experiment 4：Progressive filter kernel

這些實驗完成後會補上圖表與論文用的結果整理腳本。
