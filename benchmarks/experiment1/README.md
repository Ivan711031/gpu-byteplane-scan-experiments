# Experiment 1 — Byte-plane scaling (SOA)

目標：驗證 **讀更少 byte-planes 是否等比例節省頻寬**。資料 layout 採 SOA：把每個 FP64 value 的 8 個 bytes 拆成 8 個獨立、連續陣列（byte-plane）。

這個 benchmark 只量測「掃描讀取」本身的有效頻寬（dummy reduction：只在 register 做累加，最後每個 block 寫 1 個值到 global）。

## Build

```bash
cmake -S benchmarks/experiment1 -B build/exp1 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build/exp1 -j
```

## Run

預設：`n=1e8`，`k=1..8`，輸出單一 CSV：

```bash
./build/exp1/bench_byteplane_scan --csv exp1.csv
```

常用參數：

```bash
./build/exp1/bench_byteplane_scan \
  --device 0 \
  --n 100000000 \
  --plane_bytes 1 \
  --strategy byte \
  --k_min 1 --k_max 8 \
  --block 256 --grid_mul 1 \
  --warmup 10 --iters 200 \
  --csv exp1_byteplane_scan.csv
```

## Strategies

- `--strategy byte`：每個 thread 對每個 plane 直接做 `uint8` load（最貼近「1-byte load 是否被懲罰」的核心問題）。
- `--strategy packed32`：把每個 plane 當成 `uint32` 陣列讀，每個 load 吃 4 個 elements（避免 1-byte 指令，測試「packed read」路線）。
- `--strategy shared128`：每個 warp 先 stage 128B 到 shared（每 lane 讀 4 bytes），再每 lane 只用其中 1 byte；這是刻意 overfetch 以逼出 coalesced transaction，用來評估 staging 是否值得。

另外支援 `--plane_bytes 2`：把 planes 改為 4 個 `uint16` plane（相當於 2-byte planes）。此時策略固定為 `byte`（u16 loads）。

## Output

CSV 欄位包含：

- `logical_bytes = n * k * plane_bytes`
- `logical_GBps`：用 logical bytes 計算的 GB/s（1e9）
- `overfetch_factor`：`shared128` 會是 `4.0`（代表每個 lane 實際 stage 4 bytes，但只用 1 byte）

