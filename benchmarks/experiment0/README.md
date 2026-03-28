# Experiment 0 — HBM bandwidth microbenchmark

這個實驗用 **cudaEvent** 量測 GPU global memory 的「有效讀取頻寬」，用 **dummy reduction** 避免編譯器把讀取最佳化掉：kernel 讀完整段資料後，只在暫存器內做簡單累加，最後 **每個 thread block 只寫出 1 個數值**到 global memory。

目前支援三種模式：`seq`（coalesced sequential read）、`masked`（部分 lane 閒置）、`gather`（index array 隨機讀取）。

## Build

Linux + CUDA + CMake:

```bash
cmake -S benchmarks/experiment0 -B build/exp0 -DCMAKE_BUILD_TYPE=Release
cmake --build build/exp0 -j
```

若在 H100（SM90）上，建議指定架構以加快編譯：

```bash
cmake -S benchmarks/experiment0 -B build/exp0 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build/exp0 -j
```

## Run

預設 sweep `1MB -> 8GB`、每個 size warmup 10 次、量測 200 次，輸出單一 CSV 檔：

```bash
./build/exp0/bench_hbm_bw --mode seq --csv exp0_hbm_bw.csv
```

常用參數：

```bash
./build/exp0/bench_hbm_bw \
  --mode seq \
  --device 0 \
  --bytes_min 1MB --bytes_max 8GB --bytes_mult 2 \
  --block 256 --grid_mul 1 \
  --warmup 10 --iters 200 \
  --csv exp0_hbm_bw.csv

## Modes

- `--mode seq`: 完全 coalesced sequential read
- `--mode masked`: 以 warp lane pattern 模擬部分 lane inactive。搭配 `--mask_stride`/`--mask_active`
- `--mode gather`: 透過 index array 做隨機讀取。搭配 `--gather_span`/`--gather_seed`（`--gather_span 0` 表示全範圍）

範例：

```bash
./build/exp0/bench_hbm_bw --mode masked --mask_stride 2 --mask_active 1 --csv exp0_masked.csv
./build/exp0/bench_hbm_bw --mode gather --gather_span 1GB --csv exp0_gather.csv
```
```

CSV 欄位：

`mode, bytes, effective_bytes, block, grid, warmup, iters, ms_per_iter, GBps, active_frac, gather_span_bytes, device, sm, cc_major, cc_minor`

註：

- `GBps` 使用 1e9（GB/s）而非 GiB/s。
- `effective_bytes` 為 `masked` 模式實際讀取的 bytes（`bytes * active_frac`）。
- `gather_span_bytes` 為 `gather` 模式索引覆蓋的 bytes 範圍。
