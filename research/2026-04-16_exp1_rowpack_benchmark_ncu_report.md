# 2026-04-16 Exp1 Rowpack Strategy Benchmark and NCU Report

## 摘要

本次版本的改動是成功的。核心原因不是單純把 byte-plane scan 改得「看起來比較快」，而是同時滿足了三個條件：

1. 語意仍然正確：`rowpack4` 和 `rowpack16` 只讀同一個 byte-plane 中連續多筆 row，不會把同一 row 的多個 precision byte 偷偷打包在一起。因此它仍保留 progressive byte-plane 的 `k` 語意。
2. Benchmark 結果明顯變好：在 H200、`n=100000000`、`block=256`、`iters=1000` 下，`rowpack16` 在 `k=8` 達到 `4394.268 GB/s`，高於 `contiguous64` 的 `3929.343 GB/s`，也比 `byte ilp4` 的 `2399.070 GB/s` 快 `1.832x`。
3. NCU 支持這個機制：SASS load 指令從 `LDG.E.U8:8b` 變成 `LDG.E:32b` 或 `LDG.E.128:128b`；同時 rowpack 的 memory throughput、DRAM utilization、L2 utilization 都顯著上升，executed instructions 顯著下降，沒有 local spill。

最值得保留的主線結果是 `rowpack16`，其次是 `rowpack4`。`shared128` 是診斷路徑，不是主要成果；它雖然強迫 coalesced/staged access，但 overfetch factor 是 `4.000`，實測 throughput 也遠低於 rowpack。

## 量測來源

完整 benchmark 資料：

- `results/exp1/run_20260415_rowpack_full_after_impl/byte_baseline.csv`
- `results/exp1/run_20260415_rowpack_full_after_impl/byte_ilp4.csv`
- `results/exp1/run_20260415_rowpack_full_after_impl/rowpack4.csv`
- `results/exp1/run_20260415_rowpack_full_after_impl/rowpack16.csv`
- `results/exp1/run_20260415_rowpack_full_after_impl/shared128.csv`
- `results/exp1/run_20260415_rowpack_full_after_impl/plane16.csv`
- `results/exp1/run_20260415_rowpack_full_after_impl/contiguous64.csv`

NCU 資料：

- `results/exp1/ncu_rowpack_after_impl_20260416/*.ncu-rep`
- `results/exp1/ncu_rowpack_after_impl_20260416/ncu_summary.csv`
- `results/exp1/ncu_rowpack_after_impl_20260416/sass_load_summary.csv`
- `results/exp1/ncu_rowpack_after_impl_20260416/extract_ncu_summary.py`

NCU 環境：

- 正確環境是 `ml load cuda/12.6 && source activate gpu-byteplane-scan`
- `ncu` path: `/work/HPC_software/LMOD/ai-h200/apps/cuda-12.6/bin/ncu`
- `ncu` version: `2025.3.1.0`
- NCU profile 使用 `--set full`、`--target-processes all`、`--launch-count 1`、`--launch-skip 0`、`--import-source yes`
- NCU 有警告 6 個 `ctc__rx/tx_bytes...` metrics 無法存取，但本報告使用的核心指標都有收集到

共用 benchmark 設定：

| item | value |
|---|---:|
| GPU | NVIDIA H200 |
| SM count | 132 |
| Compute capability | 9.0 |
| n | 100000000 |
| block | 256 |
| grid_mul | 1 |
| warmup | 10 |
| timed iterations | 1000 |

## 目前版本的程式執行邏輯

### 資料語意

Exp1 的 byte-plane scan 把資料拆成多個 plane。對 1-byte plane 的主要實驗來說：

- 每一個 plane 是一段長度 `n` 的 `uint8_t` array。
- `k=1` 表示只讀第一個 byte-plane。
- `k=8` 表示讀八個 byte-plane。
- logical bytes 計算方式是 `n * k * plane_bytes`。

本次新增的 `rowpack4` 和 `rowpack16` 沒有改變 `k` 的語意。它們改變的是每一條 global load instruction 搬多少同一 plane 的連續 rows。

```text
rowpack4:
  one uint32_t load = 4 consecutive rows from the same byte-plane

rowpack16:
  one uint4 load = 16 consecutive rows from the same byte-plane
```

這與「把同一 row 的 4 或 16 個 precision bytes 打包讀進來」完全不同。後者會破壞 progressive precision 的研究假設；目前版本沒有這樣做。

### 主程式流程

主 benchmark 程式負責以下事情：

1. 解析參數與 strategy：支援 `byte`、`rowpack4`、`rowpack16`、`shared128`，並把舊名稱 `packed32` 正規化成 `rowpack4`。
2. 配置 plane memory：1-byte plane 配 `uint8_t` planes，2-byte plane 配 `uint16_t` planes。
3. 根據 strategy 和 k 選 kernel：`launch_selected_kernel` 是目前的 kernel dispatcher。
4. 用 occupancy API 算每個 kernel 的 grid：template kernel 會依照每個 k 分別算 grid。
5. 每個 k 先 warmup，再用 CUDA event 量 timed loop。
6. 輸出 CSV：包含 `strategy`、`plane_bytes`、`k`、`logical_bytes`、`overfetch_factor`、`grid`、`ms_per_iter`、`logical_GBps`。
7. 可選 `--validate`：會跑一次非 timed kernel，copy per-block sum 回 CPU 檢查總和。

### 各策略的意義

| strategy | 角色 | 讀取型態 | 本次結論 |
|---|---|---|---|
| `byte` baseline | 原本 scalar byte scan | scalar `uint8_t` loads | 慢，作為原始比較基準 |
| `byte --byte_variant ilp4` | byte scalar 的 ILP 改良 | scalar `uint8_t` loads，但每 thread 做 4 條 stride | 比 baseline 快很多，但仍受 scalar byte load 限制 |
| `rowpack4` | 主要 row-wise packed 策略 | `uint32_t` loads，4 rows/load | 成功，k 大時接近或超過 contiguous64 |
| `rowpack16` | 主要 row-wise packed 策略 | `uint4`/128-bit loads，16 rows/load | 最成功，k=1..8 幾乎都是最佳 |
| `shared128` | 診斷策略 | warp staging 128 physical bytes | overfetch 4x，結果差，不適合作主成果 |
| `plane16` | 2-byte plane 對照 | scalar `uint16_t` loads | 不是本次 rowpack 主線 |
| `contiguous64` | upper/reference baseline | contiguous `uint64_t` loads | 用來比較 byte-plane 是否接近連續 8-byte scan |

## 完整 Benchmark 數據

下表保留所有策略的主要輸出欄位。`logical_GBps` 是用 logical bytes 算出來的 throughput；對 `shared128` 要同時看 `overfetch_factor=4.000`，因為它實際搬的 physical bytes 比 logical bytes 多。

| strategy | variant | plane_bytes | k | logical_bytes | overfetch | grid | ms_per_iter | logical_GBps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| byte | baseline | 1 | 1 | 100000000 | 1.000 | 1056 | 0.139758 | 715.520 |
| byte | baseline | 1 | 2 | 200000000 | 1.000 | 1056 | 0.162017 | 1234.435 |
| byte | baseline | 1 | 3 | 300000000 | 1.000 | 1056 | 0.176865 | 1696.207 |
| byte | baseline | 1 | 4 | 400000000 | 1.000 | 1056 | 0.195613 | 2044.850 |
| byte | baseline | 1 | 5 | 500000000 | 1.000 | 1056 | 0.216509 | 2309.372 |
| byte | baseline | 1 | 6 | 600000000 | 1.000 | 1056 | 0.251384 | 2386.786 |
| byte | baseline | 1 | 7 | 700000000 | 1.000 | 1056 | 0.308096 | 2272.023 |
| byte | baseline | 1 | 8 | 800000000 | 1.000 | 1056 | 0.379572 | 2107.635 |
| byte | ilp4 | 1 | 1 | 100000000 | 1.000 | 1056 | 0.063237 | 1581.349 |
| byte | ilp4 | 1 | 2 | 200000000 | 1.000 | 1056 | 0.101164 | 1976.992 |
| byte | ilp4 | 1 | 3 | 300000000 | 1.000 | 1056 | 0.128990 | 2325.761 |
| byte | ilp4 | 1 | 4 | 400000000 | 1.000 | 792 | 0.169644 | 2357.881 |
| byte | ilp4 | 1 | 5 | 500000000 | 1.000 | 1056 | 0.214017 | 2336.266 |
| byte | ilp4 | 1 | 6 | 600000000 | 1.000 | 792 | 0.249484 | 2404.965 |
| byte | ilp4 | 1 | 7 | 700000000 | 1.000 | 792 | 0.296996 | 2356.934 |
| byte | ilp4 | 1 | 8 | 800000000 | 1.000 | 792 | 0.333463 | 2399.070 |
| rowpack4 | rowpack4 | 1 | 1 | 100000000 | 1.000 | 1056 | 0.044101 | 2267.529 |
| rowpack4 | rowpack4 | 1 | 2 | 200000000 | 1.000 | 1056 | 0.060982 | 3279.678 |
| rowpack4 | rowpack4 | 1 | 3 | 300000000 | 1.000 | 1056 | 0.078853 | 3804.545 |
| rowpack4 | rowpack4 | 1 | 4 | 400000000 | 1.000 | 1056 | 0.098016 | 4080.977 |
| rowpack4 | rowpack4 | 1 | 5 | 500000000 | 1.000 | 1056 | 0.118563 | 4217.183 |
| rowpack4 | rowpack4 | 1 | 6 | 600000000 | 1.000 | 1056 | 0.139834 | 4290.796 |
| rowpack4 | rowpack4 | 1 | 7 | 700000000 | 1.000 | 1056 | 0.162270 | 4313.787 |
| rowpack4 | rowpack4 | 1 | 8 | 800000000 | 1.000 | 1056 | 0.184047 | 4346.708 |
| rowpack16 | rowpack16 | 1 | 1 | 100000000 | 1.000 | 1056 | 0.027219 | 3673.903 |
| rowpack16 | rowpack16 | 1 | 2 | 200000000 | 1.000 | 1056 | 0.048816 | 4096.993 |
| rowpack16 | rowpack16 | 1 | 3 | 300000000 | 1.000 | 1056 | 0.070981 | 4226.476 |
| rowpack16 | rowpack16 | 1 | 4 | 400000000 | 1.000 | 1056 | 0.093724 | 4267.872 |
| rowpack16 | rowpack16 | 1 | 5 | 500000000 | 1.000 | 1056 | 0.115275 | 4337.447 |
| rowpack16 | rowpack16 | 1 | 6 | 600000000 | 1.000 | 792 | 0.136949 | 4381.197 |
| rowpack16 | rowpack16 | 1 | 7 | 700000000 | 1.000 | 1056 | 0.160442 | 4362.937 |
| rowpack16 | rowpack16 | 1 | 8 | 800000000 | 1.000 | 1056 | 0.182055 | 4394.268 |
| shared128 | shared128 | 1 | 1 | 100000000 | 4.000 | 1056 | 0.255489 | 391.406 |
| shared128 | shared128 | 1 | 2 | 200000000 | 4.000 | 1056 | 0.387876 | 515.629 |
| shared128 | shared128 | 1 | 3 | 300000000 | 4.000 | 1056 | 0.530201 | 565.823 |
| shared128 | shared128 | 1 | 4 | 400000000 | 4.000 | 1056 | 0.680584 | 587.730 |
| shared128 | shared128 | 1 | 5 | 500000000 | 4.000 | 1056 | 0.856301 | 583.907 |
| shared128 | shared128 | 1 | 6 | 600000000 | 4.000 | 1056 | 1.004316 | 597.422 |
| shared128 | shared128 | 1 | 7 | 700000000 | 4.000 | 1056 | 1.154608 | 606.266 |
| shared128 | shared128 | 1 | 8 | 800000000 | 4.000 | 1056 | 1.294951 | 617.784 |
| byte | plane16 | 2 | 1 | 200000000 | 1.000 | 792 | 0.222700 | 898.070 |
| byte | plane16 | 2 | 2 | 400000000 | 1.000 | 792 | 0.415133 | 963.547 |
| byte | plane16 | 2 | 3 | 600000000 | 1.000 | 792 | 0.442378 | 1356.306 |
| byte | plane16 | 2 | 4 | 800000000 | 1.000 | 792 | 0.323339 | 2474.185 |
| contiguous64 | contiguous64 | 8 | 1 | 800000000 | 1.000 | 1056 | 0.203596 | 3929.343 |

## Benchmark 結果解讀

### 直接吞吐量

`rowpack16` 幾乎全區間最佳：

| k | byte baseline GB/s | byte ilp4 GB/s | rowpack4 GB/s | rowpack16 GB/s | best |
|---:|---:|---:|---:|---:|---|
| 1 | 715.520 | 1581.349 | 2267.529 | 3673.903 | rowpack16 |
| 2 | 1234.435 | 1976.992 | 3279.678 | 4096.993 | rowpack16 |
| 3 | 1696.207 | 2325.761 | 3804.545 | 4226.476 | rowpack16 |
| 4 | 2044.850 | 2357.881 | 4080.977 | 4267.872 | rowpack16 |
| 5 | 2309.372 | 2336.266 | 4217.183 | 4337.447 | rowpack16 |
| 6 | 2386.786 | 2404.965 | 4290.796 | 4381.197 | rowpack16 |
| 7 | 2272.023 | 2356.934 | 4313.787 | 4362.937 | rowpack16 |
| 8 | 2107.635 | 2399.070 | 4346.708 | 4394.268 | rowpack16 |

### 重要倍率

以最重要的 `k=8` 來看：

| comparison | ratio |
|---|---:|
| byte baseline / contiguous64 | 0.536x |
| byte ilp4 / contiguous64 | 0.611x |
| rowpack4 / contiguous64 | 1.106x |
| rowpack16 / contiguous64 | 1.118x |
| rowpack4 / byte ilp4 | 1.812x |
| rowpack16 / byte ilp4 | 1.832x |
| rowpack16 / rowpack4 | 1.011x |

幾個重點：

- `byte ilp4` 已經把 scalar byte baseline 從 `2107.635 GB/s` 推到 `2399.070 GB/s`，但仍然只有 contiguous64 的 `0.611x`。
- `rowpack4` 在 `k=8` 達到 `4346.708 GB/s`，已超過 contiguous64。
- `rowpack16` 在 `k=8` 達到 `4394.268 GB/s`，是本輪 benchmark 最佳點。
- `rowpack16` 在 `k=1` 也有 `3673.903 GB/s`，已經接近 contiguous64 的 `0.935x`。這很重要，因為 progressive scan 最常需要展示小 k 的 early-exit/low precision 優勢。
- `shared128` 雖然試圖改善 transaction shape，但因為固定 `4x` overfetch，`k=8` 只有 `617.784 GB/s`，不適合作為主要策略。

## NCU 數據

NCU profile 針對 `byte_ilp4`、`rowpack4`、`rowpack16` 的 `k=1,4,8`，以及 `contiguous64`。這裡不要把 NCU 的 single-launch duration 和 1000-iteration benchmark 的 `ms_per_iter` 完全等同，因為 profiler 會改變執行環境；但同一批 NCU run 之間的相對趨勢，以及 SASS load opcode、DRAM/L2 utilization、instruction count 非常有參考價值。

### SASS Load 指令

| report | load opcode |
|---|---|
| byte_ilp4_k1 | `LDG.E.U8:8b` |
| byte_ilp4_k4 | `LDG.E.U8:8b` |
| byte_ilp4_k8 | `LDG.E.U8:8b` |
| rowpack4_k1 | `LDG.E:32b` |
| rowpack4_k4 | `LDG.E:32b` |
| rowpack4_k8 | `LDG.E:32b` |
| rowpack16_k1 | `LDG.E.128:128b` |
| rowpack16_k4 | `LDG.E.128:128b` |
| rowpack16_k8 | `LDG.E.128:128b` |
| contiguous64 | `LDG.E.64.CONSTANT:64b` |

這張表直接支持本次改動的核心假設：`rowpack4` 確實把 scalar byte load 變成 32-bit load，`rowpack16` 確實變成 128-bit load，而不是只在 C++ 原始碼中改寫但最後被 compiler 退回 scalar byte loads。

### NCU 核心 Throughput/Occupancy 指標

| report | duration_us | mem_TB/s | DRAM % | L2 % | compute SM % | SM busy % | occupancy % | regs/thread | spills | executed inst |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| byte_ilp4_k1 | 81.504 | 1.265 | 26.28 | 36.71 | 58.71 | 61.81 | 79.31 | 32 | 0 | 36716280 |
| byte_ilp4_k4 | 186.080 | 2.166 | 45.02 | 63.23 | 56.73 | 58.43 | 72.13 | 34 | 0 | 70925488 |
| byte_ilp4_k8 | 356.832 | 2.250 | 46.75 | 66.18 | 49.74 | 50.81 | 73.21 | 40 | 0 | 120982512 |
| rowpack4_k1 | 48.192 | 2.131 | 44.32 | 52.95 | 53.75 | 58.81 | 94.30 | 20 | 0 | 13851490 |
| rowpack4_k4 | 115.712 | 3.485 | 72.44 | 80.63 | 76.76 | 80.94 | 81.54 | 28 | 0 | 40464678 |
| rowpack4_k8 | 206.592 | 3.887 | 80.80 | 89.06 | 82.59 | 85.37 | 75.87 | 32 | 0 | 74907262 |
| rowpack16_k1 | 29.568 | 3.491 | 72.65 | 73.90 | 64.73 | 73.51 | 88.90 | 28 | 0 | 8382761 |
| rowpack16_k4 | 94.848 | 4.247 | 88.29 | 90.81 | 73.42 | 77.17 | 91.57 | 32 | 0 | 29331940 |
| rowpack16_k8 | 185.984 | 4.317 | 89.71 | 91.88 | 75.67 | 79.10 | 93.61 | 32 | 0 | 56726448 |
| contiguous64 | 208.992 | 3.842 | 79.85 | 89.52 | 22.82 | 23.69 | 96.99 | 16 | 0 | 31744208 |

### NCU Scheduler/Latency 指標

| report | issue slots busy % | eligible warps/sched | issued warps/sched | active warps/sched | warp cycles/issued inst | long scoreboard cycles | long scoreboard % |
|---|---:|---:|---:|---:|---:|---:|---:|
| byte_ilp4_k1 | 59.62 | 2.22 | 0.60 | 12.77 | 21.11 | 12.2 | 57.6 |
| byte_ilp4_k4 | 50.76 | 1.08 | 0.51 | 11.44 | 22.30 | 17.4 | 78.0 |
| byte_ilp4_k8 | 44.56 | 0.78 | 0.44 | 11.68 | 26.27 | 21.9 | 83.3 |
| rowpack4_k1 | 39.73 | 0.86 | 0.40 | 15.15 | 37.87 | 31.3 | 82.7 |
| rowpack4_k4 | 47.53 | 1.96 | 0.47 | 13.01 | 27.46 | 17.7 | 64.3 |
| rowpack4_k8 | 48.67 | 2.39 | 0.49 | 12.25 | 25.08 | 14.5 | 57.9 |
| rowpack16_k1 | 42.14 | 1.27 | 0.41 | 13.70 | 33.12 | 24.5 | 74.1 |
| rowpack16_k4 | 41.28 | 1.54 | 0.42 | 14.74 | 35.17 | 27.6 | 78.4 |
| rowpack16_k8 | 41.56 | 1.31 | 0.41 | 15.05 | 36.47 | 28.6 | 78.3 |
| contiguous64 | 19.95 | 0.23 | 0.20 | 15.47 | 76.85 | 70.7 | 92.0 |

## 如何解讀 NCU 指標

| 指標 | 意義 | 在本實驗中的解讀方式 |
|---|---|---|
| `duration_us` | NCU 單次 kernel duration | 只能在同一批 NCU run 內相對比較，不要直接取代 benchmark CSV 的平均時間 |
| `memory_throughput_TBps` | NCU 估計的 memory throughput | 對 scan kernel 來說越高越好，代表更接近 HBM/L2 極限 |
| `dram_throughput_pct` | DRAM pipeline 相對峰值利用率 | rowpack 的 DRAM % 明顯比 byte_ilp4 高，代表更能餵滿 DRAM |
| `l2_throughput_pct` | L2 pipeline 相對峰值利用率 | rowpack16 k8 到 `91.88%`，代表 load 形狀更接近高效率 streaming |
| `compute_sm_pct` / `sm_busy_pct` | SM 計算與忙碌程度 | scan 是 memory-bound，SM 低不一定壞；要和 memory throughput 一起看 |
| `eligible_warps_per_scheduler` | 每個 scheduler 平均有多少 warp ready | 太低代表 latency hiding 不足；rowpack4 k8 比 byte_ilp4 k8 更好 |
| `issued_warp_per_scheduler` | 每 cycle 發出的 warp 數 | 高低要搭配 memory saturation 看；memory-bound kernel 通常不會很高 |
| `warp_cycles_per_issued_inst` | warp 兩次 issue 之間平均 cycle | 越高通常表示等待越多；但如果 DRAM/L2 已高，代表在吃滿 memory bottleneck |
| `long scoreboard` | 等待 L1TEX/global memory dependency 的 stall | 高是 memory-bound scan 的常態；重點是高 stall 時 throughput 是否也高 |
| `achieved_occupancy_pct` | 實際 occupancy | 有助於 hide latency，但不是唯一目標；rowpack16 occupancy 高且 memory throughput 高 |
| `registers_per_thread` | 每 thread register 數 | 太高會壓 occupancy；目前 rowpack16 k8 是 32 registers/thread，合理 |
| `local_spill_requests` | register spill 到 local memory | 本次全部是 0，代表 rowpack 沒有因 unpacking 造成 spill |
| `executed_instructions` | 實際執行指令數 | rowpack 大幅減少 load/instruction overhead，是成功的重要證據 |

### NCU 如何支持這次改動成功

`byte_ilp4_k8` 仍然使用 `LDG.E.U8:8b`，NCU memory throughput 只有 `2.250 TB/s`，DRAM utilization `46.75%`，L2 utilization `66.18%`，executed instructions `120982512`。這代表 ILP 改良雖然有幫忙，但 scalar byte load 仍然沒有把 memory subsystem 餵滿。

`rowpack4_k8` 使用 `LDG.E:32b`，NCU memory throughput 升到 `3.887 TB/s`，DRAM utilization `80.80%`，L2 utilization `89.06%`，executed instructions 降到 `74907262`。同時 long scoreboard 從 `21.9 cycles / 83.3%` 降到 `14.5 cycles / 57.9%`。這表示 32-bit row-wise packing 同時改善了 load 粒度、memory utilization 和 dependency stall。

`rowpack16_k8` 使用 `LDG.E.128:128b`，NCU memory throughput 達 `4.317 TB/s`，DRAM utilization `89.71%`，L2 utilization `91.88%`，executed instructions 降到 `56726448`。雖然 long scoreboard 還是 `28.6 cycles / 78.3%`，但這不是失敗訊號，因為同時 DRAM/L2 已接近飽和，代表 kernel 正在被高效率 memory streaming 限制，而不是被低效率 scalar load 卡住。

`contiguous64` 的 NCU memory throughput 是 `3.842 TB/s`，DRAM utilization `79.85%`，L2 utilization `89.52%`。`rowpack16_k8` 在 NCU memory throughput 和 benchmark logical throughput 都超過 contiguous64，說明目前的 row-wise byte-plane scan 已經達到「接近甚至超過 contiguous 8-byte scan baseline」這個 Exp1 階段目標。

## 應該如何引用這次成果

可以用下面這段作為短版研究結論：

```text
Row-wise packing preserves byte-plane progressive semantics while changing the memory instruction granularity. On H200, rowpack16 replaces scalar 8-bit loads with 128-bit global loads and improves k=8 logical throughput from 2399.070 GB/s (byte ilp4) to 4394.268 GB/s, exceeding the contiguous64 baseline at 3929.343 GB/s. NCU confirms the mechanism: rowpack16 k8 reaches 4.317 TB/s memory throughput, 89.71% DRAM throughput, 91.88% L2 throughput, zero spills, and reduces executed instructions from 120.98M to 56.73M compared with byte_ilp4 k8.
```

中文短版：

```text
這次 rowpack 改動成功的關鍵是：它沒有改變 progressive byte-plane 的 k 語意，只是把同一 byte-plane 內連續 rows 用更寬的 global load 搬進來。Benchmark 顯示 rowpack16 在 k=8 達到 4394.268 GB/s，高於 contiguous64 的 3929.343 GB/s；NCU 也確認 load opcode 從 8-bit scalar load 變成 128-bit load，memory throughput 達 4.317 TB/s，DRAM/L2 utilization 分別為 89.71%/91.88%，且沒有 spills。
```

## 不清楚時應參考的程式碼位置

| 問題 | 參考檔案與行號 | 要看什麼 |
|---|---|---|
| 支援哪些參數與 strategy | `benchmarks/experiment1/bench_byteplane_scan.cu:69` | `Options` 的欄位與預設值 |
| CLI help 是否列出 rowpack | `benchmarks/experiment1/bench_byteplane_scan.cu:106` | strategy 說明與 `shared128` overfetch 說明 |
| `packed32` 怎麼處理 | `benchmarks/experiment1/bench_byteplane_scan.cu:251` | `packed32` 被 normalize 成 `rowpack4` |
| `--byte_variant` 何時合法 | `benchmarks/experiment1/bench_byteplane_scan.cu:259` | 只允許 `--strategy byte --plane_bytes 1` |
| kernel 如何被選出 | `benchmarks/experiment1/bench_byteplane_scan.cu:319` | `launch_selected_kernel` dispatcher |
| warmup 與 timed loop | `benchmarks/experiment1/bench_byteplane_scan.cu:370` | warmup launch、timed launch、CUDA event timing |
| logical bytes/GBps 怎麼算 | `benchmarks/experiment1/bench_byteplane_scan.cu:394` | `logical_bytes` 和 `logical_GBps` 公式 |
| `shared128` overfetch 何時標示 | `benchmarks/experiment1/bench_byteplane_scan.cu:399` | 1-byte `shared128` 的 overfetch factor |
| validation 怎麼做 | `benchmarks/experiment1/bench_byteplane_scan.cu:426` | 非 timed validation launch 與 CPU expected sum |
| strategy 合法性檢查 | `benchmarks/experiment1/bench_byteplane_scan.cu:529` | 1-byte/2-byte strategy 限制 |
| plane memory 配置 | `benchmarks/experiment1/bench_byteplane_scan.cu:556` | `uint8_t` / `uint16_t` plane allocation |
| device pointer array 配置 | `benchmarks/experiment1/bench_byteplane_scan.cu:584` | shared path 與 normal path pointer setup |
| occupancy/grid 選擇 | `benchmarks/experiment1/bench_byteplane_scan.cu:283` and `benchmarks/experiment1/bench_byteplane_scan.cu:617` | occupancy API 與 per-k exact kernel grid |
| CSV 欄位 | `benchmarks/experiment1/bench_byteplane_scan.cu:663` | benchmark output header |
| common plane pointer type | `benchmarks/experiment1/exp1_scan_common.cuh:13` | `PlanePointers<T, N>` 與 `U8Planes` |
| block reduction | `benchmarks/experiment1/exp1_scan_common.cuh:21` | warp/block reduction 到 per-block output |
| byte scalar baseline | `benchmarks/experiment1/exp1_kernels_byte.cuh:7` | `scan_planes_u8_byte_unrolled<K>` |
| byte ILP4 | `benchmarks/experiment1/exp1_kernels_byte.cuh:89` | `scan_planes_u8_byte_ilp4<K>` 的 4-stride accumulation |
| byte kernel wrapper | `benchmarks/experiment1/exp1_kernels_byte.cuh:208` | baseline/ilp4 kernel pointer 與 launcher |
| rowpack4 核心 | `benchmarks/experiment1/exp1_kernels_rowpack.cuh:16` | `scan_planes_u8_rowpack4<K>` |
| rowpack4 的 32-bit load | `benchmarks/experiment1/exp1_kernels_rowpack.cuh:32` | `reinterpret_cast<const uint32_t *>` |
| rowpack4 tail | `benchmarks/experiment1/exp1_kernels_rowpack.cuh:37` | `n % 4` 的 scalar tail |
| rowpack16 核心 | `benchmarks/experiment1/exp1_kernels_rowpack.cuh:112` | `scan_planes_u8_rowpack16<K>` |
| rowpack16 的 128-bit load | `benchmarks/experiment1/exp1_kernels_rowpack.cuh:128` | `reinterpret_cast<const uint4 *>` |
| rowpack16 tail | `benchmarks/experiment1/exp1_kernels_rowpack.cuh:137` | `n % 16` 的 scalar tail |
| shared128 診斷路徑 | `benchmarks/experiment1/exp1_kernels_shared.cuh:5` | warp staging kernel |
| shared128 的 128B staging 語意 | `benchmarks/experiment1/exp1_kernels_shared.cuh:10` | 註解與 warp staging 設計 |
| shared128 global load/staging | `benchmarks/experiment1/exp1_kernels_shared.cuh:31` | 每 lane load 4 bytes，寫入 shared memory |
| plane16 對照 | `benchmarks/experiment1/exp1_kernels_plane16.cuh:5` | `uint16_t` plane scan |

## 限制與後續建議

1. NCU 目前只 profile `k=1,4,8`，不是每個 k 都有 NCU。完整 performance curve 看 benchmark CSV，microarchitecture 證據看 NCU sampled points。
2. NCU 的 `duration_us` 不應和 benchmark CSV 的 `ms_per_iter` 逐字對齊；NCU 主要用來證明 load opcode、memory utilization、instruction count、stall pattern。
3. `rowpack16` 依賴 plane base pointer 有足夠 alignment。現在每個 plane 由 `cudaMalloc` 個別配置，alignment 沒問題；如果未來改成單一 slab，需要確保每個 plane offset 仍是 16-byte aligned，必要時加 padding。
4. `shared128` 應保留為診斷工具，但不要把它當作主策略，因為它的 overfetch factor 是 `4.000`，會污染 logical vs physical bytes 的解讀。
5. 若要把這段結果寫進 paper/report，建議主圖使用 `byte baseline`、`byte ilp4`、`rowpack4`、`rowpack16`、`contiguous64`；`shared128` 放到診斷或 appendix。

