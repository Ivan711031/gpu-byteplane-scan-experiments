# Handoff: Exp1 Byte-Only Refactor Progress and User Requirements

Date: 2026-04-11  
Author: Codex  
Scope: 只記錄本輪 `exp1` byte-only 改動、執行結果與後續強制規範

## 1) Code Progress (Completed)

已完成 `benchmarks/experiment1/bench_byteplane_scan.cu` 的 byte-only refactor：

1. `strategy=byte` 改為 pass-by-value pointer bundle (`U8Planes`)。
2. byte kernel 簽名改為 `scan_planes_u8_byte_unrolled(const U8Planes planes, ...)`。
3. byte hot path 不再走 device-global pointer array (`d_plane_ptrs`)。
4. `template<K>` + `switch(k)` (`K=1..8`) specialization 保留。
5. occupancy 改為 byte path 先掃 `K=1..8`，取 `max_needed_grid`。
6. `d_out` 改為一次性用 `max_needed_grid` 配置；sweep 過程不重配。
7. `packed32/shared128` kernel 內容未改。

## 2) Runtime / Profiling Artifacts (Completed)

### 2.1 Full run (與歷史 baseline 同量級參數)

- Job: `16969`
- Output dir:
  - `results/exp1/run_20260411_221234_job16969_H200/`
- Key file:
  - `results/exp1/run_20260411_221234_job16969_H200/exp1.csv`
- Parameters:
  - `--strategy byte --plane_bytes 1 --n 100000000 --k_min 1 --k_max 8 --warmup 10 --iters 200 --block 256 --grid_mul 1`

### 2.2 NCU (single-k, old "ncu_exp1_view" style; portable and small)

- Job: `16970`
- Output files:
  - `results/exp1/ncu_exp1_view_20260411_221304.ncu-rep`
  - `results/exp1/ncu_exp1_view_20260411_221304_bench.csv`
- Notes:
  - 這版和舊可開檔案格式一致（單一 `k`、小檔案）。

### 2.3 NCU (k-set: k=1,6,8; n=1e6 round)

- Job: `16972`
- Output dir:
  - `results/exp1/ncu_run_20260411_222405_job16972_byte_kset/`
- Reports:
  - `ncu_exp1_k1.ncu-rep`
  - `ncu_exp1_k6.ncu-rep`
  - `ncu_exp1_k8.ncu-rep`

## 3) User Hard Requirements (Must Follow Next)

以下是 user 明確要求，後續 agent 必須遵守：

1. 測試資料量下限：`--n 100000000`。
2. 不做 smoke test（不要再用小 `n` 當主要驗證）。
3. NCU 要 single-k profile（一次只保留一個 `k`）：
   - `--strategy byte`
   - `--plane_bytes 1`
   - `--k 1` 或 `--k 6` 或 `--k 8`
   - 其他參數固定。
4. NCU 產物放在 `results/exp1/` 下，且要有時間戳資料夾。
5. 分析重點要能支撐：
   - 為何 `k=6` 最好
   - 為何 `k=8` 掉下來

## 4) In-Flight Status to Check

已送出第二輪（user 指定）n1e8 NCU k-set 命令：

- Job: `16974`
- Intended output pattern:
  - `results/exp1/ncu_run_<timestamp>_job16974_byte_kset_n1e8/`
- At handoff time:
  - session 還在跑/等待，尚未看到落地輸出目錄。

建議接手 agent 第一件事：

1. 查 job 是否完成。
2. 若完成，定位 `job16974` 對應輸出資料夾並驗證三個 `.ncu-rep` 是否齊全。
3. 若失敗/無產物，重跑同命令（`n=1e8, warmup=1, iters=1, k=1/6/8`）。

## 5) Practical Notes

1. `--set full` 會讓 profiling 開銷非常大，`exp1_*_bench.csv` 的 `ms_per_iter` 不可和普通 benchmark 直接比較。
2. 若 UI 打不開 `.ncu-rep`，先檢查本機 Nsight Compute 版本是否至少能讀 2025.3.0.0 產物。
3. 為了可開啟性，優先保留 single-k 的小檔 `.ncu-rep`。
