# H100/H200 Runner Alignment Plan

## Summary
把 repo 整理成適合 TWCC H100/H200 混合環境使用的 benchmark repo，不強調 nano4 或 nano5 差異，重點是每次 run 都能追蹤 GPU、job、時間與參數，並讓 exp0/exp1 都可一鍵到底。

## Implemented Items

- 統一敘事為 H100/H200 混合環境。
- 正式 runner 入口為 `scripts/run_exp0.sh`、`scripts/run_exp1.sh`。
- root `run_exp0.sh` 改為 Slurm wrapper（載入環境後呼叫 `scripts/run_exp0.sh`）。
- 每次 run 建立唯一目錄：
  - `results/exp0/run_<timestamp>_job<id_or_nojob>_<gpu_tag>/`
  - `results/exp1/run_<timestamp>_job<id_or_nojob>_<gpu_tag>/`
- 每次 run 產出 `run_meta.txt`、`repro_command.txt`、`ncu_command_template.txt`。
- exp0 runner 預設跑 `seq/masked/gather` 三模式。
- exp1 runner 加入 setup 記錄（包含 estimated memory allocation）。
- `.gitignore` 補齊 `build/`、`results/`、`*.log`、`*.err`、常見 benchmark CSV。

## Validation Targets

- 連續執行兩次 runner 時，結果不覆寫，且目錄名帶 timestamp/job/gpu。
- exp0 一次 run 會產生三份 CSV。
- exp1 一次 run 可從 build 到 run 完成，無外部 dataset 依賴。
- `run_meta.txt` 的 GPU 名稱與 CSV 中 device 欄位一致。

## Notes

- `CMAKE_CUDA_ARCHITECTURES=90` 適用 Hopper（H100/H200）。
- 若後續要整合 `ncu` 進正式流程，可在 runner 上加 `RUN_WITH_NCU=1` 類型開關。
