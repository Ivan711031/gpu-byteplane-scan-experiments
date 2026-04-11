# Dev Log Addendum: Exp1 Analysis, Plotting, and Next Steps

Author: Codex (handoff addendum for Nick)  
Date: 2026-04-11  

## 1. What Was Added (Today's Changes)

1. **Python Plotting Scripts Added**:
   - `scripts/plot_exp0_results.py`
   - `scripts/plot_exp1_results.py`
   - Used for visualizing bandwidth vs. size, gather vs. seq ratios, and speedup performance across varying parameters. 

2. **Runner Metadata Context Enriched (`exp1`)**:
   - Modified `scripts/run_exp1.sh` to accept a `RUN_DESC` environment variable.
   - Example usage: `RUN_DESC="Added packed32 kernel implementation with Loop Unrolling" ./scripts/run_exp1.sh`
   - Motivation: Ensures each run's `run_meta.txt` explicitly documents the code/parameter changes that justified the experiment, avoiding confusion when analyzing results later.

3. **External Libraries Ignored**:
   - `crystal/` source code is preserved locally for research but explicitly added to `.gitignore` to keep the benchmark repo clean.

## 2. Outstanding Follow-Ups & TODOs (Action Items)

- [ ] **Run `packed32` full sweep Validation**:
  - Task: Execute a complete sweep of `strategy=packed32` to capture actual memory payload scaling behavior up to `k=8`.
  - Command: `RUN_DESC="Validating packed32 memory payload expansion" DEVICE=0 CUDA_ARCH=90 N=100000000 PLANE_BYTES=1 STRATEGY=packed32 K_MIN=1 K_MAX=8 WARMUP=10 ITERS=200 ./scripts/run_exp1.sh` (or using Slurm).

- [ ] **Investigate H200 `k=7..8` Regression**:
  - Task: Why does performance drop at `k=7` and `k=8` in the `strategy=byte` loop unrolled version? Profile with Nsight Compute to pinpoint register pressure or occupancy drops.

- [ ] **Borrow concepts from Crystal `Scan Kernels` (q11/q12/q13)**:
  - Task: Evaluate applying *Striped Mapping* for fully coalesced Load/Stores and integrating decoupled thread-local `selection_flags` predicates into our `exp1` workflow if scanning filtering (progressive filters) are introduced in the next phase.
