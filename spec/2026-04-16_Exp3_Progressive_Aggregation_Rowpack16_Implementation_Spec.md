# Exp3 Progressive Aggregation Rowpack16 Implementation Spec

Date: 2026-04-16  
Status: Implementation-ready draft  
Owner: Nick / Codex  
Target audience: junior developer implementing Experiment 3 v1

## 1. Purpose

Experiment 3 is the next step after Exp1.

Exp1 answered the hardware-access question:

> Can byte-plane/subcolumn layout be scanned efficiently on H200?

The current Exp1 answer is yes, if the implementation uses row-wise packed loads:

- `rowpack4`: one `uint32_t` load = 4 consecutive rows from the same byte-plane.
- `rowpack16`: one `uint4` / 128-bit load = 16 consecutive rows from the same byte-plane.

The latest Exp1 rowpack report shows:

```text
rowpack16 k=8: 4394.268 logical GB/s
contiguous64:  3929.343 logical GB/s
byte ilp4 k=8: 2399.070 logical GB/s
```

NCU confirms the mechanism:

```text
byte_ilp4:  LDG.E.U8:8b
rowpack4:   LDG.E:32b
rowpack16:  LDG.E.128:128b
```

Therefore, Exp3 should not restart from scalar `uint8_t` loads as the main path.

Exp3 v1 should use the Exp1 conclusion directly:

> Use row-wise packed subcolumn loads, with `rowpack16` as the primary performance path.

The goal of this spec is to define a concrete, minimal, verifiable Exp3 v1 implementation.

## 2. Exp3 Research Goal

The high-level Experiment 3 goal from the initial guide is:

> Implement a Buff-style progressive aggregation kernel and show that each additional subcolumn reduces throughput while reducing error.

The final paper deliverable is:

```text
precision-throughput curve
```

where each point corresponds to:

```text
(refinement depth, throughput, error)
```

However, Experiment 2 is not ready yet. Real dataset encoding and final error analysis are not available.

Therefore, Exp3 must start with a performance-only synthetic benchmark.

Exp3 v1 deliverable:

```text
refinement_depth -> throughput
```

Later, Exp2 will provide:

```text
refinement_depth -> error
```

The final curve is a join:

```text
Exp3 throughput CSV + Exp2 error CSV
```

## 3. Scope Summary

### 3.1 Implement now

Implement a standalone synthetic progressive aggregation benchmark:

```text
benchmarks/experiment3/bench_progressive_aggregation.cu
```

It must:

- allocate synthetic Buff-style subcolumns on GPU
- run progressive SUM aggregation for `refinement_depth = 0..K_MAX`
- use `rowpack16` for fractional `uint8_t` subcolumns
- output throughput CSV
- optionally validate numerical output outside the timed loop
- follow Exp1 run directory / metadata conventions

### 3.2 Defer

Do not implement these in v1:

- real dataset loader
- complete Buff encoder
- Exp2 empirical error measurement
- final plotting pipeline
- MIN / MAX / VAR
- progressive filter
- multi-GPU
- multi-word accumulation

## 4. Key Definitions

### 4.1 `refinement_depth`

Use `refinement_depth`, not only `k`, in Exp3.

Reason:

- Exp1 uses `k` to mean number of byte-planes read.
- Exp3 has one integer component plus zero or more fractional refinement subcolumns.

Define:

```text
refinement_depth = 0
  read integer subcolumn only

refinement_depth = 1
  read integer subcolumn + fractional subcolumn 0

refinement_depth = 2
  read integer subcolumn + fractional subcolumns 0..1

...

refinement_depth = F
  read integer subcolumn + fractional subcolumns 0..F-1
```

CSV may also include a short `k` alias if helpful, but `refinement_depth` is the canonical name.

### 4.2 Logical subcolumns read

For Exp3 v1:

```text
logical_subcolumns_read = 1 + refinement_depth
```

where:

- `1` is the integer component
- `refinement_depth` is the number of fractional byte subcolumns

### 4.3 Logical bytes

For v1 default layout:

```text
integer component = uint32_t = 4 bytes per row
fractional subcolumn = uint8_t = 1 byte per row
```

Therefore:

```text
logical_bytes = n * (4 + refinement_depth)
```

If a later version supports `int_bits=8` or `int_bits=16`, update the formula:

```text
logical_bytes = n * (int_bits / 8 + refinement_depth)
```

### 4.4 Row-wise packed load

`rowpack16` means:

```text
one 128-bit load from one subcolumn
that load contains 16 consecutive rows from the same subcolumn
```

It does not mean:

```text
read 16 precision bytes from the same row
```

This distinction must be documented in code comments and README.

## 5. Fixed-Point Synthetic Model

Exp3 v1 needs a simple synthetic model that is close enough to Buff-style progressive aggregation to measure kernel behavior, but not blocked by the final encoder.

Use this v1 model:

```text
value_i(depth) =
  base_segment(i)
  + scale_segment(i) * (
      integer_i
      + frac0_i / 2^8
      + frac1_i / 2^16
      + ...
      + frac(depth-1)_i / 2^(8 * depth)
    )
```

For SUM:

```text
SUM(depth) =
  sum_i base_segment(i)
  + sum_i scale_segment(i) * integer_i
  + sum_i scale_segment(i) * fractional_refinement_i(depth)
```

For v1 synthetic default:

```text
base_segment = 0.0
scale_segment = 1.0
```

This makes validation simple while still preserving the kernel structure:

```text
integer sum + weighted fractional sums
```

The implementation should still allocate and use segment metadata arrays for base and scale, because the final Exp3 model requires them.

### 5.1 Error bound in v1

Exp3 v1 is not responsible for final Exp2 error analysis.

However, because the v1 synthetic model has a clean fixed-point interpretation, it may output a synthetic worst-case bound:

```text
per_value_abs_error_bound(depth) = scale * 2^(-8 * depth)
sum_abs_error_bound(depth) = n * max_scale * 2^(-8 * depth)
avg_abs_error_bound(depth) = max_scale * 2^(-8 * depth)
```

Important:

```text
This synthetic bound is not the final Exp2 result.
```

If included in CSV, name it clearly:

```text
synthetic_sum_abs_error_bound
synthetic_avg_abs_error_bound
error_bound_source=synthetic_fixed_point
```

Do not call it the final paper error unless Exp2 later confirms it.

## 6. Data Layout

### 6.1 Required arrays

Allocate these GPU arrays:

```cpp
uint32_t *d_integer;
uint8_t  *d_frac_planes[MAX_FRAC_PLANES];
double   *d_segment_base;
double   *d_segment_scale;
double   *d_partial_out;
```

Required v1 defaults:

```text
MAX_FRAC_PLANES = 8
integer type = uint32_t
fraction type = uint8_t
base type = double
scale type = double
partial output type = double
```

### 6.2 Fractional planes

Use SOA layout:

```text
frac_plane[0][row]
frac_plane[1][row]
...
frac_plane[7][row]
```

Each fractional plane is a separate `cudaMalloc` allocation in v1.

Reason:

- This matches Exp1 byte-plane allocation.
- Separate `cudaMalloc` gives sufficient base alignment for `uint4` rowpack16 loads.

### 6.3 Future slab allocation warning

Do not implement slab allocation in v1.

If a future version uses:

```cpp
uint8_t *d_slab;
frac_plane[p] = d_slab + p * pitch;
```

then `pitch` must be aligned to at least 16 bytes for `uint4` loads:

```text
pitch = round_up(n, 16)
```

Otherwise `rowpack16` may become misaligned for planes after plane 0.

## 7. Segment and Tile Model

### 7.1 Why segment/tile is needed

The initial Exp3 guide says:

```text
Each thread block processes one segment's subcolumn data.
FOR base and scale are applied once per block or segment, not per row.
```

But one large segment can contain far more rows than one block should process directly.

Therefore, v1 should use a segment-tile model:

```text
segment = logical FOR segment with one base/scale pair
tile    = block-sized chunk inside a segment
block   = processes one tile
```

Multiple blocks may process the same segment.

Each block:

1. accumulates integer/fractional sums for its tile
2. loads that segment's base/scale once
3. applies base/scale once to produce one partial SUM
4. writes one `double` partial result to `d_partial_out[blockIdx.x]`

### 7.2 Required mapping

Use:

```text
segment_rows
items_per_thread
pack_width
tile_rows = block_threads * items_per_thread * pack_width
tiles_per_segment = ceil_div(segment_rows, tile_rows)
num_segments = ceil_div(n, segment_rows)
grid = num_segments * tiles_per_segment
```

For rowpack16:

```text
pack_width = 16
```

For rowpack4 if implemented:

```text
pack_width = 4
```

For scalar fallback:

```text
pack_width = 1
```

### 7.3 Tile row range

For a given block:

```cpp
uint64_t segment_id = blockIdx.x / tiles_per_segment;
uint64_t tile_in_segment = blockIdx.x % tiles_per_segment;

uint64_t segment_start = segment_id * segment_rows;
uint64_t tile_start = segment_start + tile_in_segment * tile_rows;
uint64_t tile_end = min(tile_start + tile_rows,
                        min(segment_start + segment_rows, n));
```

This ensures a tile never crosses a segment boundary.

That matters because each segment has its own:

```text
base
scale
```

### 7.4 Default values

Recommended v1 defaults:

```text
n = 100000000
segment_rows = 1048576
block_threads = 256
items_per_thread = 1
pack_width = 16 for rowpack16
tile_rows = 4096 for rowpack16
```

This gives many blocks:

```text
100000000 / 4096 ~= 24415 tiles
```

which is enough to saturate the GPU.

Do not use one block per 1M-row segment; that would create too few blocks.

## 8. Kernel Variants

### 8.1 Required v1 kernel

Implement:

```cpp
template <int DEPTH, int ITEMS_PER_THREAD>
__global__ void progressive_sum_rowpack16_u32int_u8frac(...);
```

Where:

```text
DEPTH = refinement_depth in [0, 8]
ITEMS_PER_THREAD = fixed template value, default 1
```

The fractional loop over `DEPTH` must be compile-time specialized:

```cpp
#pragma unroll
for (int p = 0; p < DEPTH; ++p)
```

Do not use a runtime `for (p < refinement_depth)` in the hot path.

### 8.2 Optional v1 comparison kernels

Optional, if time permits:

```cpp
progressive_sum_scalar_u32int_u8frac<DEPTH>
progressive_sum_rowpack4_u32int_u8frac<DEPTH>
```

These are useful ablations, but not required for the first Exp3 v1.

The main path is `rowpack16`.

### 8.3 Do not implement shared128 in Exp3 v1

Do not port Exp1 `shared128` into Exp3 v1.

Reason:

- Exp1 already shows `shared128` is a diagnostic path.
- It has `4x` overfetch.
- It performed much worse than rowpack.
- Exp3 should not start from a diagnostic overfetch strategy.

## 9. Kernel Computation Details

### 9.1 Integer accumulation

Each row has:

```cpp
uint32_t integer_i;
```

Each thread accumulates:

```cpp
uint64_t int_sum = 0;
```

Since `uint32_t * n` can exceed `uint64_t` if `n` is extremely large and values are max, v1 must include an overflow note in metadata.

For default synthetic values, validation should use small values like:

```text
integer_i = 1
```

so overflow cannot occur.

For performance runs, values can still be small; memory behavior is what matters.

### 9.2 Fractional accumulation

For each fractional subcolumn `p < DEPTH`, accumulate:

```cpp
uint64_t frac_sum_p = 0;
```

For `DEPTH <= 8`, the simplest implementation uses separate accumulators:

```cpp
uint64_t frac0 = 0;
uint64_t frac1 = 0;
...
uint64_t frac7 = 0;
```

Then only use the first `DEPTH` accumulators.

Alternative:

```cpp
uint64_t frac_sums[8];
```

But be careful: local arrays can become local memory if the compiler cannot keep them in registers.

Recommended v1:

```text
Use explicit scalar accumulators or a templated helper that compiler can fully unroll.
Check NCU local spill requests.
```

### 9.3 Rowpack16 fractional load

Use the same mechanism proven in Exp1:

```cpp
const uint4 *plane128 = reinterpret_cast<const uint4 *>(frac_planes.ptrs[p]);
uint4 pack = plane128[pack_index];
```

Then use:

```cpp
byte_sum_u32(pack.x)
byte_sum_u32(pack.y)
byte_sum_u32(pack.z)
byte_sum_u32(pack.w)
```

Reference:

- `benchmarks/experiment1/exp1_kernels_rowpack.cuh`

### 9.4 Integer load

The integer column is `uint32_t`, so each thread should load consecutive `uint32_t` rows.

For a rowpack16 fractional tile, a natural approach is:

```text
one thread handles 16 rows
load 16 integer values
load 16 bytes from each fractional plane
```

This means the integer side does more loads than the fractional side.

For v1, keep it simple:

```cpp
for each row in the 16-row pack:
  int_sum += d_integer[row]
```

This adds integer load work but matches the model.

Future optimization:

- vectorize integer loads using `uint4` over `uint32_t` values if needed
- or test `int_bits=8/16` synthetic modes

Do not optimize integer vectorization in v1 unless profiling shows it dominates.

### 9.5 Applying base and scale

After block reduction of integer and fractional sums:

```text
partial_sum =
  tile_rows_actual * base
  + scale * (
      int_sum
      + frac0 / 2^8
      + frac1 / 2^16
      + ...
    )
```

Use `double` for this final computation.

This final FP work occurs once per block, not once per row.

That preserves the initial guide's intent:

```text
integer accumulation in the hot loop
one segment/tile-level floating-point correction
```

### 9.6 Block reduction

Exp1 currently reduces one scalar `unsigned long long`.

Exp3 needs to reduce multiple quantities:

```text
int_sum
frac0_sum
...
frac7_sum
```

Implement a small reduction helper in Exp3 common code.

Options:

1. Reduce each accumulator independently using a warp/block reduction helper.
2. Store per-thread partials in shared memory and reduce all fields.

Recommended v1:

```text
Use a simple templated block reduction helper per uint64 accumulator.
Call it once for int_sum and once per active fractional accumulator.
```

This is not the most instruction-minimal approach, but it is easy to review and safe.

Performance impact should be small because the kernel is memory dominated for large `n`.

If reduction overhead appears in NCU, optimize later.

## 10. Output Semantics

Each block writes:

```cpp
d_partial_out[blockIdx.x] = partial_sum;
```

The timed benchmark does not need to copy the full output back to host.

For validation:

1. Run one non-timed launch.
2. Copy `d_partial_out` to host.
3. Sum all partials on CPU.
4. Compare to expected synthetic SUM.

Do not copy `d_partial_out` during the timed loop.

## 11. File Layout

Create:

```text
benchmarks/experiment3/
  CMakeLists.txt
  README.md
  bench_progressive_aggregation.cu
  exp3_common.cuh
  exp3_kernels_progressive.cuh
```

Add scripts:

```text
scripts/run_exp3.sh
run_exp3.sh
```

### 11.1 `bench_progressive_aggregation.cu`

Responsibilities:

- parse CLI
- allocate synthetic arrays
- initialize synthetic data
- allocate `d_partial_out`
- dispatch templated kernels
- warmup and timed loops
- CSV output
- optional validation
- cleanup

Do not put large kernel bodies here.

### 11.2 `exp3_common.cuh`

Responsibilities:

- CUDA reduction helpers
- `ceil_div`
- fixed constants
- plane pointer structs
- `byte_sum_u32`
- small device helpers

Reuse ideas from:

- `benchmarks/experiment1/exp1_scan_common.cuh`
- `benchmarks/experiment1/exp1_kernels_rowpack.cuh`

Do not include host CLI parsing here.

### 11.3 `exp3_kernels_progressive.cuh`

Responsibilities:

- `progressive_sum_rowpack16_u32int_u8frac<DEPTH, ITEMS_PER_THREAD>`
- kernel pointer helper
- launch helper

Expected helpers:

```cpp
const void *progressive_sum_rowpack16_kernel_ptr(int refinement_depth);

void launch_progressive_sum_rowpack16(
    int refinement_depth,
    int grid,
    int block_threads,
    ...);
```

Use switch dispatch for `DEPTH=0..8`.

Do not use runtime-depth hot loops.

## 12. CLI Design

Required CLI:

```bash
./build/exp3/bench_progressive_aggregation \
  --device 0 \
  --n 100000000 \
  --segment_rows 1048576 \
  --frac_planes 8 \
  --refine_min 0 \
  --refine_max 8 \
  --load_strategy rowpack16 \
  --block 256 \
  --items_per_thread 1 \
  --warmup 10 \
  --iters 200 \
  --csv results/exp3/progressive_aggregation.csv
```

### 12.1 Required options

Implement:

```text
--device
--n
--segment_rows
--frac_planes
--refine_min
--refine_max
--load_strategy
--block
--items_per_thread
--warmup
--iters
--csv
--validate
```

### 12.2 `load_strategy`

For v1:

```text
rowpack16
```

Optional:

```text
scalar
rowpack4
```

If only `rowpack16` is implemented, reject other values with a clear error.

### 12.3 Argument constraints

Validate:

```text
n > 0
segment_rows > 0
frac_planes in [0, 8]
refine_min >= 0
refine_max <= frac_planes
refine_min <= refine_max
block is multiple of 32
items_per_thread == 1 in v1 unless additional template dispatch is implemented
segment_rows >= block * items_per_thread * pack_width
```

If `segment_rows` is not a multiple of `tile_rows`, allow it; the last tile in each segment handles the segment tail.

## 13. CSV Schema

Use this schema:

```text
benchmark,dataset,mode,aggregation,load_strategy,refinement_depth,
n,segment_rows,tile_rows,frac_planes,int_bits,frac_bits,
logical_subcolumns_read,logical_bytes,block,grid,warmup,iters,
ms_per_iter,rows_per_sec,billion_rows_per_sec,logical_GBps,
accumulator_bits,base_value,scale_value,
synthetic_sum_abs_error_bound,synthetic_avg_abs_error_bound,error_bound_source,
validated,device,sm,cc_major,cc_minor
```

### 13.1 Fixed v1 values

```text
benchmark=progressive_aggregation
dataset=synthetic
mode=synthetic_fixed_point_subcolumns
aggregation=sum
load_strategy=rowpack16
int_bits=32
frac_bits=8
accumulator_bits=64
base_value=0
scale_value=1
error_bound_source=synthetic_fixed_point
```

### 13.2 Throughput formulas

```text
seconds = ms_per_iter / 1000
rows_per_sec = n / seconds
billion_rows_per_sec = rows_per_sec / 1e9
logical_GBps = logical_bytes / seconds / 1e9
```

### 13.3 Logical bytes formula

```text
logical_bytes = n * (4 + refinement_depth)
```

because v1 integer component is `uint32_t`.

## 14. Synthetic Data Initialization

Implement a non-timed initialization kernel.

Default values:

```text
integer_i = 1
frac_plane[p][i] = p + 1
base_segment = 0.0
scale_segment = 1.0
```

This makes validation easy:

```text
expected_sum(depth) =
  n * 1
  + n * (1 / 2^8)
  + n * (2 / 2^16)
  + ...
  + n * (depth / 2^(8 * depth))
```

For `depth = 0`:

```text
expected_sum = n
```

Use `double` expected value and a small tolerance.

Suggested tolerance:

```text
absolute tolerance = max(1e-6 * abs(expected), 1e-3)
```

If floating-point summation order causes larger differences, report the observed error and justify the tolerance.

## 15. Validation

If `--validate` is passed:

1. Launch the selected kernel once outside timed loop.
2. Copy `d_partial_out` to host.
3. Sum `grid` partial values.
4. Compare with synthetic expected sum.
5. Record `validated=true` in CSV.

If validation is not run:

```text
validated=false
```

Validation must not be included in `ms_per_iter`.

## 16. Benchmark Runner

Add:

```text
scripts/run_exp3.sh
```

Follow the style of:

```text
scripts/run_exp1.sh
```

The runner should:

- create `results/exp3/run_<timestamp>_job<id>_<gpu_tag>/`
- write `setup_estimate.txt`
- build `benchmarks/experiment3`
- run the benchmark
- write `run_meta.txt`
- write `repro_command.txt`
- write `ncu_command_template.txt`

Add root wrapper:

```text
run_exp3.sh
```

following the existing root `run_exp1.sh` convention.

## 17. CMake

Add:

```text
benchmarks/experiment3/CMakeLists.txt
```

Use the same compile style as Exp1:

```cmake
cmake_minimum_required(VERSION 3.24)
project(gpu_byteplane_scan_experiment3 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(bench_progressive_aggregation bench_progressive_aggregation.cu)

target_compile_options(bench_progressive_aggregation PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3 -lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

set_target_properties(bench_progressive_aggregation PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
)
```

Build command:

```bash
cmake -S benchmarks/experiment3 -B build/exp3 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build/exp3 -j
```

## 18. Benchmark Plan

### 18.1 Smoke tests

Run on GPU:

```bash
./build/exp3/bench_progressive_aggregation \
  --device 0 \
  --n 1024 \
  --segment_rows 1024 \
  --frac_planes 2 \
  --refine_min 0 \
  --refine_max 2 \
  --load_strategy rowpack16 \
  --block 256 \
  --items_per_thread 1 \
  --warmup 0 \
  --iters 1 \
  --validate \
  --csv /tmp/exp3_smoke.csv
```

Acceptance:

- exits successfully
- outputs 3 CSV rows for depths 0, 1, 2
- validation passes

### 18.2 Full benchmark

Run:

```bash
./build/exp3/bench_progressive_aggregation \
  --device 0 \
  --n 100000000 \
  --segment_rows 1048576 \
  --frac_planes 8 \
  --refine_min 0 \
  --refine_max 8 \
  --load_strategy rowpack16 \
  --block 256 \
  --items_per_thread 1 \
  --warmup 10 \
  --iters 1000 \
  --csv results/exp3/progressive_aggregation_rowpack16.csv
```

Expected qualitative result:

```text
refinement_depth increases -> logical bytes increases
rows/sec should generally decrease
logical_GBps should approach Exp1 rowpack16-style bandwidth if aggregation overhead is small
```

Do not require perfect monotonicity for every point until real profiling confirms overheads.

### 18.3 Comparison targets

Compare against:

- Exp1 `rowpack16` throughput
- Exp1 `contiguous64` throughput
- Exp1 `byte ilp4` as scalar-load ablation

Do not compare Exp3 row-for-row against Exp1 as identical work.

Reason:

Exp3 does more work:

- integer load
- fractional weighted aggregation
- segment base/scale correction
- multiple accumulator reductions

The relevant question is whether Exp3 remains bandwidth-oriented and scales with refinement depth.

## 19. Nsight Compute Plan

Profile:

```text
refinement_depth = 0
refinement_depth = 4
refinement_depth = 8
```

For:

```text
load_strategy=rowpack16
```

Use environment noted in Exp1 report:

```bash
ml load cuda/12.6
source activate gpu-byteplane-scan
```

Use:

```bash
ncu --set full \
  --target-processes all \
  --launch-count 1 \
  --launch-skip 0 \
  --import-source yes \
  --source-folders <repo-root> \
  --export <output> \
  ./build/exp3/bench_progressive_aggregation ...
```

Inspect:

- SASS load opcode
- DRAM throughput
- L2 throughput
- achieved occupancy
- registers/thread
- local spill requests
- executed instructions
- Long Scoreboard
- eligible warps / scheduler

Expected:

```text
fractional subcolumn loads should use LDG.E.128 or equivalent 128-bit load
local spill requests should be 0
DRAM/L2 utilization should rise with refinement depth
```

Do not claim success unless NCU confirms the wide load.

## 20. Relationship to Exp2

Exp3 v1 does not wait for Exp2.

Exp3 v1 produces:

```text
dataset=synthetic
refinement_depth -> throughput
```

Exp2 will later produce:

```text
dataset
aggregation
refinement_depth
error
error_bound
```

The join key should be:

```text
dataset, aggregation, refinement_depth
```

For v1:

```text
dataset=synthetic
aggregation=sum
```

When Exp2 is ready, add an analysis script that joins:

```text
results/exp3/...csv
results/exp2/...csv
```

and emits:

```text
precision_throughput_curve.csv
```

Do not block Exp3 kernel development on this script.

## 21. Acceptance Criteria

### 21.1 Code acceptance

Required files exist:

```text
benchmarks/experiment3/CMakeLists.txt
benchmarks/experiment3/README.md
benchmarks/experiment3/bench_progressive_aggregation.cu
benchmarks/experiment3/exp3_common.cuh
benchmarks/experiment3/exp3_kernels_progressive.cuh
scripts/run_exp3.sh
run_exp3.sh
```

### 21.2 Build acceptance

This must pass:

```bash
cmake -S benchmarks/experiment3 -B build/exp3 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build/exp3 -j
```

### 21.3 Runtime acceptance

Smoke test with `--validate` must pass.

Full run must produce one CSV row per refinement depth.

### 21.4 Semantic acceptance

- `refinement_depth=0` reads integer component only.
- `refinement_depth=d` reads exactly `d` fractional subcolumns.
- rowpack16 reads consecutive rows from the same fractional plane.
- rowpack16 never reads additional precision bytes from the same row.
- base/scale correction is applied once per tile/block, not per row.

### 21.5 Performance acceptance

Do not require a fixed GB/s target before first run.

But the first H200 full run should report:

- rows/sec
- billion rows/sec
- logical GB/s
- `refinement_depth=0,4,8` comparison

If `rowpack16` Exp3 throughput is far below Exp1 rowpack16, inspect:

- integer load overhead
- reduction overhead
- tile size
- register spills
- SASS load opcode

### 21.6 NCU acceptance

For `refinement_depth=8`, NCU should confirm:

- 128-bit fractional plane loads
- zero local spills
- reasonable occupancy

If NCU shows scalar `LDG.E.U8` for fractional loads, the implementation does not satisfy this spec.

## 22. Non-Goals

Do not implement:

- real Buff encoding
- real dataset loading
- Exp2 final error analysis
- multi-GPU
- filter kernels
- shared128 strategy
- 2-byte fractional plane variant
- CUB/Thrust reductions
- a generic query engine

Do not change:

- Exp1 code
- Exp1 results
- contiguous baseline

## 23. Risks and Mitigations

### 23.1 Reduction overhead

Exp3 reduces multiple accumulators, unlike Exp1.

Mitigation:

- keep v1 simple
- profile first
- optimize reduction only if NCU shows it matters

### 23.2 Integer column dominates bandwidth

With `uint32_t` integer component, `refinement_depth=0` already reads 4 bytes per row.

Mitigation:

- report logical bytes accurately
- optionally add future `int_bits=8/16` modes
- do not misinterpret rows/sec decline as purely fractional overhead

### 23.3 Register pressure

Multiple fractional accumulators may increase registers/thread.

Mitigation:

- use compile-time `DEPTH`
- avoid local arrays that spill
- verify NCU local spill requests

### 23.4 Segment-tile mismatch

Tiles must not cross segment boundaries.

Mitigation:

- use `segment_id` and `tile_in_segment` mapping
- clamp `tile_end` to both segment end and `n`

### 23.5 Confusing v1 synthetic error with final Exp2 error

Mitigation:

- name synthetic bounds explicitly
- keep final precision-throughput curve join as later work

## 24. Developer Implementation Order

Follow this order:

1. Create `benchmarks/experiment3/CMakeLists.txt`.
2. Create `bench_progressive_aggregation.cu` with CLI parsing and empty skeleton.
3. Create `exp3_common.cuh` with reduction helpers and `byte_sum_u32`.
4. Implement synthetic allocation and initialization.
5. Implement `progressive_sum_rowpack16_u32int_u8frac<DEPTH, ITEMS_PER_THREAD>`.
6. Implement switch dispatch for `DEPTH=0..8`.
7. Implement timing loop and CSV output.
8. Implement `--validate`.
9. Add `benchmarks/experiment3/README.md`.
10. Add `scripts/run_exp3.sh`.
11. Add root `run_exp3.sh`.
12. Run smoke test.
13. Run H200 full benchmark.
14. Run NCU at depths `0,4,8`.
15. Write a handoff report with results.

## 25. Developer Report Format

When done, report with:

1. Summary
2. Files Added
3. CLI
4. Data Layout
5. Kernel Mapping
6. Validation Result
7. Smoke Test Result
8. Full Benchmark Result
9. NCU Result
10. Caveats
11. Deferred Work

Avoid unsupported claims.

Do not say:

```text
Exp3 is complete
```

unless throughput, validation, and NCU evidence are all available.

Say instead:

```text
Exp3 v1 synthetic progressive SUM throughput benchmark is implemented.
```

## 26. References Inside This Repo

Read these before implementing:

- `spec/M03-initial-guide.md`
- `spec/2026-04-14_Exp3_Progressive_Aggregation.md`
- `research/2026-04-16_exp1_rowpack_benchmark_ncu_report.md`
- `benchmarks/experiment1/exp1_kernels_rowpack.cuh`
- `benchmarks/experiment1/exp1_scan_common.cuh`
- `scripts/run_exp1.sh`
- `benchmarks/experiment1/CMakeLists.txt`

Use Exp1 rowpack as the model for:

- row-wise packed load semantics
- `uint4` load
- byte summation helper
- tail handling
- compile-time depth dispatch pattern

Use Exp1 runner as the model for:

- run directory
- metadata
- repro command
- NCU command template
