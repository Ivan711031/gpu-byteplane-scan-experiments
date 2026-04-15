# 2026-04-15 Exp1 byte-plane baseline and packed-read discussion notes

## Context

This note summarizes a discussion about Experiment 1 in this repository.

The current project goal is to optimize:

- `benchmarks/experiment1/bench_byteplane_scan.cu`

and compare it against:

- `benchmarks/experiment1/bench_contiguous_baseline.cu`

The immediate research question is whether byte-plane scan can approach the contiguous 8-byte scan baseline while preserving the progressive byte-plane semantics.

## Terminology Clarification

Some spoken terms from the discussion were normalized as follows:

- "Xperia 1" means Experiment 1 / Exp1.
- "BestLite" likely means baseline, specifically `bench_contiguous_baseline.cu`.
- "Bites only" means byte-only / byte-plane scan, especially the `k=1` case.
- "packed32" should be understood as row-wise packing inside one byte-plane, not packing multiple precision bytes from one row.

## Main Concern

The key concern was:

If we use `packed32` or `128-bit` loads, are we violating the original paper idea of reading byte-planes progressively, one byte of precision at a time?

The answer depends on the packing direction.

## Correct Interpretation of Packed Reads

Packed reads are valid if they read consecutive rows from the same byte-plane.

Example for one byte-plane:

```text
plane 7:
row0_b7 row1_b7 row2_b7 row3_b7 row4_b7 ...
```

A valid `packed32` load reads:

```text
row0_b7 row1_b7 row2_b7 row3_b7
```

This means:

- same byte-plane
- four consecutive rows
- one precision byte per row

It does not read:

```text
row0_b7 row0_b6 row0_b5 row0_b4
```

The latter would read multiple precision bytes from the same row and would violate the progressive byte-plane semantics.

## Why Packed Reads Do Not Violate the Research Goal

The research goal is:

```text
For each row, read only the first k byte-planes needed for the required precision.
```

The goal is not:

```text
Every GPU memory instruction must load exactly one byte.
```

Packed reads change how bytes are transferred from HBM to registers. They do not change which byte-planes are read.

For `k = 1`, a row-wise packed implementation still reads only one byte-plane:

```text
read plane 7 only
each row contributes one byte
logical bytes = N * 1
```

For `k = 8`, it reads all eight byte-planes:

```text
read plane 7..0
each row contributes eight bytes
logical bytes = N * 8
```

Thus, row-wise packed reads preserve the same `k` semantics as scalar byte loads.

## Packed32

`packed32` means:

```text
uint32_t = 4 bytes = 4 consecutive rows from the same byte-plane
```

For example:

```text
thread0 reads row0_b7 row1_b7 row2_b7 row3_b7
thread1 reads row4_b7 row5_b7 row6_b7 row7_b7
```

Advantages:

- larger payload per load instruction than `uint8_t`
- fewer global load instructions for the same logical bytes
- likely better memory coalescing / scoreboard behavior
- preserves byte-plane precision semantics

Costs:

- register unpacking is required
- more integer ALU work
- tail handling is needed when `n` is not divisible by 4

Suggested clearer name:

```text
rowpack4
```

This avoids confusion with packing multiple bytes from one row.

## 128-bit Row Packing

A 128-bit row-wise packed version is also valid if it reads consecutive rows from the same byte-plane.

For `uint8_t` plane elements:

```text
128 bits = 16 bytes = 16 rows
```

So a 128-bit load reads:

```text
same plane, 16 consecutive rows
```

It does not read 8 bytes from one FP64 value.

Suggested clearer name:

```text
rowpack16
```

Advantages:

- even larger payload per load instruction
- more likely to approach contiguous scan throughput
- still preserves `k` semantics

Costs:

- more unpacking work
- stricter alignment requirements
- more careful tail handling
- may be less suitable for sparse progressive filtering

Important alignment note:

- If each plane is separately allocated with `cudaMalloc`, the base pointer should be sufficiently aligned.
- If planes are later allocated as one slab, each plane offset must be aligned for 128-bit loads. This may require padding each plane pitch to a multiple of 16 bytes.

## Shared Memory Staging

The initial guide suggested:

```text
Shared memory staging:
one warp loads a 128-byte chunk into shared memory,
then each lane reads individual bytes from shared memory.
```

This is a diagnostic / alternative strategy.

It makes global memory access more coalesced, but it may overfetch.

Example:

```text
one warp stages 128 bytes
but may only logically consume 32 bytes
```

This can create a 4x overfetch factor.

Therefore:

- It can help diagnose whether the bottleneck is memory transaction shape.
- It may not be the cleanest primary result for the paper, because physical bytes transferred may exceed logical bytes.
- It should be reported with both logical and physical/overfetch interpretation.

Suggested role:

```text
diagnostic / upper-bound experiment, not the main semantic implementation
```

## 2-byte Planes

The initial guide also suggested:

```text
2-byte planes:
split FP64 into four 2-byte planes instead of eight 1-byte planes
```

This changes precision granularity.

Original byte-plane layout:

```text
8 planes, 1 byte each
k = 1..8
```

2-byte-plane layout:

```text
4 planes, 2 bytes each
k = 1..4
```

This does not completely violate the project goal, but it weakens the byte-level progressive precision argument.

It trades:

```text
finer precision control
```

for:

```text
more hardware-friendly load granularity
```

Suggested role:

```text
fallback / sensitivity experiment
```

not the first main implementation.

## Comparison With Contiguous Baseline

The contiguous baseline is implemented in:

- `benchmarks/experiment1/bench_contiguous_baseline.cu`

It allocates one contiguous `uint64_t` array:

```text
n elements * 8 bytes
```

The kernel performs:

```text
one uint64_t load per row
register accumulation
block reduction into d_out
```

The byte-plane benchmark instead allocates eight separate `uint8_t` planes:

```text
8 planes * n bytes
```

For `k = 1`, it reads:

```text
n * 1 byte
```

For `k = 8`, it reads:

```text
n * 8 bytes
```

## What Is Comparable

The two benchmarks are comparable for performance:

- `ms_per_iter`
- logical GB/s
- scaling with `k`
- overhead of byte-plane layout at `k = 8`
- runtime benefit when `k < 8`

The most apples-to-apples performance comparison is:

```text
contiguous64 vs byte-plane k=8
```

because both read:

```text
n * 8 logical bytes
```

## What Is Not Comparable

The numerical output sums are not directly comparable.

Reason:

- contiguous baseline sums `uint64_t` values
- byte-plane benchmark sums individual byte values

These benchmarks are bandwidth microbenchmarks, not correctness tests for a real query result.

The `d_out` output exists mainly to keep loads and reductions live so the compiler cannot remove the work.

## Current Data Interpretation

Current H200 result:

```text
contiguous64:
  0.203540 ms
  3930.441 logical GB/s

byte ilp4 k=8:
  0.333311 ms
  2400.164 logical GB/s
```

Interpretation:

- At `k = 8`, byte-plane reads the same logical bytes as contiguous baseline but is slower.
- This exposes byte-plane layout / load-granularity overhead.
- At low `k`, byte-plane can still win in absolute time because it reads fewer logical bytes.
- Logical GB/s remains lower than contiguous, showing lower hardware efficiency per byte.

## Is the Baseline Accurate?

As a throughput baseline, it is reasonable because:

- allocation and initialization are outside the timed region
- CUDA events are used for timing
- warmup iterations are used
- many timed iterations are averaged
- the kernel uses grid-stride traversal
- accumulation is kept live through `d_out`
- measured H200 throughput is close to expected HBM scan bandwidth

However, for paper-level confidence, optional validation would be useful.

## Suggested Validation Improvement

Add a non-timed validation mode, such as:

```text
--validate
```

For small `n`:

1. Run the kernel once.
2. Copy `d_out` back to host.
3. Sum all per-block outputs.
4. Compare against expected value.

Expected contiguous baseline:

```text
n * uint64_value_from_memset_pattern
```

Expected byte-plane benchmark:

```text
n * k * byte_value
```

For the current `cudaMemset(..., 0xAB, ...)` pattern:

```text
byte expected = n * k * 0xAB
```

This validation should not be included in timed benchmark loops.

## Recommended Implementation Direction

Priority order:

1. Implement a templated row-wise packed 32-bit variant (`rowpack4`).
2. Implement a templated row-wise packed 128-bit variant (`rowpack16`).
3. Keep shared-memory staging as a diagnostic strategy.
4. Keep 2-byte planes as a fallback / sensitivity experiment.

Recommended naming:

```text
rowpack4   = same byte-plane, 4 consecutive rows
rowpack16  = same byte-plane, 16 consecutive rows
shared128  = same byte-plane, 128B staged chunk
plane16    = 2-byte precision planes
```

This naming avoids confusion between:

```text
packing rows within one plane
```

and:

```text
packing multiple precision bytes from one row
```

Only the first preserves the original progressive byte-plane semantics.

## Open Questions for Next Discussion

- Should `rowpack4` and `rowpack16` be exposed as new strategies or byte variants?
- Should CSV include a clearer implementation variant field?
- Should the benchmark report physical bytes / overfetch separately from logical bytes?
- Should validation be added now or kept separate from the optimization PR?
- Should `shared128` remain in the main benchmark or be moved to a diagnostic-only path?
