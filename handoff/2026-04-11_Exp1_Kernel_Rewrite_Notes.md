# Exp1 kernel rewrite notes (Crystal-aligned)

Date: 2026-04-11

Purpose
- Provide a concrete reference for rewriting Exp1 scan kernels in [benchmarks/experiment1/bench_byteplane_scan.cu](benchmarks/experiment1/bench_byteplane_scan.cu) using Crystal's scan/predicate execution model.
- Emphasize coalesced loads (striped mapping), tile-based blocking, and tail handling.

Crystal patterns to mirror
- Striped mapping for coalesced loads and stores:
  - Per-thread pointer: thread_itr = block_itr + tid
  - Per-item access: thread_itr[ITEM * BLOCK_THREADS]
  - See [crystal/crystal/load.cuh](../crystal/crystal/load.cuh#L1-L39) and [crystal/crystal/store.cuh](../crystal/crystal/store.cuh#L1-L39).
- Tile partitioning and last-tile guard:
  - tile_offset = blockIdx.x * TILE_SIZE
  - num_tile_items = (last block) ? total - tile_offset : TILE_SIZE
  - Used throughout [crystal/src/ssb/q12.cu](../crystal/src/ssb/q12.cu#L35-L55).
- Predicate flags are thread-local (not a global bitmap):
  - int selection_flags[ITEMS_PER_THREAD]
  - Computed and used within a single kernel (scan).
  - See [crystal/crystal/pred.cuh](../crystal/crystal/pred.cuh#L3-L143).

Why this matters for Exp1
- Exp1 measures bandwidth of k-plane reads. To match Crystal, the kernel should:
  - Operate on a fixed tile per block (TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD).
  - Use striped mapping so each warp reads contiguous addresses for each ITEM iteration.
  - Handle tail safely using the same tid + ITEM * BLOCK_THREADS check.

Rewrite blueprint (per strategy)
1) byte (u8 planes, k in [1..8])
- Use template K and ITEMS_PER_THREAD to fully unroll inner loops.
- Replace grid-stride with tile iteration:
  - For block: tile_offset = blockIdx.x * TILE_SIZE
  - For each ITEM: idx = tile_offset + tid + ITEM * BLOCK_THREADS
- For each plane p < K:
  - if idx < num_tile_items, sum += planes[p][idx]
- This matches Crystal's BlockLoadDirect pattern while preserving the byte-sum behavior.

2) packed32 (u8 planes read as uint32)
- Keep the packed32 data path, but apply the same striped mapping:
  - Let idx4 = (tile_offset + tid + ITEM * BLOCK_THREADS) >> 2
  - Ensure idx4 < n4 for full pack, then sum 4 bytes
  - Handle the tail separately (like the current kernel)
- This makes the packed loads coalesced by construction.

3) shared128 (overfetch to force coalescing)
- The current shared128 kernel already stages 128B/warp. Align it to tiles by:
  - Using a tile-based loop in units of 128 elements per warp (consistent with Crystal's contiguous tile idea)
  - Keeping the per-warp shared staging but ensuring the outer loop increments by warp_stride * 128

Suggested helper skeleton (conceptual)
- Define constants:
  - BLOCK_THREADS, ITEMS_PER_THREAD, TILE_SIZE
- Kernel outline:
  - int tid = threadIdx.x
  - tile_offset = blockIdx.x * TILE_SIZE
  - num_tile_items = (last block) ? n - tile_offset : TILE_SIZE
  - for ITEM in [0..ITEMS_PER_THREAD):
      idx = tile_offset + tid + ITEM * BLOCK_THREADS
      if idx < num_tile_items: load plane bytes -> sum
  - block_reduce_store(sum, per_block_out)

Integration steps in bench_byteplane_scan.cu
1) Pick a fixed BLOCK_THREADS and ITEMS_PER_THREAD pairing (e.g., 256 x 4, same as Crystal's 128 x 4 pattern but scaled).
2) Add a tile-based variant of the byte kernel (new kernel or replace the current grid-stride loop).
3) Update packed32 and shared128 to use tile-style indexing for consistent comparisons.
4) Keep the existing reduction (block_reduce_store) so output remains comparable.

Cross-checks after rewrite
- Ensure each warp reads contiguous addresses for each ITEM iteration.
- Verify tail handling matches Crystal's guard condition.
- Confirm occupancy_grid uses the new kernel pointer for accurate block count.

Relevant files
- [benchmarks/experiment1/bench_byteplane_scan.cu](benchmarks/experiment1/bench_byteplane_scan.cu)
- [crystal/crystal/load.cuh](../crystal/crystal/load.cuh)
- [crystal/crystal/store.cuh](../crystal/crystal/store.cuh)
- [crystal/crystal/pred.cuh](../crystal/crystal/pred.cuh)
- [crystal/src/ssb/q12.cu](../crystal/src/ssb/q12.cu)
