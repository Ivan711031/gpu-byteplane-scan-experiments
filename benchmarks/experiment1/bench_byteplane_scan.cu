#include <cuda_runtime.h>

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace
{

  [[nodiscard]] const char *cuda_err_str(cudaError_t err) { return cudaGetErrorString(err); }

  [[noreturn]] void die(const char *msg)
  {
    std::fprintf(stderr, "error: %s\n", msg);
    std::exit(2);
  }

  void cuda_check(cudaError_t err, const char *where)
  {
    if (err == cudaSuccess)
      return;
    std::fprintf(stderr, "cuda error at %s: %s\n", where, cuda_err_str(err));
    std::exit(2);
  }

  [[nodiscard]] bool parse_u64(std::string_view s, uint64_t &out)
  {
    if (s.empty())
      return false;
    uint64_t value = 0;
    for (char c : s)
    {
      if (c < '0' || c > '9')
        return false;
      uint64_t digit = static_cast<uint64_t>(c - '0');
      if (value > (std::numeric_limits<uint64_t>::max() - digit) / 10ull)
        return false;
      value = value * 10ull + digit;
    }
    out = value;
    return true;
  }

  enum class ByteVariant
  {
    Baseline,
    Ilp4,
  };

  [[nodiscard]] bool parse_byte_variant(std::string_view s, ByteVariant &out)
  {
    if (s == "baseline")
    {
      out = ByteVariant::Baseline;
      return true;
    }
    if (s == "ilp4")
    {
      out = ByteVariant::Ilp4;
      return true;
    }
    return false;
  }

  struct Options
  {
    int device = 0;
    uint64_t n = 100000000ull; // number of FP64 values (logical rows)

    int plane_bytes = 1;           // 1 => 8 planes of uint8, 2 => 4 planes of uint16
    std::string strategy = "byte"; // byte|packed32|shared128
    ByteVariant byte_variant = ByteVariant::Baseline;

    bool single_k = false;
    int k_single = 0;
    int k_min = 1;
    int k_max = 8;

    int block_threads = 256;
    int grid_mul = 1;
    int warmup = 10;
    int iters = 200;

    std::string csv_path = "exp1_byteplane_scan.csv";
  };

  void print_usage(const char *argv0)
  {
    std::fprintf(stderr,
                 "Usage: %s [options]\n"
                 "\n"
                 "Experiment 1: Byte-plane scan bandwidth scaling (SOA planes).\n"
                 "Allocates 8 bytes/value in SOA form and measures throughput vs. k planes read.\n"
                 "Dummy reduction: reads then reduces in registers, writes 1 value per block.\n"
                 "\n"
                 "Options:\n"
                 "  --device N            CUDA device index (default: 0)\n"
                 "  --n N                 Number of FP64 values (default: 100000000)\n"
                 "  --plane_bytes 1|2     1-byte planes (8 planes) or 2-byte planes (4 planes) (default: 1)\n"
                 "  --strategy NAME       byte | packed32 | shared128 (default: byte)\n"
                 "  --byte_variant NAME   baseline | ilp4; only for --strategy byte --plane_bytes 1 (default: baseline)\n"
                 "  --k N                 Single k (planes to read)\n"
                 "  --k_min N             Sweep min k (default: 1)\n"
                 "  --k_max N             Sweep max k (default: 8)\n"
                 "  --block T             Threads per block (default: 256)\n"
                 "  --grid_mul M          Grid = SMs * maxActiveBlocksPerSM * M (default: 1)\n"
                 "  --warmup N            Warmup iterations (default: 10)\n"
                 "  --iters N             Timed iterations (default: 200)\n"
                 "  --csv PATH            Output CSV path (default: exp1_byteplane_scan.csv)\n"
                 "\n"
                 "Notes:\n"
                 "  - Throughput uses *logical bytes* = n * k * plane_bytes.\n"
                 "  - shared128 intentionally overfetches (128B/warp) to force coalesced loads.\n"
                 "\n",
                 argv0);
  }

  Options parse_args(int argc, char **argv)
  {
    Options opt;
    bool k_min_set = false;
    bool k_max_set = false;
    bool k_set = false;
    bool byte_variant_set = false;
    for (int i = 1; i < argc; i++)
    {
      std::string_view a(argv[i]);
      auto need_value = [&](std::string_view flag) -> std::string_view
      {
        if (i + 1 >= argc)
        {
          std::string msg = "missing value for ";
          msg += flag;
          die(msg.c_str());
        }
        return std::string_view(argv[++i]);
      };

      if (a == "--help" || a == "-h")
      {
        print_usage(argv[0]);
        std::exit(0);
      }
      else if (a == "--device")
      {
        uint64_t v = 0;
        if (!parse_u64(need_value(a), v) || v > static_cast<uint64_t>(std::numeric_limits<int>::max()))
          die("invalid --device");
        opt.device = static_cast<int>(v);
      }
      else if (a == "--n")
      {
        uint64_t v = 0;
        if (!parse_u64(need_value(a), v) || v == 0)
          die("invalid --n");
        opt.n = v;
      }
      else if (a == "--plane_bytes")
      {
        uint64_t v = 0;
        if (!parse_u64(need_value(a), v) || (v != 1 && v != 2))
          die("invalid --plane_bytes");
        opt.plane_bytes = static_cast<int>(v);
      }
      else if (a == "--strategy")
      {
        opt.strategy = std::string(need_value(a));
      }
      else if (a == "--byte_variant")
      {
        if (!parse_byte_variant(need_value(a), opt.byte_variant))
          die("invalid --byte_variant");
        byte_variant_set = true;
      }
      else if (a == "--k")
      {
        uint64_t v = 0;
        if (!parse_u64(need_value(a), v) || v == 0 || v > 8)
          die("invalid --k");
        opt.single_k = true;
        opt.k_single = static_cast<int>(v);
        k_set = true;
      }
      else if (a == "--k_min")
      {
        uint64_t v = 0;
        if (!parse_u64(need_value(a), v) || v == 0 || v > 8)
          die("invalid --k_min");
        opt.k_min = static_cast<int>(v);
        k_min_set = true;
      }
      else if (a == "--k_max")
      {
        uint64_t v = 0;
        if (!parse_u64(need_value(a), v) || v == 0 || v > 8)
          die("invalid --k_max");
        opt.k_max = static_cast<int>(v);
        k_max_set = true;
      }
      else if (a == "--block")
      {
        uint64_t v = 0;
        if (!parse_u64(need_value(a), v) || v == 0 || v > 2048)
          die("invalid --block");
        opt.block_threads = static_cast<int>(v);
      }
      else if (a == "--grid_mul")
      {
        uint64_t v = 0;
        if (!parse_u64(need_value(a), v) || v == 0 || v > 1024)
          die("invalid --grid_mul");
        opt.grid_mul = static_cast<int>(v);
      }
      else if (a == "--warmup")
      {
        uint64_t v = 0;
        if (!parse_u64(need_value(a), v) || v > 1000000)
          die("invalid --warmup");
        opt.warmup = static_cast<int>(v);
      }
      else if (a == "--iters")
      {
        uint64_t v = 0;
        if (!parse_u64(need_value(a), v) || v == 0 || v > 100000000)
          die("invalid --iters");
        opt.iters = static_cast<int>(v);
      }
      else if (a == "--csv")
      {
        opt.csv_path = std::string(need_value(a));
      }
      else
      {
        std::string msg = "unknown arg: ";
        msg += std::string(a);
        die(msg.c_str());
      }
    }

    if (opt.block_threads % 32 != 0)
      die("--block must be a multiple of 32");
    if (opt.plane_bytes == 2 && opt.strategy == "shared128")
    {
      die("--strategy shared128 only supported for --plane_bytes 1");
    }
    if (opt.byte_variant == ByteVariant::Ilp4 && !(opt.plane_bytes == 1 && opt.strategy == "byte"))
    {
      die("--byte_variant ilp4 only supported for --strategy byte --plane_bytes 1");
    }
    if (byte_variant_set && opt.plane_bytes == 2 && opt.strategy == "byte")
    {
      die("--byte_variant only applies to --strategy byte --plane_bytes 1");
    }

    int planes_total = (opt.plane_bytes == 1) ? 8 : 4;
    if (opt.single_k && opt.k_single > planes_total)
      die("--k exceeds number of planes");
    if (!opt.single_k)
    {
      if (opt.k_min > opt.k_max)
        die("--k_min > --k_max");
      if (k_min_set && opt.k_min > planes_total)
        die("--k_min exceeds number of planes");
      if (k_max_set && opt.k_max > planes_total)
        die("--k_max exceeds number of planes");
    }
    // If user didn't set k bounds, clamp defaults in make_k_sweep().
    if (k_set && (k_min_set || k_max_set))
      die("do not mix --k with --k_min/--k_max");

    return opt;
  }

  __device__ __forceinline__ unsigned long long warp_reduce_sum_ull(unsigned long long v)
  {
    for (int offset = 16; offset > 0; offset >>= 1)
      v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
  }

  __device__ __forceinline__ void block_reduce_store(unsigned long long v,
                                                     unsigned long long *__restrict__ out)
  {
    v = warp_reduce_sum_ull(v);
    __shared__ unsigned long long warp_sums[32]; // up to 1024 threads
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0)
      warp_sums[warp] = v;
    __syncthreads();

    if (warp == 0)
    {
      unsigned long long block_sum = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0ull;
      block_sum = warp_reduce_sum_ull(block_sum);
      if (lane == 0)
        out[blockIdx.x] = block_sum;
    }
  }

  template <typename T, int N>
  struct PlanePointers
  {
    const T *ptrs[N];
  };

  using U8Planes = PlanePointers<uint8_t, 8>;

  template <int K>
  __global__ void scan_planes_u8_byte_unrolled(const U8Planes planes,
                                               uint64_t n,
                                               unsigned long long *__restrict__ per_block_out)
  {
    static_assert(K >= 1 && K <= 8, "K must be in [1, 8]");
    unsigned long long sum = 0;
    uint64_t tid = static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x) +
                   static_cast<uint64_t>(threadIdx.x);
    uint64_t stride = static_cast<uint64_t>(gridDim.x) * static_cast<uint64_t>(blockDim.x);
    for (uint64_t i = tid; i < n; i += stride)
    {
#pragma unroll
      for (int p = 0; p < K; p++)
        sum += static_cast<unsigned long long>(planes.ptrs[p][i]);
    }
    block_reduce_store(sum, per_block_out);
  }

  const void *scan_planes_u8_byte_unrolled_kernel_ptr(int k)
  {
    switch (k)
    {
    case 1:
      return reinterpret_cast<const void *>(scan_planes_u8_byte_unrolled<1>);
    case 2:
      return reinterpret_cast<const void *>(scan_planes_u8_byte_unrolled<2>);
    case 3:
      return reinterpret_cast<const void *>(scan_planes_u8_byte_unrolled<3>);
    case 4:
      return reinterpret_cast<const void *>(scan_planes_u8_byte_unrolled<4>);
    case 5:
      return reinterpret_cast<const void *>(scan_planes_u8_byte_unrolled<5>);
    case 6:
      return reinterpret_cast<const void *>(scan_planes_u8_byte_unrolled<6>);
    case 7:
      return reinterpret_cast<const void *>(scan_planes_u8_byte_unrolled<7>);
    case 8:
      return reinterpret_cast<const void *>(scan_planes_u8_byte_unrolled<8>);
    default:
      die("invalid k for byte strategy kernel pointer");
    }
  }

  void launch_scan_planes_u8_byte_unrolled(int k,
                                           int grid,
                                           int block_threads,
                                           const U8Planes &planes,
                                           uint64_t n,
                                           unsigned long long *per_block_out)
  {
    switch (k)
    {
    case 1:
      scan_planes_u8_byte_unrolled<1><<<grid, block_threads>>>(planes, n, per_block_out);
      return;
    case 2:
      scan_planes_u8_byte_unrolled<2><<<grid, block_threads>>>(planes, n, per_block_out);
      return;
    case 3:
      scan_planes_u8_byte_unrolled<3><<<grid, block_threads>>>(planes, n, per_block_out);
      return;
    case 4:
      scan_planes_u8_byte_unrolled<4><<<grid, block_threads>>>(planes, n, per_block_out);
      return;
    case 5:
      scan_planes_u8_byte_unrolled<5><<<grid, block_threads>>>(planes, n, per_block_out);
      return;
    case 6:
      scan_planes_u8_byte_unrolled<6><<<grid, block_threads>>>(planes, n, per_block_out);
      return;
    case 7:
      scan_planes_u8_byte_unrolled<7><<<grid, block_threads>>>(planes, n, per_block_out);
      return;
    case 8:
      scan_planes_u8_byte_unrolled<8><<<grid, block_threads>>>(planes, n, per_block_out);
      return;
    default:
      die("invalid k for byte strategy launch");
    }
  }

  template <int K>
  __global__ void scan_planes_u8_byte_ilp4(const U8Planes planes,
                                           uint64_t n,
                                           unsigned long long *__restrict__ per_block_out)
  {
    static_assert(K >= 1 && K <= 8, "K must be in [1, 8]");
    unsigned long long sum0 = 0;
    unsigned long long sum1 = 0;
    unsigned long long sum2 = 0;
    unsigned long long sum3 = 0;

    uint64_t tid = static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x) +
                   static_cast<uint64_t>(threadIdx.x);
    uint64_t stride = static_cast<uint64_t>(gridDim.x) * static_cast<uint64_t>(blockDim.x);
    uint64_t step = stride * 4ull;
    uint64_t i = tid;

    for (; i < n;)
    {
      uint64_t i1 = i + stride;
      uint64_t i2 = i1 + stride;
      uint64_t i3 = i2 + stride;
      if (i1 < i || i2 < i1 || i3 < i2 || i3 >= n)
        break;

#pragma unroll
      for (int p = 0; p < K; p++)
      {
        sum0 += static_cast<unsigned long long>(planes.ptrs[p][i]);
        sum1 += static_cast<unsigned long long>(planes.ptrs[p][i1]);
        sum2 += static_cast<unsigned long long>(planes.ptrs[p][i2]);
        sum3 += static_cast<unsigned long long>(planes.ptrs[p][i3]);
      }

      uint64_t next = i + step;
      if (next <= i)
        break;
      i = next;
    }

    for (; i < n;)
    {
#pragma unroll
      for (int p = 0; p < K; p++)
        sum0 += static_cast<unsigned long long>(planes.ptrs[p][i]);

      uint64_t next = i + stride;
      if (next <= i)
        break;
      i = next;
    }

    unsigned long long sum = sum0 + sum1 + sum2 + sum3;
    block_reduce_store(sum, per_block_out);
  }

  const void *scan_planes_u8_byte_ilp4_kernel_ptr(int k)
  {
    switch (k)
    {
    case 1:
      return reinterpret_cast<const void *>(scan_planes_u8_byte_ilp4<1>);
    case 2:
      return reinterpret_cast<const void *>(scan_planes_u8_byte_ilp4<2>);
    case 3:
      return reinterpret_cast<const void *>(scan_planes_u8_byte_ilp4<3>);
    case 4:
      return reinterpret_cast<const void *>(scan_planes_u8_byte_ilp4<4>);
    case 5:
      return reinterpret_cast<const void *>(scan_planes_u8_byte_ilp4<5>);
    case 6:
      return reinterpret_cast<const void *>(scan_planes_u8_byte_ilp4<6>);
    case 7:
      return reinterpret_cast<const void *>(scan_planes_u8_byte_ilp4<7>);
    case 8:
      return reinterpret_cast<const void *>(scan_planes_u8_byte_ilp4<8>);
    default:
      die("invalid k for byte ilp4 strategy kernel pointer");
    }
  }

  void launch_scan_planes_u8_byte_ilp4(int k,
                                       int grid,
                                       int block_threads,
                                       const U8Planes &planes,
                                       uint64_t n,
                                       unsigned long long *per_block_out)
  {
    switch (k)
    {
    case 1:
      scan_planes_u8_byte_ilp4<1><<<grid, block_threads>>>(planes, n, per_block_out);
      return;
    case 2:
      scan_planes_u8_byte_ilp4<2><<<grid, block_threads>>>(planes, n, per_block_out);
      return;
    case 3:
      scan_planes_u8_byte_ilp4<3><<<grid, block_threads>>>(planes, n, per_block_out);
      return;
    case 4:
      scan_planes_u8_byte_ilp4<4><<<grid, block_threads>>>(planes, n, per_block_out);
      return;
    case 5:
      scan_planes_u8_byte_ilp4<5><<<grid, block_threads>>>(planes, n, per_block_out);
      return;
    case 6:
      scan_planes_u8_byte_ilp4<6><<<grid, block_threads>>>(planes, n, per_block_out);
      return;
    case 7:
      scan_planes_u8_byte_ilp4<7><<<grid, block_threads>>>(planes, n, per_block_out);
      return;
    case 8:
      scan_planes_u8_byte_ilp4<8><<<grid, block_threads>>>(planes, n, per_block_out);
      return;
    default:
      die("invalid k for byte ilp4 strategy launch");
    }
  }

  const void *scan_planes_u8_byte_kernel_ptr(ByteVariant variant, int k)
  {
    switch (variant)
    {
    case ByteVariant::Baseline:
      return scan_planes_u8_byte_unrolled_kernel_ptr(k);
    case ByteVariant::Ilp4:
      return scan_planes_u8_byte_ilp4_kernel_ptr(k);
    }
    die("invalid byte variant for kernel pointer");
  }

  void launch_scan_planes_u8_byte(ByteVariant variant,
                                  int k,
                                  int grid,
                                  int block_threads,
                                  const U8Planes &planes,
                                  uint64_t n,
                                  unsigned long long *per_block_out)
  {
    switch (variant)
    {
    case ByteVariant::Baseline:
      launch_scan_planes_u8_byte_unrolled(k, grid, block_threads, planes, n, per_block_out);
      return;
    case ByteVariant::Ilp4:
      launch_scan_planes_u8_byte_ilp4(k, grid, block_threads, planes, n, per_block_out);
      return;
    }
    die("invalid byte variant for launch");
  }

  __global__ void scan_planes_u8_packed32(const uint32_t *const *__restrict__ planes32,
                                          int k,
                                          uint64_t n,
                                          unsigned long long *__restrict__ per_block_out)
  {
    unsigned long long sum = 0;

    uint64_t n4 = n >> 2; // floor(n/4)
    uint64_t tid = static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x) +
                   static_cast<uint64_t>(threadIdx.x);
    uint64_t stride = static_cast<uint64_t>(gridDim.x) * static_cast<uint64_t>(blockDim.x);

    for (uint64_t i4 = tid; i4 < n4; i4 += stride)
    {
      for (int p = 0; p < k; p++)
      {
        uint32_t v = planes32[p][i4];
        sum += static_cast<unsigned long long>(v & 0xffu);
        sum += static_cast<unsigned long long>((v >> 8) & 0xffu);
        sum += static_cast<unsigned long long>((v >> 16) & 0xffu);
        sum += static_cast<unsigned long long>((v >> 24) & 0xffu);
      }
    }

    // tail
    for (uint64_t i = (n4 << 2) + tid; i < n; i += stride)
    {
      const uint8_t *const *planes = reinterpret_cast<const uint8_t *const *>(planes32);
      for (int p = 0; p < k; p++)
        sum += static_cast<unsigned long long>(planes[p][i]);
    }

    block_reduce_store(sum, per_block_out);
  }

  __global__ void scan_planes_u8_shared128(const uint8_t *const *__restrict__ planes,
                                           int k,
                                           uint64_t n,
                                           unsigned long long *__restrict__ per_block_out)
  {
    // Each warp stages 128 bytes (4 bytes per lane) into shared, then each lane consumes 1 byte.
    // This is intentionally an overfetch strategy to force coalesced global accesses.
    unsigned long long sum = 0;

    int lane = threadIdx.x & 31;
    int warp_in_block = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;

    extern __shared__ uint32_t smem32[];
    uint32_t *warp_smem32 = smem32 + static_cast<size_t>(warp_in_block) * 32ull;

    uint64_t warp_global = static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(warps_per_block) +
                           static_cast<uint64_t>(warp_in_block);
    uint64_t warp_stride = static_cast<uint64_t>(gridDim.x) * static_cast<uint64_t>(warps_per_block);

    uint64_t base = warp_global * 128ull; // 128 elements per warp-iteration
    while (base < n)
    {
      uint64_t offset = base + static_cast<uint64_t>(lane) * 4ull;

      for (int p = 0; p < k; p++)
      {
        uint32_t pack = 0;
        if (offset + 3 < n)
        {
          pack = *reinterpret_cast<const uint32_t *>(planes[p] + offset);
        }
        else
        {
          if (offset + 0 < n)
            pack |= static_cast<uint32_t>(planes[p][offset + 0]) << 0;
          if (offset + 1 < n)
            pack |= static_cast<uint32_t>(planes[p][offset + 1]) << 8;
          if (offset + 2 < n)
            pack |= static_cast<uint32_t>(planes[p][offset + 2]) << 16;
          if (offset + 3 < n)
            pack |= static_cast<uint32_t>(planes[p][offset + 3]) << 24;
        }

        warp_smem32[lane] = pack;
        __syncwarp();

        uint64_t i = base + static_cast<uint64_t>(lane);
        if (i < n)
        {
          uint32_t ppack = warp_smem32[lane >> 2];
          uint32_t b = (ppack >> (8u * static_cast<uint32_t>(lane & 3))) & 0xffu;
          sum += static_cast<unsigned long long>(b);
        }
        __syncwarp();
      }

      base += warp_stride * 128ull;
    }

    block_reduce_store(sum, per_block_out);
  }

  __global__ void scan_planes_u16(const uint16_t *const *__restrict__ planes16,
                                  int k,
                                  uint64_t n,
                                  unsigned long long *__restrict__ per_block_out)
  {
    unsigned long long sum = 0;
    uint64_t tid = static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x) +
                   static_cast<uint64_t>(threadIdx.x);
    uint64_t stride = static_cast<uint64_t>(gridDim.x) * static_cast<uint64_t>(blockDim.x);
    for (uint64_t i = tid; i < n; i += stride)
    {
      for (int p = 0; p < k; p++)
        sum += static_cast<unsigned long long>(planes16[p][i]);
    }
    block_reduce_store(sum, per_block_out);
  }

  int occupancy_grid(int device, int block_threads, int grid_mul, const void *kernel, size_t shmem)
  {
    cudaDeviceProp prop{};
    cuda_check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");

    int max_active = 0;
    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active, kernel, block_threads, shmem),
               "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
    long long grid = static_cast<long long>(prop.multiProcessorCount) * max_active * grid_mul;
    if (grid < 1)
      grid = 1;
    if (grid > std::numeric_limits<int>::max())
      grid = std::numeric_limits<int>::max();
    return static_cast<int>(grid);
  }

  struct RunResult
  {
    int k = 0;
    uint64_t n = 0;
    int plane_bytes = 0;
    std::string strategy;
    int block = 0;
    int grid = 0;
    int warmup = 0;
    int iters = 0;
    float ms_per_iter = 0.0f;
    double logical_GBps = 0.0;
    double overfetch_factor = 1.0;
  };

  RunResult run_one(const Options &opt,
                    int k,
                    const U8Planes &u8_planes,
                    void *d_plane_ptrs,
                    unsigned long long *d_out,
                    int grid)
  {
    cuda_check(cudaSetDevice(opt.device), "cudaSetDevice");

    // Warmup.
    for (int i = 0; i < opt.warmup; i++)
    {
      if (opt.plane_bytes == 2)
      {
        scan_planes_u16<<<grid, opt.block_threads>>>(
            reinterpret_cast<const uint16_t *const *>(d_plane_ptrs), k, opt.n, d_out);
      }
      else if (opt.strategy == "byte")
      {
        launch_scan_planes_u8_byte(opt.byte_variant,
                                   k,
                                   grid,
                                   opt.block_threads,
                                   u8_planes,
                                   opt.n,
                                   d_out);
      }
      else if (opt.strategy == "packed32")
      {
        scan_planes_u8_packed32<<<grid, opt.block_threads>>>(
            reinterpret_cast<const uint32_t *const *>(d_plane_ptrs), k, opt.n, d_out);
      }
      else if (opt.strategy == "shared128")
      {
        size_t shmem = static_cast<size_t>(opt.block_threads / 32) * 128ull;
        scan_planes_u8_shared128<<<grid, opt.block_threads, shmem>>>(
            reinterpret_cast<const uint8_t *const *>(d_plane_ptrs), k, opt.n, d_out);
      }
    }
    cuda_check(cudaGetLastError(), "warmup launch");
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(warmup)");

    cudaEvent_t start{}, stop{};
    cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    cuda_check(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int i = 0; i < opt.iters; i++)
    {
      if (opt.plane_bytes == 2)
      {
        scan_planes_u16<<<grid, opt.block_threads>>>(
            reinterpret_cast<const uint16_t *const *>(d_plane_ptrs), k, opt.n, d_out);
      }
      else if (opt.strategy == "byte")
      {
        launch_scan_planes_u8_byte(opt.byte_variant,
                                   k,
                                   grid,
                                   opt.block_threads,
                                   u8_planes,
                                   opt.n,
                                   d_out);
      }
      else if (opt.strategy == "packed32")
      {
        scan_planes_u8_packed32<<<grid, opt.block_threads>>>(
            reinterpret_cast<const uint32_t *const *>(d_plane_ptrs), k, opt.n, d_out);
      }
      else if (opt.strategy == "shared128")
      {
        size_t shmem = static_cast<size_t>(opt.block_threads / 32) * 128ull;
        scan_planes_u8_shared128<<<grid, opt.block_threads, shmem>>>(
            reinterpret_cast<const uint8_t *const *>(d_plane_ptrs), k, opt.n, d_out);
      }
      else
      {
        die("unknown strategy");
      }
    }
    cuda_check(cudaGetLastError(), "timed launch");
    cuda_check(cudaEventRecord(stop), "cudaEventRecord(stop)");
    cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

    float ms_total = 0.0f;
    cuda_check(cudaEventElapsedTime(&ms_total, start, stop), "cudaEventElapsedTime");
    cuda_check(cudaEventDestroy(start), "cudaEventDestroy(start)");
    cuda_check(cudaEventDestroy(stop), "cudaEventDestroy(stop)");

    float ms_per_iter = ms_total / static_cast<float>(opt.iters);
    double seconds = static_cast<double>(ms_per_iter) / 1000.0;

    double logical_bytes = static_cast<double>(opt.n) * static_cast<double>(k) *
                           static_cast<double>(opt.plane_bytes);
    double logical_GBps = (logical_bytes / seconds) / 1e9;

    double overfetch = 1.0;
    if (opt.plane_bytes == 1 && opt.strategy == "shared128")
      overfetch = 4.0; // 4 bytes staged/lane

    RunResult r{};
    r.k = k;
    r.n = opt.n;
    r.plane_bytes = opt.plane_bytes;
    r.strategy = opt.strategy;
    r.block = opt.block_threads;
    r.grid = grid;
    r.warmup = opt.warmup;
    r.iters = opt.iters;
    r.ms_per_iter = ms_per_iter;
    r.logical_GBps = logical_GBps;
    r.overfetch_factor = overfetch;
    return r;
  }

  std::vector<int> make_k_sweep(const Options &opt)
  {
    int planes_total = (opt.plane_bytes == 1) ? 8 : 4;
    int k_min = opt.single_k ? opt.k_single : opt.k_min;
    int k_max = opt.single_k ? opt.k_single : opt.k_max;
    if (k_min < 1)
      k_min = 1;
    if (k_max > planes_total)
      k_max = planes_total;
    if (k_min > k_max)
      die("invalid k sweep after clamping");

    std::vector<int> ks;
    for (int k = k_min; k <= k_max; k++)
      ks.push_back(k);
    return ks;
  }

  std::string csv_escape(std::string_view s)
  {
    bool needs_quotes = false;
    for (char c : s)
    {
      if (c == ',' || c == '"' || c == '\n' || c == '\r')
      {
        needs_quotes = true;
        break;
      }
    }
    if (!needs_quotes)
      return std::string(s);
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    for (char c : s)
    {
      if (c == '"')
        out.push_back('"');
      out.push_back(c);
    }
    out.push_back('"');
    return out;
  }

} // namespace

int main(int argc, char **argv)
{
  Options opt = parse_args(argc, argv);

  cuda_check(cudaSetDevice(opt.device), "cudaSetDevice");
  cudaDeviceProp prop{};
  cuda_check(cudaGetDeviceProperties(&prop, opt.device), "cudaGetDeviceProperties");

  int planes_total = (opt.plane_bytes == 1) ? 8 : 4;
  if (opt.plane_bytes == 1)
  {
    if (opt.strategy != "byte" && opt.strategy != "packed32" && opt.strategy != "shared128")
    {
      die("invalid --strategy for --plane_bytes 1");
    }
  }
  else
  {
    if (opt.strategy != "byte")
    {
      die("for --plane_bytes 2, only --strategy byte is supported (u16 loads)");
    }
  }

  // Allocate planes: SOA, each plane is contiguous.
  std::vector<void *> d_planes;
  d_planes.reserve(static_cast<size_t>(planes_total));

  if (opt.plane_bytes == 1)
  {
    for (int p = 0; p < planes_total; p++)
    {
      void *ptr = nullptr;
      cuda_check(cudaMalloc(&ptr, static_cast<size_t>(opt.n) * sizeof(uint8_t)), "cudaMalloc(plane)");
      cuda_check(cudaMemset(ptr, 0xAB, static_cast<size_t>(opt.n) * sizeof(uint8_t)),
                 "cudaMemset(plane)");
      d_planes.push_back(ptr);
    }
  }
  else
  {
    for (int p = 0; p < planes_total; p++)
    {
      void *ptr = nullptr;
      cuda_check(cudaMalloc(&ptr, static_cast<size_t>(opt.n) * sizeof(uint16_t)), "cudaMalloc(plane16)");
      cuda_check(cudaMemset(ptr, 0xAB, static_cast<size_t>(opt.n) * sizeof(uint16_t)),
                 "cudaMemset(plane16)");
      d_planes.push_back(ptr);
    }
  }

  // Device pointer array for kernels.
  void *d_plane_ptrs = nullptr;
  U8Planes h_u8_planes{};

  if (opt.plane_bytes == 1 && opt.strategy == "packed32")
  {
    // Each plane pointer is interpreted as uint32_t* for packed32 loads.
    std::vector<uint32_t *> h_ptrs;
    h_ptrs.reserve(d_planes.size());
    for (void *p : d_planes)
      h_ptrs.push_back(reinterpret_cast<uint32_t *>(p));
    cuda_check(cudaMalloc(&d_plane_ptrs, d_planes.size() * sizeof(uint32_t *)), "cudaMalloc(ptrs32)");
    cuda_check(cudaMemcpy(d_plane_ptrs,
                          h_ptrs.data(),
                          d_planes.size() * sizeof(uint32_t *),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(ptrs32)");
  }
  else if (opt.plane_bytes == 1 && opt.strategy == "shared128")
  {
    std::vector<uint8_t *> h_ptrs;
    h_ptrs.reserve(d_planes.size());
    for (void *p : d_planes)
      h_ptrs.push_back(reinterpret_cast<uint8_t *>(p));
    cuda_check(cudaMalloc(&d_plane_ptrs, d_planes.size() * sizeof(uint8_t *)), "cudaMalloc(ptrs8)");
    cuda_check(cudaMemcpy(d_plane_ptrs,
                          h_ptrs.data(),
                          d_planes.size() * sizeof(uint8_t *),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(ptrs8)");
  }
  else if (opt.plane_bytes == 1 && opt.strategy == "byte")
  {
    for (int p = 0; p < 8; p++)
      h_u8_planes.ptrs[p] = static_cast<const uint8_t *>(d_planes[static_cast<size_t>(p)]);
  }
  else
  {
    std::vector<uint16_t *> h_ptrs;
    h_ptrs.reserve(d_planes.size());
    for (void *p : d_planes)
      h_ptrs.push_back(reinterpret_cast<uint16_t *>(p));
    cuda_check(cudaMalloc(&d_plane_ptrs, d_planes.size() * sizeof(uint16_t *)), "cudaMalloc(ptrs16)");
    cuda_check(cudaMemcpy(d_plane_ptrs,
                          h_ptrs.data(),
                          d_planes.size() * sizeof(uint16_t *),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(ptrs16)");
  }

  // Allocate per-block output to keep loads alive.
  int grid_single = 1;
  std::vector<int> grid_by_k(9, 1); // use indices [1..8]
  int max_needed_grid = 1;

  if (opt.plane_bytes == 1 && opt.strategy == "byte")
  {
    for (int kk = 1; kk <= 8; kk++)
    {
      int g = occupancy_grid(opt.device, opt.block_threads, opt.grid_mul,
                             scan_planes_u8_byte_kernel_ptr(opt.byte_variant, kk), 0);
      grid_by_k[static_cast<size_t>(kk)] = g;
      if (g > max_needed_grid)
        max_needed_grid = g;
    }
  }
  else if (opt.plane_bytes == 2)
  {
    grid_single = occupancy_grid(opt.device, opt.block_threads, opt.grid_mul,
                                 reinterpret_cast<const void *>(scan_planes_u16), 0);
    max_needed_grid = grid_single;
  }
  else if (opt.strategy == "packed32")
  {
    grid_single = occupancy_grid(opt.device, opt.block_threads, opt.grid_mul,
                                 reinterpret_cast<const void *>(scan_planes_u8_packed32), 0);
    max_needed_grid = grid_single;
  }
  else if (opt.strategy == "shared128")
  {
    size_t shmem = static_cast<size_t>(opt.block_threads / 32) * 128ull;
    grid_single = occupancy_grid(opt.device, opt.block_threads, opt.grid_mul,
                                 reinterpret_cast<const void *>(scan_planes_u8_shared128), shmem);
    max_needed_grid = grid_single;
  }
  else
  {
    die("unknown strategy");
  }

  unsigned long long *d_out = nullptr;
  cuda_check(cudaMalloc(&d_out, static_cast<size_t>(max_needed_grid) * sizeof(unsigned long long)),
             "cudaMalloc(d_out)");
  cuda_check(cudaMemset(d_out, 0, static_cast<size_t>(max_needed_grid) * sizeof(unsigned long long)),
             "cudaMemset(d_out)");

  std::FILE *f = std::fopen(opt.csv_path.c_str(), "wb");
  if (!f)
    die("failed to open --csv output path");
  std::fprintf(f,
               "strategy,plane_bytes,k,n,logical_bytes,overfetch_factor,block,grid,warmup,iters,ms_per_iter,logical_GBps,device,sm,cc_major,cc_minor\n");

  for (int k : make_k_sweep(opt))
  {
    int grid = (opt.plane_bytes == 1 && opt.strategy == "byte")
                   ? grid_by_k[static_cast<size_t>(k)]
                   : grid_single;
    RunResult r = run_one(opt, k, h_u8_planes, d_plane_ptrs, d_out, grid);
    double logical_bytes = static_cast<double>(r.n) * static_cast<double>(r.k) *
                           static_cast<double>(r.plane_bytes);
    std::fprintf(f,
                 "%s,%d,%d,%" PRIu64 ",%.0f,%.3f,%d,%d,%d,%d,%.6f,%.3f,%s,%d,%d,%d\n",
                 r.strategy.c_str(),
                 r.plane_bytes,
                 r.k,
                 static_cast<uint64_t>(r.n),
                 logical_bytes,
                 r.overfetch_factor,
                 r.block,
                 r.grid,
                 r.warmup,
                 r.iters,
                 r.ms_per_iter,
                 r.logical_GBps,
                 csv_escape(prop.name).c_str(),
                 prop.multiProcessorCount,
                 prop.major,
                 prop.minor);
    std::fflush(f);

    std::fprintf(stderr,
                 "[exp1] strat=%s planeB=%d k=%d n=%" PRIu64 " ms=%.3f logical_GB/s=%.1f\n",
                 r.strategy.c_str(),
                 r.plane_bytes,
                 r.k,
                 static_cast<uint64_t>(r.n),
                 r.ms_per_iter,
                 r.logical_GBps);
  }

  std::fclose(f);

  cuda_check(cudaFree(d_out), "cudaFree(d_out)");
  if (d_plane_ptrs != nullptr)
    cuda_check(cudaFree(d_plane_ptrs), "cudaFree(d_plane_ptrs)");
  for (void *p : d_planes)
    cuda_check(cudaFree(p), "cudaFree(plane)");

  return 0;
}
