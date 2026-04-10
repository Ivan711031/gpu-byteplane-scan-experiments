cmake_minimum_required(VERSION 3.24)

project(gpu_byteplane_scan_experiment1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(bench_byteplane_scan bench_byteplane_scan.cu)

target_compile_options(bench_byteplane_scan PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3 -lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

set_target_properties(bench_byteplane_scan PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
)
