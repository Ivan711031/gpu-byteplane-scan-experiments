## 1\. 實驗目的

本輪實驗目標是比較 `strategy=byte` 下兩個 kernel variant：

`baseline`：原本單一 accumulator 的 byte-plane scan。

`ilp4`：新增 4-way outer unroll，並使用 `sum0..sum3` 四條 thread-local accumulator chain。

核心問題是：

> 原本 byte kernel 的瓶頸是否部分來自 load-to-use dependency chain 太長與 ILP 不足？

---

## 2\. Benchmark 結果

固定參數：

GPU: NVIDIA H200  
n = 100000000  
plane\_bytes = 1  
strategy = byte  
block = 256  
grid\_mul = 1  
warmup = 10  
iters = 1000  
k = 1..8

### Runtime 對照

| k | baseline ms | ilp4 ms | speedup |
| --- | --- | --- | --- |
| 1 | 0.139864 | 0.063163 | 2.21x |
| 2 | 0.162170 | 0.101275 | 1.60x |
| 3 | 0.176967 | 0.128839 | 1.37x |
| 4 | 0.195910 | 0.169460 | 1.16x |
| 5 | 0.216605 | 0.213901 | 1.01x |
| 6 | 0.251330 | 0.249330 | 1.01x |
| 7 | 0.308131 | 0.296925 | 1.04x |
| 8 | 0.379756 | 0.333311 | 1.14x |

### Throughput 對照

| k | baseline logical GB/s | ilp4 logical GB/s |
| --- | --- | --- |
| 1 | 714.982 | 1583.208 |
| 2 | 1233.275 | 1974.827 |
| 3 | 1695.227 | 2328.496 |
| 4 | 2041.750 | 2360.433 |
| 5 | 2308.348 | 2337.526 |
| 6 | 2387.296 | 2406.449 |
| 7 | 2271.763 | 2357.494 |

   → ilp4 無法完全解決，k=5~8 仍明顯存在