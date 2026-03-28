# C5 GPU 上的 progressive Byte-Plane 欄位掃描

## 背景

分析型查詢 (analytical query) 的核心操作是 columnar scan: 對一整欄的數值做 SUM、AVG、COUNT、MIN、MAX 等 aggregation 運算。這類運算的計算量極低 (每個值只做一次加法或比較)，瓶頸完全在記憶體頻寬: 能多快把資料從記憶體讀進計算單元，就決定了查詢能跑多快。

在 GPU 上，這個頻寬瓶頸更加明顯。以 NVIDIA H100 為例，HBM3 提供約 3.35 TB/s 的讀取頻寬，但計算單元能處理超過 1000 TFLOPS 的 FP64 運算。對一個 columnar scan 來說，每讀 8 bytes (一個 FP64 值) 只做 1 次浮點運算，計算能力被浪費了三個數量級。每一個從 HBM 讀進來但對查詢結果沒有貢獻的 byte，都是浪費。

現有的 GPU 分析引擎 (HeavyDB、RAPIDS cuDF、Kinetica) 不管查詢需要多少精度，都會讀取每個 FP64 值的全部 8 bytes。一個只需要算平均溫度到 0.01 度的 dashboard 查詢，和一個需要 15 位有效數字的科學計算，讀的資料量完全一樣。

但 IEEE 754 FP64 的 8 個 byte 攜帶的資訊量差異極大:

| Byte (從 MSB 算起) | 內容 | 大約精度貢獻 |
| :---- | :---- | :---- |
| Byte 7 (MSB) | sign bit \+ 上半 exponent | 決定數量級 |
| Byte 6 | 下半 exponent \+ 上半 mantissa | \~10^-2 相對精度 |
| Byte 5 | mantissa bits | \~10^-5 |
| Byte 4 | mantissa bits | \~10^-7 |
| Byte 3 | mantissa bits | \~10^-10 |
| Bytes 0-2 (LSB) | 最低位 mantissa bits | \~10^-13 到 10^-16 |

一個容忍 0.01 誤差的查詢不需要 byte 0-4。一個只需要排序或篩選 (top-k、argmax、WHERE x \> threshold) 的查詢，可能只需要最高位的幾個 byte 就能確定大多數 row 的結果。

HBM 的另一個特性是容量小但昂貴。H100 的 80 GB HBM 每 GB 成本是 DDR5 的 30-50 倍，容量只有 server DDR5 的 1/6 到 1/25。如果能只存和讀每個值需要的 byte，等於同時提升了有效容量和查詢吞吐量，兩者相乘。這個效益在 CPU 上不明顯 (DDR5 便宜、cache 層級複雜)，但在 GPU 上因為 HBM 的極端頻寬 / 容量比，效益會很直接。

## 觀察

**觀察一: 浮點數可以按 byte 拆成 subcolumn。** 把一欄 FP64 值拆成 8 個 byte-width 的 subcolumn，從 MSB 到 LSB 排列。查詢時從最高位的 subcolumn 開始逐層讀取，每多讀一層就收緊誤差範圍，達到需要的精度就停。如果一個查詢只需要讀 k 個 subcolumn (k \< 8)，資料量是原本的 k/8，在頻寬瓶頸的 GPU 上，吞吐量理論上是 8/k 倍。

**觀察二: Buff-style 編碼讓 aggregation 運算變成整數運算。** 直接對 raw byte 做截斷的問題是誤差分析複雜 (exponent 和 mantissa 交織在同一個 byte 裡)。Buff (Bounded Fast Floats) 的做法是先對每個 segment 做 Frame-of-Reference (FOR) 編碼: 減去 segment 最小值得到一個有界整數，再把整數和小數部分分別存成 byte-width subcolumn。這樣 SUM/AVG 變成整數累加 (用 GPU 上閒置的 INT32 ALU)，最後每個 segment 只需一次浮點修正。誤差界也變得乾淨: 讀了 integer subcolumn 加 k 個 fractional subcolumn 後，每個值的最大絕對誤差是 scale \* 2^(-8k)。

**觀察三: 對 predicate (WHERE x \> threshold) 也能 progressive 處理。** 從 MSB subcolumn 開始，每讀一層就把 row 分成三類: 確定通過、確定不通過、尚未確定。只有「尚未確定」的 row 需要讀下一層。如果大部分 row 在前兩層就能決定，後面的 subcolumn 根本不用讀。

**觀察四: GPU 的 warp 執行模型對 progressive filter 有獨特挑戰。** 一個 warp 是 32 個 thread 同步執行。如果 progressive filter 讓越來越多 thread 變成 inactive (row 已確定)，warp utilization 會下降。需要比較至少兩種策略: passive masking (inactive thread 閒置) 和 stream compaction (把 active row 壓縮成連續排列)，找出在不同 selectivity 下的 crossover point。

**觀察五: byte-granularity 的讀取在 GPU 上能否有效 coalesce 是未知的。** GPU 全域記憶體存取最有效率的模式是 32 個 thread 讀 32 個連續的 4-byte word (128-byte coalesced access)。在 byte-plane layout 下，32 個 thread 讀 32 個連續 byte，只有 32 bytes。H100 硬體是否能有效處理這種 sub-4-byte 讀取，文件沒有說明，必須實測。這是整個專題最優先要驗證的前提。

## 研究問題與目標

這個專題要回答一個核心問題:

給定一個分析型查詢、誤差預算 epsilon、和資料分布，需要讀幾層 byte-plane subcolumn 才能讓結果落在 exact answer 的 epsilon 範圍內? 在 GPU HBM 上對應的吞吐量是多少?

答案是一條 precision-throughput curve: 每個 subcolumn 數量 k 對應一個吞吐量和一個查詢誤差。這條曲線是這個專題最重要的 deliverable，它告訴系統設計者「在這個誤差容忍度下，你能得到多少加速」。

具體的研究子問題:

* byte-plane 掃描在 GPU 上是否真的能按比例節省頻寬? (coalescing 問題)

* 對每種 aggregation 運算 (SUM, AVG, COUNT, MIN, MAX, VAR)，截斷誤差的解析界和實測值分別是多少? 理論界有多鬆?

* progressive filter 的 warp utilization 問題，在不同 selectivity 下哪種策略最有效?

* FOR 參數 (segment size, integer part bit-width) 如何影響收斂速度?

* 多 GPU 擴展時，throughput 是否線性 scale?

以下問題我們不探討：

* 一個 GPU 資料庫。沒有 query parsing、optimization、join、transaction。Deliverable是 scan/aggregation kernel library。

* 基於 sampling 的 approximate query processing。AQP (如 BlinkDB) 抽樣 row; 我們讀所有 row 但減少每個 row 的 bit 精度。誤差模型不同: sampling error 是統計性的，truncation error 是確定性的、有界的。

* 宣稱「GPU 比 CPU 快」。那是已知的。核心結果是: 在精度 k 下，progressive GPU scan 達到 full-precision throughput 的多少比例，誤差界是多少。

## 實驗設計

所有實驗都在 GPU HBM 上已駐留的資料上操作。從 FP64 編碼成 byte-plane / Buff-style 格式的步驟是離線的 (載入時做一次)，不是研究貢獻，不量測。

### Experiment 0: 硬體 Baseline

**目標:** 建立所有後續實驗的參考基準。

**方法:** 寫一個 CUDA HBM bandwidth microbenchmark: 在 GPU global memory 分配 N bytes，用 kernel 全部讀一遍，量測吞吐量。N 從 1 MB 變化到 8 GB，確認峰值接近 H100 spec (\~3.35 TB/s)。再量測三種存取模式的頻寬: (a) 完全 coalesced sequential read, (b) 帶 bitmask 的 read (部分 lane 閒置), (c) 透過 index array 的 gather。記錄頻寬衰減，這些數字校準所有後續實驗。

**Deliverable:** 硬體參數和實測峰值頻寬的表格 (論文的 experimental setup section)。

### Experiment 1: Byte-Plane 掃描是否真的節省頻寬?

**目標:** 驗證核心前提: 讀更少的 byte-plane 是否等比例加速 GPU scan。如果 GPU memory coalescing 對 byte-granularity read 有嚴重懲罰，整個前提就不成立。這個實驗必須最先做。

**方法:** 對每個 k (1 到 8 planes): \- 把 Dataset A (N \= 10^8 個 FP64 值) 存成 byte-plane layout: 8 個各 N bytes 的連續陣列。 \- 啟動 CUDA kernel 讀取最高的 k 個 plane，計算部分值的 SUM。 \- 量測 wall-clock time (warmup 10 次後取 1000 次平均)。 \- 計算 throughput: (N \* k bytes) / time。

同時量測 contiguous baseline: 同樣的資料存成標準 FP64 陣列 (N \* 8 bytes)，用標準 SUM scan。

**如果 single-byte read 太慢，要測試以下替代方案:** \- Packed reads: 每個 thread 讀一個 uint32 (4 個值的同一 plane 的 byte)，在 register 裡拆開。 \- Shared memory staging: 一個 warp 用 coalesced read 載入 128-byte chunk 到 shared memory，再從 shared memory 取 individual byte。 \- 2-byte planes: 拆成 4 個 2-byte plane 而不是 8 個 1-byte plane。

用 Nsight Compute profile 確認最終選擇的方案達到預期的 memory transaction pattern。

**Deliverable:** scan throughput (GB/s) vs. plane 數量的 bar chart，標示 contiguous baseline。這是論文的 Figure 1，整篇論文的前提立足於此。

### Experiment 2: 截斷誤差分析

**目標:** 量化讀更少 subcolumn 帶來的誤差。對每種 aggregation 和每個 dataset，需要幾個 subcolumn 才能滿足給定的誤差預算? 建立 (aggregation, dataset, epsilon) 到最小 subcolumn 數 k\* 的映射。

**Part A (解析推導):** 對每種運算推導最大絕對誤差作為 k 和 segment 參數的函式:

| 運算 | 注意事項 |
| :---- | :---- |
| SUM | Error \<= N \* max\_per\_value\_error(k)，線性成長於 N |
| AVG | \= SUM / N，截斷誤差可能因正負相消而部分抵消 |
| COUNT(x \> t) | 截斷可能把值推過 threshold，誤差取決於 threshold 附近的資料密度 |
| MIN / MAX | MSB 決定排序，截斷很少改變極值，但 top 值接近時需要所有 subcolumn |
| VAR | 涉及 (x \- mean)^2，截斷誤差可能因平方而放大 |

每個運算交付一個 lemma，精確陳述，附證明。注意: worst-case bound 假設所有誤差同方向累積，實際上誤差可能部分相消。要同時分析 worst-case 和 typical-case 行為。如果 bound 是 distribution-dependent，明確寫出來。

**Part B (實測):** 對每個 dataset、每個 k (1-8)、每個 aggregation f: \- 計算 f\_exact: full-precision FP64 的 aggregation 結果。 \- 計算 f\_trunc(k): 只用 top k subcolumn 重建的資料的 aggregation 結果。 \- 記錄絕對誤差和相對誤差。

把實測誤差跟 Part A 的解析界比較。Gap (界有多鬆) 本身就是一個結果。

**Deliverable:** (aggregation, dataset, epsilon) 到 minimum k\* 的映射表。解析界 vs. 實測誤差的圖。

### Experiment 3: progressive aggregation Kernel

**目標:** 核心效能結果。展示每多讀一個 subcolumn，throughput 線性下降而 error 指數下降。

**方法:** 實作 Buff-style progressive aggregation kernel: \- 每個 thread block 處理一個 segment 的 subcolumn 資料。 \- Round 0: 讀 integer subcolumn，INT32/INT64 累加。 \- Round k: 讀 fractional subcolumn k，refinement 累加。 \- 最後套用 FOR base 和 scale 修正 (每個 thread block 一次 FP 運算，不是每個 row)。 \- 輸出: (approximate\_result, error\_bound) pair。

量測每個 k 在每個 dataset 上的 throughput (billion rows/sec)，結合 Experiment 2 的誤差畫出 precision-throughput curve。

注意 integer accumulation overflow: 每個 row 的 subcolumn 值最大 255，10^9 個 row 的 partial sum 約 2.55 \* 10^11，INT64 可容納。但如果 segment 用更寬的 integer part (16-bit 或 32-bit FOR-encoded)，range 會增大，需要判斷何時需要 multi-word accumulation。

**Deliverable:** precision-throughput curve: throughput 為一軸，error 為另一軸，每個 k 一個點，每種 aggregation 和 dataset 一條曲線。這是論文的主圖。

### Experiment 4: progressive Filter Kernel

**目標:** 量測 WHERE x \> threshold 的 progressive predicate evaluation 效能，比較不同的 warp utilization 策略。

**方法:** \- Round 0: 讀 MSB subcolumn，分類每個 row 為 qualified / disqualified / ambiguous，輸出 bitmask。 \- Round 1: 只對 ambiguous row 讀下一個 subcolumn，更新 bitmask。 \- 重複到 ambiguous set 為空或 k 個 subcolumn 用完。

**至少實作兩種 warp management 策略並比較:**

* **Passive masking:** inactive thread 閒置 (predicated execution)。簡單，但 warp utilization 隨 survival rate 下降。

* **Intra-block stream compaction:** 每輪後用 \_\_ballot\_sync \+ \_\_popc 把 surviving row index 壓縮成 dense array。compaction 開銷換取更好的 utilization。

* **Deferred gather:** 所有 filter round 用 passive masking 產出 final survivor bitmask，然後另起一個 gather kernel 只讀 survivor 的值做 aggregation。

crossover point 取決於 selectivity。sweep selectivity 從 1% 到 99%，量測每種策略的 throughput。

也實作 combined filter \+ aggregate: SELECT AVG(x) FROM T WHERE x \> threshold，量測端到端 throughput。

**Deliverable:** (a) throughput vs. selectivity 圖，比較各策略。(b) combined filter+aggregate 的 precision-throughput curve。(c) 分析何時 progressive filtering 有效 (大部分 row 早期就能決定) vs. 無效 (selectivity 接近 50%)。

### Experiment 5: 多 GPU 擴展

**目標:** 確認 progressive scan 跨 GPU partition 線性擴展。

**方法:** Time-series segment 分配到不同 GPU (round-robin 或 range-partitioned)。每個 GPU 獨立跑 progressive scan，產出 (partial\_result, count, error\_bound)。用 NCCL allreduce 合併。

量測 1 \-\> 2 \-\> 4 \-\> 8 GPU (單節點) \-\> 16 GPU (雙節點) 的 throughput scaling，在多個精度層級 (k \= 3, 5, 8)。如果 scaling 是 sublinear，找出瓶頸 (NCCL latency? PCIe host traffic? load imbalance?)。

不要蓋一個 distributed query engine。多 GPU 層就是 partitioned parallel scan \+ one reduce。

**Deliverable:** scaling curve (GPU 數量 vs. throughput)，每個精度層級一條線。

### Experiment 6: FOR 參數敏感度

**目標:** FOR 的 base 和 scale 決定 integer part 和 fractional subcolumn 之間的 bit 分配。這個分配控制誤差界隨 subcolumn 增加的收斂速度。有沒有最佳參數? 是否 data-dependent?

**方法:** 對每個 dataset，變動 segment size (每個 FOR segment 的 row 數) 和 integer part 的 bit-width。對每個組合: \- 編碼資料。 \- 量測每個 k 的誤差。 \- 記錄達到 epsilon \= 10^-2, 10^-4, 10^-6 所需的最小 k。

**Deliverable:** minimum k vs. segment configuration 的圖，按 dataset 分層。選擇 FOR 參數的實務指引。

### 資料集

**合成資料 (N \= 10^8 開發用，10^9 做最終量測):**

* **Dataset A (“Sensor”):** 模擬溫度讀數 (\~15-35 度)。FOR range 窄，MSB subcolumn entropy 低。這是 favorable case: 容忍 0.01 誤差的查詢可以跳過大部分 subcolumn。

* **Dataset B (“Uniform”):** \[0, 1000\] 均勻分布。每個 mantissa bit 都有意義。這是 adversarial case，測試 progressive scanning 的下限。

* **Dataset C (“Heavy-tailed”):** log-normal 分布 (sigma=5)，值跨多個數量級。壓力測試 FOR encoding (per-segment integer part 要容納寬 range)。

* **Dataset D (“Zipfian”):** 偏斜分布帶偶發的大 outlier。測試 FOR segment 如何處理 outlier，以及 progressive filtering 在極端值主導時的行為。

**真實資料 (至少一個):**

* NYC Taxi trip data (公開，10 億+ rows，有 fare、distance、coordinates 等 float 欄位)

* NOAA GSOD 每日溫度資料

* UCI ML Repository “Individual Household Electric Power Consumption”

**資料集特徵化 (實驗前必做):** 對每個 dataset 在 Buff-style 編碼後報告: integer part width 分布、per-subcolumn entropy、per-subcolumn byte-value histogram。這些特徵決定 progressive scanning 的效果。如果某個 dataset 在 MSB subcolumn 有很高 entropy，progressive filtering 效益有限，要誠實記錄。

### Baseline 和比較對象

| Baseline | 用途 |
| :---- | :---- |
| Full-precision GPU scan (自己實作) | k \= all-subcolumns 的比較點，顯示精度門控帶來的加速 |
| DuckDB (CPU, latest release) | 絕對效能參考。DuckDB 是目前最強的單節點 CPU 分析引擎 |
| HeavyDB (GPU, full precision) | 顯示加速來自 progressive precision，不是來自「我們用了 GPU」 |

核心結果不是「GPU 比 CPU 快」，而是 precision-throughput curve。

### 工作分工建議

**同學1 (Kernel Engineering):** \- 實作 byte-plane 分解/重建 (raw 和 Buff-style) \- 解決 coalescing 問題 (最高優先) \- 實作所有 aggregation kernel 的 progressive byte-plane 版本 \- 實作 progressive filter kernel (至少兩種 warp management 策略) \- 實作 combined filter+aggregate kernel \- 跑 Experiment 0, 1, 3, 4 \- 用 Nsight Compute profile 所有 kernel \- 實作 NCCL multi-GPU reduction

**同學2 (Formalization, Encoding, Evaluation):** \- 推導所有 aggregation 運算的 formal error bound (Experiment 2 Part A) \- 建 Buff-style encoding library (先 CPU C++ 確認正確性，需要時 port 到 CUDA) \- 分析 FOR parameter sensitivity (Experiment 6\) \- 分析 integer accumulation overflow 條件 \- 準備所有 dataset: 產生合成資料、下載真實資料、計算 per-subcolumn characterization \- 設定 DuckDB 和 HeavyDB baseline \- 跑 Experiment 2 Part B, 5, 6 \- 產出所有圖表 \- 主導論文寫作

同學1的誤差量測依賴同學2的 kernel 實作。 Milestone: 同學1有 working byte-plane SUM kernel 並有 throughput 數字 (Experiment 1 完成)。同學2驗證 encoding library 的 round-trip correctness 且完成 SUM 和 AVG 的 error bound proof。

## 建議閱讀文獻

* **Decomposed Bounded Floats for Fast Compression and Queries** (Liu et al., VLDB 2021). 這是 Buff-style encoding 的原始論文，說明如何把浮點數拆成 byte-width subcolumn 並直接在壓縮格式上做查詢。重點看 Section 3 的編碼方法和 Section 5 的 query operator 設計。[PDF](http://www.vldb.org/pvldb/vol14/p2586-liu.pdf)

* **A Study of the Fundamental Performance Characteristics of GPUs and CPUs for Database Analytics** (Shanbhag, Madden, Yu, SIGMOD 2020). 提出 Crystal library，一組 GPU 上執行 SQL scan 的 parallel routine。重點看它的 predicate evaluation kernel 設計和 warp utilization 分析，這是你們 filter kernel 的直接參考。[PDF](https://anilshanbhag.in/static/papers/crystal_sigmod20.pdf)

* **Improved Query Performance with Variant Indexes** (O’Neil & Quass, SIGMOD 1997). Bit-sliced index 的原始論文，提出按 bit position 拆 column 來加速 aggregation 和 predicate 的概念。Byte-plane decomposition 是這個想法在 GPU \+ floating point 上的延伸。重點看 Section 4 的 bit-sliced index arithmetic。[PDF](https://www.cs.cmu.edu/~15721-f24/papers/Bitmap_Indices.pdf)

* **MonetDB/X100: Hyper-Pipelining Query Execution** (Boncz, Zukowski, Nes, CIDR 2005). Vectorized query execution 的經典論文。理解 vector-at-a-time 處理模型，這是 GPU kernel 設計的基本參考。重點看 Section 3 的 X100 execution model。[PDF](https://www.cidrdb.org/cidr2005/papers/P19.pdf)

* **How to Wring a Table Dry: Entropy Compression of Relations and Querying of Compressed Relations** (Raman & Swart, VLDB 2006). 說明如何在壓縮格式上直接做 predicate evaluation，特別是 byte-sliced 的比較操作。重點看 Section 4 的 compressed predicate evaluation。[PDF](https://15721.courses.cs.cmu.edu/spring2023/papers/05-compression/p858-raman.pdf)

* **Fixed-Rate Compressed Floating-Point Arrays** (Lindstrom, IEEE TVCG 2014). zfp 壓縮器的論文，同樣利用 low-order bit 的可犧牲性，但用 transform coding 而非 byte-level slicing。了解它跟 byte-plane decomposition 的差異有助於定位自己的 contribution。[PDF](https://vis.cs.ucdavis.edu/vis2014papers/TVCG/papers/2674_20tvcg12-lindstrom-2346458.pdf)

* **ALP: Adaptive Lossless floating-Point Compression** (Afroozeh & Boncz, SIGMOD 2024). DuckDB 目前預設的浮點壓縮方法。了解 production system 怎麼處理浮點壓縮，有助於理解 byte-plane decomposition 在整個 stack 中的定位。[PDF](https://ir.cwi.nl/pub/33334/33334.pdf)

* **BlinkDB: Queries with Bounded Errors and Bounded Response Times on Very Large Data** (Agarwal et al., EuroSys 2013). 基於 sampling 的 approximate query processing。這是你們的對照: BlinkDB 抽樣 row (statistical error)，你們讀全部 row 但減少 bit precision (deterministic error)。理解差異有助於在論文中清楚定位。[PDF](https://sameeragarwal.github.io/blinkdb_eurosys13.pdf)

* **Use-cases of lossy compression for floating-point data** (Argonne National Laboratory)：科學運算資料的實際特徵與壓縮可行性 [PDF](https://www.bsc.es/sites/default/files/public/u2416/compression-barcelona.pdf)

## 建議探索的 Codebase

* **Crystal** ([https://github.com/anilshanbhag/crystal](https://github.com/anilshanbhag/crystal))

  * 先看: src/ 下的 scan kernel 和 predicate evaluation 實作

  * 為什麼: 這是 GPU columnar scan kernel 的直接參考實作，你們的 kernel 可以從這裡出發

* **HeavyDB** ([https://github.com/heavyai/heavydb](https://github.com/heavyai/heavydb))

  * 先看: QueryEngine/ 下的 columnar scan path

  * 為什麼: 了解 production GPU database 怎麼做 full-precision scan，這是你們的 baseline 之一

* **RAPIDS cuDF** ([https://github.com/rapidsai/cudf](https://github.com/rapidsai/cudf))

  * 先看: cpp/src/reduction/ 和 cpp/src/io/parquet/

  * 為什麼: 了解 NVIDIA 官方的 GPU DataFrame 怎麼做 columnar aggregation 和資料格式處理

* **DuckDB** ([https://github.com/duckdb/duckdb](https://github.com/duckdb/duckdb))

  * 先看: src/execution/ 下的 scan 和 aggregation operator

  * 為什麼: 這是 CPU baseline。理解它的 vectorized execution 有助於對比 GPU 的設計選擇

* **Buff (Rust 實作)** ([https://github.com/paradedb/buff-rs](https://github.com/paradedb/buff-rs))

  * 先看: encoding/decoding 的核心邏輯

  * 為什麼: Buff-style encoding 的參考實作，可以對照論文理解編碼細節

## 如何開始?

* **取得國網 cluster 存取權限。** 確認 GPU 型號、CUDA 版本、driver 版本。寫一個 HBM bandwidth test，確認 throughput 接近 spec。跑三種存取模式的 microbenchmark (coalesced, bitmask, gather)，記錄數字。

* **推導 byte-plane 編碼/解碼公式。** 拿一個 FP64 值，手動拆成 8 個 byte，再重建回來。推導讀 k 個 subcolumn 時的最大截斷誤差。再對 Buff-style encoding 做同樣的事: 推導 FOR base/scale 參數下，讀 integer subcolumn \+ k 個 fractional subcolumn 的誤差界。寫在紙上，不要跳過這步。

* **產生 Dataset A (N \= 10^7)，寫 CPU 程式做 byte-plane 分解和重建。** 驗證 bit-exact reconstruction (8 個 subcolumn 全讀回來要跟原值完全一致)。然後 port 到 CUDA，在 GPU 上驗證。

* **實作 byte-plane SUM kernel，跑 Experiment 1。** 對 k \= 1 到 8 量測 throughput。這是第一個真正的結果。如果 throughput 不 scale，立即診斷 coalescing 問題，嘗試 packed read 和 shared memory staging 等替代方案。

## 善用 Claude Code

老師會提供 Claude Code 的 API key。這個專題涉及 CUDA kernel 開發、浮點數編碼、效能分析等多個不熟悉的領域，Claude Code 可以幫你加速，但要用對地方。

### 理解 GPU 框架和 DuckDB 內部

GPU 程式設計和資料庫引擎的內部結構跟一般應用程式很不一樣。你可以把相關的 source code clone 下來，用 Claude Code 讀:

* 「讀 Crystal 的 scan kernel，解釋它怎麼做 coalesced memory access 和 predicate evaluation，每個 warp 處理多少 row」

* 「讀 DuckDB 的 src/execution/aggregate\_hashtable.cpp，解釋 vectorized aggregation 的流程」

* 「解釋 CUDA 的 \_\_ballot\_sync 和 \_\_popc intrinsic 怎麼用在 stream compaction 裡」

### 寫 profiling 和 benchmarking 程式碼

實驗需要大量的 benchmark harness: timing loop、CSV output、多次 warmup \+ measurement。這些是 Claude Code 最適合的任務:

* 「幫我寫一個 CUDA benchmark，分配 N bytes 的 GPU memory，用 kernel 讀一遍，量測 effective bandwidth。支援變動 N、warmup runs、output CSV」

* 「幫我寫一個 Buff-style encoder (C++)，輸入是 FP64 array 和 segment size，輸出是 FOR base/scale \+ byte-plane subcolumns。附 round-trip correctness test」

* 「幫我寫一個 NCCL allreduce wrapper，把多個 GPU 的 partial sum 合併」

### 資料分析和視覺化

實驗產出大量數據需要畫圖:

* 「用 Python matplotlib 畫 precision-throughput curve: x 軸是 subcolumn 數量 k，左 y 軸是 throughput (billion rows/sec)，右 y 軸是 relative error (log scale)，每個 dataset 一條線」

* 「把這組 CSV 資料整理成論文用的 LaTeX table」

### 注意事項

* **先理解再產生。** Byte-plane SUM kernel 的 coalescing 策略和 warp management 是研究貢獻，你必須理解每一行。先請 Claude Code 解釋現有的 GPU scan kernel 怎麼運作，確認你理解 memory transaction pattern，再請它幫你實作。

* **Error bound proof 要自己做。** 截斷誤差的解析推導是論文的理論貢獻，Claude Code 可以幫你驗算，但推導過程和 insight 必須是你自己的。

* **用 Nsight Compute 驗證。** Claude Code 產生的 CUDA kernel 不一定有正確的 memory access pattern。每個 kernel 都要用 Nsight Compute profile，確認 coalesced access 比例和 memory transaction 數量符合預期。

* **提供 context。** 把相關的 CUDA header、kernel launch configuration、GPU spec 一起提供，Claude Code 的輸出會更準確。

* **不要信任未驗證的數字。** Claude Code 可能對 H100 的 spec 或 CUDA 的行為有不準確的認知。所有效能數字都要在你的實際硬體上量測確認。