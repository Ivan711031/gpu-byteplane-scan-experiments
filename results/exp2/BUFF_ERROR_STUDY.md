# BUFF Subcolumn Error Study

這份報告量化在正確的 BuFF split 編碼下，只讀取前 `k` 個高位 subcolumn 時，各種 aggregation 的截斷誤差與解析界。

## 實作假設

- Segment size: `4096`
- 每個 segment 先把值轉成共同 binary fixed-point code `q_i = x_i 2^F`，其中 `F` 是該段可精確表示所有值所需的最大 fractional bit 數。
- 只對整數部分做 range offset：`q_i = (b + o_i) 2^F + f_i`，其中 `b = min floor(x_i)`、`o_i` 是 integer offset、`f_i` 是 `F` 個 fractional bits。
- 儲存格式把 `[o_i | f_i]` 從高位到低位切成 byte-oriented subcolumns；若最後一個 subcolumn 不滿 8 bits，保留為 trailing bits plane。
- `f_trunc(k)` 只保留前 `k` 個最高位 subcolumns，剩餘低位補 `0`，再 materialize 成最近的 `double`。
- `COUNT(x > t)` thresholds: `Sensor=25`, `Uniform=500`, `Heavy-tailed=1`, `Zipfian=16`.
- `VAR` 使用 population variance。
- 目前資料與 encoder 都限定 `finite`、`non-negative` `FP64`；因此零填尾端 bits 會形成 one-sided downward truncation。

## 解析界

記 `F_seg` 為 segment 的 fractional bit 數，`W_seg` 為 integer offset bit 數，`T_seg = F_seg + W_seg`，`w_j` 為第 `j` 個 subcolumn 的 bit-width，`k` 為保留的最高位 subcolumn 數，`tau_seg(k) = T_seg - sum_{j < k} w_j` 為被省略的低位 bit 數。令

`E_seg(k) = 2^{-F_seg} (2^{tau_seg(k)} - 1)`.

這是該 segment 在 zero-fill truncation 下的最大 per-value 絕對誤差。

### Lemma 1 (單值截斷誤差)

對於 segment 中任一 row，若只保留前 `k` 個 subcolumns，則其重建值 `x_i^(k)` 滿足 `0 <= x_i - x_i^(k) <= E_seg(k)`。

證明。把精確 fixed-point code 寫成 `c_i = h_i 2^{tau_seg(k)} + l_i`，其中 `0 <= l_i < 2^{tau_seg(k)}`。截斷後把低 `tau_seg(k)` bits 清成 0，因此 `x_i^(k) = (c_i - l_i) 2^{-F_seg}`，故 `x_i - x_i^(k) = l_i 2^{-F_seg}`，最大值即 `E_seg(k)`。

Typical-case。若被省略的低位 bits 在資料上近似均勻，則 `l_i` 近似均勻分布於 `[0, 2^{tau_seg(k)})`，平均 per-value 誤差約為 `E_seg(k)/2`。由於這裡是向下截斷而不是對稱 rounding，誤差在非負資料上不會在期望上相消。

### Lemma 2 (SUM / AVG)

令 `e_i = x_i - x_i^(k)`。則 `|SUM(X) - SUM(X^(k))| = sum_i e_i <= sum_seg n_seg E_seg(k)`，且 `|AVG(X) - AVG(X^(k))| <= (1/N) sum_seg n_seg E_seg(k)`。

證明。由 Lemma 1 可知 `e_i >= 0` 且 `e_i <= E_seg(k)`。對所有 row 加總得到 SUM 的界，再除以總 row 數 `N` 得到 AVG 的界。

Typical-case。對目前這個 non-negative、zero-fill 的設定，SUM/AVG 的 drift 仍然是 one-sided，因此典型值通常仍與 `N` 成正比，只是常數因子比 worst-case 小。若改成 signed data 搭配 unbiased rounding，則誤差才可能出現顯著抵消。

### Lemma 3 (COUNT(x > t))

對任何 threshold `t`，截斷只能把 true positive 變成 false negative，因此

`0 <= COUNT(x_i > t) - COUNT(x_i^(k) > t) <= sum_seg #{ i in seg : t < x_i <= t + E_seg(k) }`.

證明。若 `x_i^(k) > t`，因為 `x_i >= x_i^(k)`，所以 `x_i > t`，因此不會出現 false positive。row 只有在 `x_i > t` 但 `x_i^(k) <= t` 時才會改變分類；由 Lemma 1，這要求 `x_i - t <= E_seg(k)`。

Typical-case。若 threshold 附近的密度為 `f_seg(t)`，則 segment 的期望誤差大約是 `n_seg f_seg(t) E_seg(k)`。因此這個界本質上是 distribution-dependent，取決於 threshold 附近的資料密度。

### Lemma 4 (MIN / MAX)

對 global minimum 與 maximum 都有

`0 <= MIN(X) - MIN(X^(k)) <= max_seg E_seg(k)`,

`0 <= MAX(X) - MAX(X^(k)) <= max_seg E_seg(k)`.

證明。對每個 row 都有 `x_i^(k) in [x_i - E_seg(k), x_i]`。因此 exact extremum 在截斷後最多下降 `E_seg(k)`；另一方面，任何一個被選為 truncated extremum 的 row，其截斷值也不會比自己的 exact 值少超過 `E_seg(k)`，而 exact 值至少不小於 global minimum、且不大於 global maximum。綜合即可得界。

Typical-case。若真正極值與次極值之間的 gap 明顯大於對應 segment 的 `E_seg(k)`，則排序通常不會改變；只有當 top order statistics 非常接近，或極值 row 本身的尾端 bits 被截得很多時，才需要更多 subcolumns。

### Lemma 5 (VAR)

令 `mu` 與 `mu^(k)` 為 exact 與 truncated mean，`Delta_mu = (1/N) sum_seg n_seg E_seg(k)`，`U_seg = max_{i in seg} x_i`。則

`|VAR(X) - VAR(X^(k))| <= (1/N) sum_seg 2 n_seg U_seg E_seg(k) + (2 mu + Delta_mu) Delta_mu`.

證明。由 `x_i^(k) = x_i - e_i` 且 `0 <= e_i <= E_seg(k)`，得

`x_i^2 - (x_i^(k))^2 = 2 x_i e_i - e_i^2 <= 2 U_seg E_seg(k)`.

對每個 segment 平均後得到 second moment 的界。另一方面，`|mu - mu^(k)| <= Delta_mu`，因此

`|mu^2 - (mu^(k))^2| = |mu - mu^(k)| |mu + mu^(k)| <= Delta_mu (2 mu + Delta_mu)`.

把 second moment 與 mean-square 的兩部分相加，即得 variance 的界。

Typical-case。平方項會放大大值的影響，所以 heavy-tailed 與 outlier-dominated segment 會讓 VAR 的 worst-case bound 特別鬆，這也是實測 gap 的主要來源。

## 產物

- Metrics CSV: [metrics.csv](./results/metrics.csv)
- Absolute `k*` map: [kstar_absolute.csv](./results/kstar_absolute.csv)
- Relative `k*` map: [kstar_relative.csv](./results/kstar_relative.csv)
- Plots: [plots/](./plots)

## 圖表索引

- `sum`: [Sensor](./plots/sum_Sensor.svg), [Uniform](./plots/sum_Uniform.svg), [Heavy-tailed](./plots/sum_Heavy-tailed.svg), [Zipfian](./plots/sum_Zipfian.svg)
- `avg`: [Sensor](./plots/avg_Sensor.svg), [Uniform](./plots/avg_Uniform.svg), [Heavy-tailed](./plots/avg_Heavy-tailed.svg), [Zipfian](./plots/avg_Zipfian.svg)
- `count_gt`: [Sensor](./plots/count_gt_Sensor.svg), [Uniform](./plots/count_gt_Uniform.svg), [Heavy-tailed](./plots/count_gt_Heavy-tailed.svg), [Zipfian](./plots/count_gt_Zipfian.svg)
- `min`: [Sensor](./plots/min_Sensor.svg), [Uniform](./plots/min_Uniform.svg), [Heavy-tailed](./plots/min_Heavy-tailed.svg), [Zipfian](./plots/min_Zipfian.svg)
- `max`: [Sensor](./plots/max_Sensor.svg), [Uniform](./plots/max_Uniform.svg), [Heavy-tailed](./plots/max_Heavy-tailed.svg), [Zipfian](./plots/max_Zipfian.svg)
- `var`: [Sensor](./plots/var_Sensor.svg), [Uniform](./plots/var_Uniform.svg), [Heavy-tailed](./plots/var_Heavy-tailed.svg), [Zipfian](./plots/var_Zipfian.svg)

## Metrics 節錄

| dataset | aggregation | k | abs error | bound | gap |
|---|---|---:|---:|---:|---:|
| Sensor | sum | 1 | 6.065090e+05 | 1.212964e+06 | 1.999911e+00 |
| Sensor | avg | 1 | 6.065090e-03 | 1.212964e-02 | 1.999911e+00 |
| Sensor | count_gt | 1 | 1.471420e+05 | 1.471420e+05 | 1.000000e+00 |
| Sensor | min | 1 | 5.413836e-03 | 3.125000e-02 | 5.772248e+00 |
| Sensor | max | 1 | 7.297654e-04 | 3.125000e-02 | 4.282198e+01 |
| Sensor | var | 1 | 8.014433e-04 | 1.239011e+00 | 1.545974e+03 |
| Sensor | sum | 2 | 2.369124e+03 | 4.738141e+03 | 1.999955e+00 |
| Sensor | avg | 2 | 2.369124e-05 | 4.738141e-05 | 1.999955e+00 |
| Sensor | count_gt | 2 | 5.930000e+02 | 5.930000e+02 | 1.000000e+00 |
| Sensor | min | 2 | 1.222418e-05 | 1.220703e-04 | 9.985972e+00 |
| Sensor | max | 2 | 5.837873e-05 | 1.220703e-04 | 2.091007e+00 |
| Sensor | var | 2 | 3.019631e-06 | 4.839313e-03 | 1.602617e+03 |
| Sensor | sum | 3 | 9.253884e+00 | 1.850836e+01 | 2.000064e+00 |
| Sensor | avg | 3 | 9.253884e-08 | 1.850836e-07 | 2.000064e+00 |
| Sensor | count_gt | 3 | 0.000000e+00 | 0.000000e+00 | 1.000000e+00 |
| Sensor | min | 3 | 6.483210e-08 | 4.768372e-07 | 7.354955e+00 |
