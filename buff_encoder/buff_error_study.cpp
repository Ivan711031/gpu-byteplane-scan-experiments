#include "buff_codec.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t kMinK = 1;
constexpr std::size_t kMaxK = 8;
constexpr std::array<std::string_view, 6> kAggregations = {
    "sum", "avg", "count_gt", "min", "max", "var",
};

struct DatasetConfig {
    std::string name;
    std::filesystem::path input_path;
    double threshold = 0.0;
};

struct KahanSum {
    double sum = 0.0;
    double correction = 0.0;

    void add(double value) {
        double adjusted = value - correction;
        double updated = sum + adjusted;
        correction = (updated - sum) - adjusted;
        sum = updated;
    }
};

struct WelfordVariance {
    std::uint64_t count = 0;
    double mean = 0.0;
    double m2 = 0.0;

    void add(double value) {
        count += 1;
        double delta = value - mean;
        mean += delta / static_cast<double>(count);
        double delta2 = value - mean;
        m2 += delta * delta2;
    }

    double variance() const {
        return count == 0 ? 0.0 : m2 / static_cast<double>(count);
    }
};

struct AggregateState {
    KahanSum sum;
    WelfordVariance var;
    double min = std::numeric_limits<double>::infinity();
    double max = -std::numeric_limits<double>::infinity();
    std::uint64_t count_gt = 0;

    void add(double value, double threshold) {
        sum.add(value);
        var.add(value);
        min = std::min(min, value);
        max = std::max(max, value);
        if (value > threshold) {
            count_gt += 1;
        }
    }

    double average() const {
        return var.count == 0 ? 0.0 : sum.sum / static_cast<double>(var.count);
    }
};

struct BoundState {
    double sum_abs = 0.0;
    double avg_abs = 0.0;
    double count_abs = 0.0;
    double min_abs = 0.0;
    double max_abs = 0.0;
    double var_abs = 0.0;

    double value_for(std::string_view aggregation) const {
        if (aggregation == "sum") {
            return sum_abs;
        }
        if (aggregation == "avg") {
            return avg_abs;
        }
        if (aggregation == "count_gt") {
            return count_abs;
        }
        if (aggregation == "min") {
            return min_abs;
        }
        if (aggregation == "max") {
            return max_abs;
        }
        if (aggregation == "var") {
            return var_abs;
        }
        throw std::runtime_error("unknown aggregation");
    }
};

struct KResult {
    AggregateState approx;
    double bound_sum_accumulator = 0.0;
    double bound_second_moment_accumulator = 0.0;
    double bound_count_band = 0.0;
    double bound_max_abs = 0.0;
};

struct MetricRow {
    std::string dataset;
    std::string aggregation;
    std::size_t k = 0;
    double threshold = 0.0;
    double exact = 0.0;
    double approx = 0.0;
    double abs_error = 0.0;
    double rel_error = 0.0;
    double analytic_bound = 0.0;
    double bound_gap = 0.0;
};

std::uint64_t file_value_count(const std::filesystem::path& input_path) {
    std::uint64_t size = std::filesystem::file_size(input_path);
    if (size % sizeof(double) != 0) {
        throw std::runtime_error("input file size is not aligned to FP64");
    }
    return size / sizeof(double);
}

std::uint64_t parse_u64(const std::string& text) {
    std::size_t consumed = 0;
    unsigned long long value = std::stoull(text, &consumed, 10);
    if (consumed != text.size()) {
        throw std::runtime_error("invalid integer: " + text);
    }
    return static_cast<std::uint64_t>(value);
}

double safe_relative_error(double exact, double approx) {
    double abs_error = std::abs(exact - approx);
    double denom = std::max(std::abs(exact), 1e-30);
    return abs_error / denom;
}

double safe_gap(double bound, double empirical) {
    if (empirical == 0.0) {
        return bound == 0.0 ? 1.0 : std::numeric_limits<double>::infinity();
    }
    return bound / empirical;
}

void ensure_directory(const std::filesystem::path& path) {
    std::filesystem::create_directories(path);
}

std::vector<double> default_abs_epsilons() {
    return {
        0.0, 1e-12, 1e-9, 1e-6, 1e-3, 1e-2, 1e-1, 1.0, 10.0,
        100.0, 1e3, 1e4, 1e5, 1e6, 1e8, 1e10, 1e12
    };
}

std::vector<double> default_rel_epsilons() {
    return {
        0.0, 1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1
    };
}

void write_metrics_csv(const std::filesystem::path& output_path, const std::vector<MetricRow>& rows) {
    std::ofstream out(output_path, std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to open metrics csv: " + output_path.string());
    }

    out << "dataset,aggregation,k,threshold,exact,approx,abs_error,rel_error,analytic_bound,bound_gap\n";
    out << std::setprecision(17);
    for (const auto& row : rows) {
        out << row.dataset << ','
            << row.aggregation << ','
            << row.k << ','
            << row.threshold << ','
            << row.exact << ','
            << row.approx << ','
            << row.abs_error << ','
            << row.rel_error << ','
            << row.analytic_bound << ','
            << row.bound_gap << '\n';
    }
}

void write_kstar_csv(const std::filesystem::path& output_path,
                     const std::vector<MetricRow>& rows,
                     const std::vector<double>& epsilons,
                     bool use_relative) {
    std::ofstream out(output_path, std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to open k* csv: " + output_path.string());
    }

    out << "epsilon_type,epsilon,aggregation,dataset,k_star\n";
    out << std::setprecision(17);

    for (std::string_view aggregation : kAggregations) {
        for (const auto& dataset_name : {"Sensor", "Uniform", "Heavy-tailed", "Zipfian"}) {
            std::vector<const MetricRow*> matching;
            for (const auto& row : rows) {
                if (row.aggregation == aggregation && row.dataset == dataset_name) {
                    matching.push_back(&row);
                }
            }
            std::sort(matching.begin(), matching.end(), [](const MetricRow* lhs, const MetricRow* rhs) {
                return lhs->k < rhs->k;
            });

            for (double epsilon : epsilons) {
                std::string k_star = "NA";
                for (const MetricRow* row : matching) {
                    double error = use_relative ? row->rel_error : row->abs_error;
                    if (error <= epsilon) {
                        k_star = std::to_string(row->k);
                        break;
                    }
                }
                out << (use_relative ? "relative" : "absolute") << ','
                    << epsilon << ','
                    << aggregation << ','
                    << dataset_name << ','
                    << k_star << '\n';
            }
        }
    }
}

double nice_positive_floor(double value) {
    if (value > 0.0 && std::isfinite(value)) {
        return value;
    }
    return 1e-18;
}

void write_svg_plot(const std::filesystem::path& output_path,
                    const std::string& title,
                    const std::vector<std::pair<double, double>>& empirical,
                    const std::vector<std::pair<double, double>>& bound) {
    constexpr int width = 760;
    constexpr int height = 420;
    constexpr int left = 72;
    constexpr int right = 24;
    constexpr int top = 48;
    constexpr int bottom = 56;
    const int plot_width = width - left - right;
    const int plot_height = height - top - bottom;

    double min_y = std::numeric_limits<double>::infinity();
    double max_y = 0.0;
    for (const auto& point : empirical) {
        min_y = std::min(min_y, nice_positive_floor(point.second));
        max_y = std::max(max_y, nice_positive_floor(point.second));
    }
    for (const auto& point : bound) {
        min_y = std::min(min_y, nice_positive_floor(point.second));
        max_y = std::max(max_y, nice_positive_floor(point.second));
    }
    if (!std::isfinite(min_y)) {
        min_y = 1e-18;
    }
    if (max_y <= min_y) {
        max_y = min_y * 10.0;
    }

    double log_min = std::log10(min_y);
    double log_max = std::log10(max_y);
    if (log_max - log_min < 1e-12) {
        log_max = log_min + 1.0;
    }

    auto x_for = [&](double k) {
        return left + (k - static_cast<double>(kMinK)) / static_cast<double>(kMaxK - kMinK) * plot_width;
    };
    auto y_for = [&](double value) {
        double log_value = std::log10(nice_positive_floor(value));
        double ratio = (log_value - log_min) / (log_max - log_min);
        return top + (1.0 - ratio) * plot_height;
    };

    auto polyline_points = [&](const std::vector<std::pair<double, double>>& points) {
        std::ostringstream out;
        out << std::fixed << std::setprecision(2);
        for (const auto& point : points) {
            out << x_for(point.first) << ',' << y_for(point.second) << ' ';
        }
        return out.str();
    };

    std::ofstream out(output_path, std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to open svg path: " + output_path.string());
    }

    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width << "\" height=\"" << height << "\" viewBox=\"0 0 "
        << width << ' ' << height << "\">\n";
    out << "<rect width=\"100%\" height=\"100%\" fill=\"#fffdf8\"/>\n";
    out << "<text x=\"" << left << "\" y=\"28\" font-family=\"Georgia, serif\" font-size=\"20\" fill=\"#2b2b2b\">"
        << title << "</text>\n";
    out << "<line x1=\"" << left << "\" y1=\"" << top + plot_height << "\" x2=\"" << left + plot_width
        << "\" y2=\"" << top + plot_height << "\" stroke=\"#555\" stroke-width=\"1.2\"/>\n";
    out << "<line x1=\"" << left << "\" y1=\"" << top << "\" x2=\"" << left
        << "\" y2=\"" << top + plot_height << "\" stroke=\"#555\" stroke-width=\"1.2\"/>\n";

    for (std::size_t k = kMinK; k <= kMaxK; ++k) {
        double x = x_for(static_cast<double>(k));
        out << "<line x1=\"" << x << "\" y1=\"" << top << "\" x2=\"" << x << "\" y2=\"" << top + plot_height
            << "\" stroke=\"#ece7dc\" stroke-width=\"1\"/>\n";
        out << "<text x=\"" << x << "\" y=\"" << top + plot_height + 24
            << "\" text-anchor=\"middle\" font-family=\"Menlo, monospace\" font-size=\"12\" fill=\"#444\">"
            << k << "</text>\n";
    }

    for (int tick = 0; tick <= 4; ++tick) {
        double ratio = static_cast<double>(tick) / 4.0;
        double value = std::pow(10.0, log_min + (1.0 - ratio) * (log_max - log_min));
        double y = top + ratio * plot_height;
        out << "<line x1=\"" << left << "\" y1=\"" << y << "\" x2=\"" << left + plot_width
            << "\" y2=\"" << y << "\" stroke=\"#ece7dc\" stroke-width=\"1\"/>\n";
        out << "<text x=\"" << left - 10 << "\" y=\"" << y + 4
            << "\" text-anchor=\"end\" font-family=\"Menlo, monospace\" font-size=\"12\" fill=\"#444\">"
            << std::scientific << std::setprecision(1) << value << "</text>\n";
        out << std::defaultfloat;
    }

    out << "<polyline fill=\"none\" stroke=\"#c45d2c\" stroke-width=\"2.5\" points=\""
        << polyline_points(empirical) << "\"/>\n";
    out << "<polyline fill=\"none\" stroke=\"#176087\" stroke-width=\"2.5\" stroke-dasharray=\"8 6\" points=\""
        << polyline_points(bound) << "\"/>\n";

    out << "<text x=\"" << left + 12 << "\" y=\"" << top + 16
        << "\" font-family=\"Menlo, monospace\" font-size=\"12\" fill=\"#c45d2c\">empirical abs error</text>\n";
    out << "<text x=\"" << left + 220 << "\" y=\"" << top + 16
        << "\" font-family=\"Menlo, monospace\" font-size=\"12\" fill=\"#176087\">analytic bound</text>\n";
    out << "<text x=\"" << left + plot_width / 2 << "\" y=\"" << height - 12
        << "\" text-anchor=\"middle\" font-family=\"Menlo, monospace\" font-size=\"12\" fill=\"#444\">top-k subcolumns kept</text>\n";
    out << "<text x=\"18\" y=\"" << top + plot_height / 2
        << "\" transform=\"rotate(-90 18 " << top + plot_height / 2
        << ")\" text-anchor=\"middle\" font-family=\"Menlo, monospace\" font-size=\"12\" fill=\"#444\">absolute error (log scale)</text>\n";
    out << "</svg>\n";
}

void write_markdown_report(const std::filesystem::path& output_path,
                           std::uint64_t segment_size,
                           const std::vector<MetricRow>& rows) {
    std::ofstream out(output_path, std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to open report path: " + output_path.string());
    }

    out << "# BUFF Subcolumn Error Study\n\n";
    out << "這份報告量化在正確的 BuFF split 編碼下，只讀取前 `k` 個高位 subcolumn 時，各種 aggregation 的截斷誤差與解析界。\n\n";
    out << "## 實作假設\n\n";
    out << "- Segment size: `" << segment_size << "`\n";
    out << "- 每個 segment 先把值轉成共同 binary fixed-point code `q_i = x_i 2^F`，其中 `F` 是該段可精確表示所有值所需的最大 fractional bit 數。\n";
    out << "- 只對整數部分做 range offset：`q_i = (b + o_i) 2^F + f_i`，其中 `b = min floor(x_i)`、`o_i` 是 integer offset、`f_i` 是 `F` 個 fractional bits。\n";
    out << "- 儲存格式把 `[o_i | f_i]` 從高位到低位切成 byte-oriented subcolumns；若最後一個 subcolumn 不滿 8 bits，保留為 trailing bits plane。\n";
    out << "- `f_trunc(k)` 只保留前 `k` 個最高位 subcolumns，剩餘低位補 `0`，再 materialize 成最近的 `double`。\n";
    out << "- `COUNT(x > t)` thresholds: `Sensor=25`, `Uniform=500`, `Heavy-tailed=1`, `Zipfian=16`.\n";
    out << "- `VAR` 使用 population variance。\n";
    out << "- 目前資料與 encoder 都限定 `finite`、`non-negative` `FP64`；因此零填尾端 bits 會形成 one-sided downward truncation。\n\n";

    out << "## 解析界\n\n";
    out << "記 `F_seg` 為 segment 的 fractional bit 數，`W_seg` 為 integer offset bit 數，`T_seg = F_seg + W_seg`，`w_j` 為第 `j` 個 subcolumn 的 bit-width，`k` 為保留的最高位 subcolumn 數，`tau_seg(k) = T_seg - sum_{j < k} w_j` 為被省略的低位 bit 數。令\n\n";
    out << "`E_seg(k) = 2^{-F_seg} (2^{tau_seg(k)} - 1)`.\n\n";
    out << "這是該 segment 在 zero-fill truncation 下的最大 per-value 絕對誤差。\n\n";

    out << "### Lemma 1 (單值截斷誤差)\n\n";
    out << "對於 segment 中任一 row，若只保留前 `k` 個 subcolumns，則其重建值 `x_i^(k)` 滿足 `0 <= x_i - x_i^(k) <= E_seg(k)`。\n\n";
    out << "證明。把精確 fixed-point code 寫成 `c_i = h_i 2^{tau_seg(k)} + l_i`，其中 `0 <= l_i < 2^{tau_seg(k)}`。截斷後把低 `tau_seg(k)` bits 清成 0，因此 `x_i^(k) = (c_i - l_i) 2^{-F_seg}`，故 `x_i - x_i^(k) = l_i 2^{-F_seg}`，最大值即 `E_seg(k)`。\n\n";
    out << "Typical-case。若被省略的低位 bits 在資料上近似均勻，則 `l_i` 近似均勻分布於 `[0, 2^{tau_seg(k)})`，平均 per-value 誤差約為 `E_seg(k)/2`。由於這裡是向下截斷而不是對稱 rounding，誤差在非負資料上不會在期望上相消。\n\n";

    out << "### Lemma 2 (SUM / AVG)\n\n";
    out << "令 `e_i = x_i - x_i^(k)`。則 `|SUM(X) - SUM(X^(k))| = sum_i e_i <= sum_seg n_seg E_seg(k)`，且 `|AVG(X) - AVG(X^(k))| <= (1/N) sum_seg n_seg E_seg(k)`。\n\n";
    out << "證明。由 Lemma 1 可知 `e_i >= 0` 且 `e_i <= E_seg(k)`。對所有 row 加總得到 SUM 的界，再除以總 row 數 `N` 得到 AVG 的界。\n\n";
    out << "Typical-case。對目前這個 non-negative、zero-fill 的設定，SUM/AVG 的 drift 仍然是 one-sided，因此典型值通常仍與 `N` 成正比，只是常數因子比 worst-case 小。若改成 signed data 搭配 unbiased rounding，則誤差才可能出現顯著抵消。\n\n";

    out << "### Lemma 3 (COUNT(x > t))\n\n";
    out << "對任何 threshold `t`，截斷只能把 true positive 變成 false negative，因此\n\n";
    out << "`0 <= COUNT(x_i > t) - COUNT(x_i^(k) > t) <= sum_seg #{ i in seg : t < x_i <= t + E_seg(k) }`.\n\n";
    out << "證明。若 `x_i^(k) > t`，因為 `x_i >= x_i^(k)`，所以 `x_i > t`，因此不會出現 false positive。row 只有在 `x_i > t` 但 `x_i^(k) <= t` 時才會改變分類；由 Lemma 1，這要求 `x_i - t <= E_seg(k)`。\n\n";
    out << "Typical-case。若 threshold 附近的密度為 `f_seg(t)`，則 segment 的期望誤差大約是 `n_seg f_seg(t) E_seg(k)`。因此這個界本質上是 distribution-dependent，取決於 threshold 附近的資料密度。\n\n";

    out << "### Lemma 4 (MIN / MAX)\n\n";
    out << "對 global minimum 與 maximum 都有\n\n";
    out << "`0 <= MIN(X) - MIN(X^(k)) <= max_seg E_seg(k)`,\n\n";
    out << "`0 <= MAX(X) - MAX(X^(k)) <= max_seg E_seg(k)`.\n\n";
    out << "證明。對每個 row 都有 `x_i^(k) in [x_i - E_seg(k), x_i]`。因此 exact extremum 在截斷後最多下降 `E_seg(k)`；另一方面，任何一個被選為 truncated extremum 的 row，其截斷值也不會比自己的 exact 值少超過 `E_seg(k)`，而 exact 值至少不小於 global minimum、且不大於 global maximum。綜合即可得界。\n\n";
    out << "Typical-case。若真正極值與次極值之間的 gap 明顯大於對應 segment 的 `E_seg(k)`，則排序通常不會改變；只有當 top order statistics 非常接近，或極值 row 本身的尾端 bits 被截得很多時，才需要更多 subcolumns。\n\n";

    out << "### Lemma 5 (VAR)\n\n";
    out << "令 `mu` 與 `mu^(k)` 為 exact 與 truncated mean，`Delta_mu = (1/N) sum_seg n_seg E_seg(k)`，`U_seg = max_{i in seg} x_i`。則\n\n";
    out << "`|VAR(X) - VAR(X^(k))| <= (1/N) sum_seg 2 n_seg U_seg E_seg(k) + (2 mu + Delta_mu) Delta_mu`.\n\n";
    out << "證明。由 `x_i^(k) = x_i - e_i` 且 `0 <= e_i <= E_seg(k)`，得\n\n";
    out << "`x_i^2 - (x_i^(k))^2 = 2 x_i e_i - e_i^2 <= 2 U_seg E_seg(k)`.\n\n";
    out << "對每個 segment 平均後得到 second moment 的界。另一方面，`|mu - mu^(k)| <= Delta_mu`，因此\n\n";
    out << "`|mu^2 - (mu^(k))^2| = |mu - mu^(k)| |mu + mu^(k)| <= Delta_mu (2 mu + Delta_mu)`.\n\n";
    out << "把 second moment 與 mean-square 的兩部分相加，即得 variance 的界。\n\n";
    out << "Typical-case。平方項會放大大值的影響，所以 heavy-tailed 與 outlier-dominated segment 會讓 VAR 的 worst-case bound 特別鬆，這也是實測 gap 的主要來源。\n\n";

    out << "## 產物\n\n";
    out << "- Metrics CSV: [metrics.csv](./results/metrics.csv)\n";
    out << "- Absolute `k*` map: [kstar_absolute.csv](./results/kstar_absolute.csv)\n";
    out << "- Relative `k*` map: [kstar_relative.csv](./results/kstar_relative.csv)\n";
    out << "- Plots: [plots/](./plots)\n\n";

    out << "## 圖表索引\n\n";
    for (std::string_view aggregation : kAggregations) {
        out << "- `" << aggregation << "`: ";
        bool first = true;
        for (const auto& dataset_name : {"Sensor", "Uniform", "Heavy-tailed", "Zipfian"}) {
            if (!first) {
                out << ", ";
            }
            std::string slug = std::string(dataset_name);
            std::replace(slug.begin(), slug.end(), ' ', '_');
            out << "[" << dataset_name << "](./plots/" << aggregation << "_" << slug << ".svg)";
            first = false;
        }
        out << '\n';
    }

    out << "\n## Metrics 節錄\n\n";
    out << "| dataset | aggregation | k | abs error | bound | gap |\n";
    out << "|---|---|---:|---:|---:|---:|\n";
    std::size_t printed = 0;
    for (const auto& row : rows) {
        out << "| " << row.dataset << " | " << row.aggregation << " | " << row.k << " | "
            << std::setprecision(6) << std::scientific << row.abs_error << " | "
            << row.analytic_bound << " | " << row.bound_gap << " |\n";
        printed += 1;
        if (printed >= 16) {
            break;
        }
    }
}

void print_usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0
        << " [--segment-size N] [--max-values N] [--out-dir DIR]\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::uint64_t segment_size = 4096;
        std::uint64_t max_values = 0;
        std::filesystem::path out_dir = "analysis";

        for (int index = 1; index < argc; ++index) {
            std::string_view arg = argv[index];
            if (arg == "--segment-size" && index + 1 < argc) {
                segment_size = parse_u64(argv[++index]);
            } else if (arg == "--max-values" && index + 1 < argc) {
                max_values = parse_u64(argv[++index]);
            } else if (arg == "--out-dir" && index + 1 < argc) {
                out_dir = argv[++index];
            } else if (arg == "--help" || arg == "-h") {
                print_usage(argv[0]);
                return 0;
            } else {
                throw std::runtime_error("unknown option: " + std::string(arg));
            }
        }

        std::vector<DatasetConfig> datasets = {
            {"Sensor", "data/dev/sensor.f64le.bin", 25.0},
            {"Uniform", "data/dev/uniform.f64le.bin", 500.0},
            {"Heavy-tailed", "data/dev/heavy_tailed.f64le.bin", 1.0},
            {"Zipfian", "data/dev/zipfian.f64le.bin", 16.0},
        };

        ensure_directory(out_dir / "results");
        ensure_directory(out_dir / "plots");

        std::vector<MetricRow> rows;
        rows.reserve(datasets.size() * kAggregations.size() * kMaxK);

        for (const auto& dataset : datasets) {
            std::cout << "Studying " << dataset.name << " from " << dataset.input_path << '\n';
            std::uint64_t total_values = file_value_count(dataset.input_path);
            if (max_values != 0) {
                total_values = std::min(total_values, max_values);
            }

            AggregateState exact;
            std::array<KResult, kMaxK + 1> results{};
            std::ifstream input(dataset.input_path, std::ios::binary);
            if (!input) {
                throw std::runtime_error("failed to open dataset: " + dataset.input_path.string());
            }

            std::vector<double> buffer(static_cast<std::size_t>(segment_size));
            std::uint64_t remaining = total_values;
            std::uint64_t processed = 0;
            std::uint64_t progress_step = std::max<std::uint64_t>(1, total_values / 10);
            std::uint64_t next_progress = progress_step;

            while (remaining > 0) {
                std::uint64_t current = std::min<std::uint64_t>(remaining, segment_size);
                input.read(reinterpret_cast<char*>(buffer.data()),
                           static_cast<std::streamsize>(current * sizeof(double)));
                if (!input) {
                    throw std::runtime_error("failed to read dataset payload");
                }

                double segment_max = 0.0;
                for (std::size_t row = 0; row < current; ++row) {
                    double value = buffer[row];
                    exact.add(value, dataset.threshold);
                    segment_max = std::max(segment_max, value);
                }

                buff::EncodedSegment segment =
                    buff::encode_segment(std::span<const double>(buffer.data(), static_cast<std::size_t>(current)));

                for (std::size_t k = kMinK; k <= kMaxK; ++k) {
                    double bound = buff::segment_max_abs_error_bound(segment, k);
                    results[k].bound_sum_accumulator += static_cast<double>(current) * bound;
                    results[k].bound_second_moment_accumulator += static_cast<double>(current) * 2.0 * segment_max * bound;
                    results[k].bound_max_abs = std::max(results[k].bound_max_abs, bound);

                    for (std::size_t row = 0; row < current; ++row) {
                        double value = buffer[row];
                        if (value > dataset.threshold && value <= dataset.threshold + bound) {
                            results[k].bound_count_band += 1.0;
                        }
                    }

                    std::vector<double> approx =
                        buff::decode_segment_top_k(segment, k);
                    for (double value : approx) {
                        results[k].approx.add(value, dataset.threshold);
                    }
                }

                processed += current;
                remaining -= current;
                if (processed >= next_progress || processed == total_values) {
                    std::cout << "  progress " << processed << "/" << total_values << '\n';
                    next_progress += progress_step;
                }
            }

            double exact_sum = exact.sum.sum;
            double exact_avg = exact.average();
            double exact_min = exact.min;
            double exact_max = exact.max;
            double exact_var = exact.var.variance();
            double exact_count = static_cast<double>(exact.count_gt);

            for (std::size_t k = kMinK; k <= kMaxK; ++k) {
                const auto& approx = results[k].approx;
                double approx_sum = approx.sum.sum;
                double approx_avg = approx.average();
                double approx_min = approx.min;
                double approx_max = approx.max;
                double approx_var = approx.var.variance();
                double approx_count = static_cast<double>(approx.count_gt);

                double delta_mu_bound = results[k].bound_sum_accumulator / static_cast<double>(total_values);
                BoundState bound_state;
                bound_state.sum_abs = results[k].bound_sum_accumulator;
                bound_state.avg_abs = delta_mu_bound;
                bound_state.count_abs = results[k].bound_count_band;
                bound_state.min_abs = results[k].bound_max_abs;
                bound_state.max_abs = results[k].bound_max_abs;
                bound_state.var_abs =
                    results[k].bound_second_moment_accumulator / static_cast<double>(total_values) +
                    (2.0 * exact_avg + delta_mu_bound) * delta_mu_bound;

                auto append_row = [&](std::string_view aggregation, double exact_value, double approx_value) {
                    double abs_error = std::abs(exact_value - approx_value);
                    double rel_error = safe_relative_error(exact_value, approx_value);
                    double analytic_bound = bound_state.value_for(aggregation);
                    rows.push_back(MetricRow{
                        .dataset = dataset.name,
                        .aggregation = std::string(aggregation),
                        .k = k,
                        .threshold = dataset.threshold,
                        .exact = exact_value,
                        .approx = approx_value,
                        .abs_error = abs_error,
                        .rel_error = rel_error,
                        .analytic_bound = analytic_bound,
                        .bound_gap = safe_gap(analytic_bound, abs_error),
                    });
                };

                append_row("sum", exact_sum, approx_sum);
                append_row("avg", exact_avg, approx_avg);
                append_row("count_gt", exact_count, approx_count);
                append_row("min", exact_min, approx_min);
                append_row("max", exact_max, approx_max);
                append_row("var", exact_var, approx_var);
            }
        }

        write_metrics_csv(out_dir / "results" / "metrics.csv", rows);
        write_kstar_csv(out_dir / "results" / "kstar_absolute.csv", rows, default_abs_epsilons(), false);
        write_kstar_csv(out_dir / "results" / "kstar_relative.csv", rows, default_rel_epsilons(), true);

        for (std::string_view aggregation : kAggregations) {
            for (const auto& dataset_name : {"Sensor", "Uniform", "Heavy-tailed", "Zipfian"}) {
                std::vector<std::pair<double, double>> empirical;
                std::vector<std::pair<double, double>> bound;
                for (const auto& row : rows) {
                    if (row.aggregation == aggregation && row.dataset == dataset_name) {
                        empirical.push_back({static_cast<double>(row.k), row.abs_error});
                        bound.push_back({static_cast<double>(row.k), row.analytic_bound});
                    }
                }
                std::sort(empirical.begin(), empirical.end());
                std::sort(bound.begin(), bound.end());
                std::string slug = std::string(dataset_name);
                std::replace(slug.begin(), slug.end(), ' ', '_');
                write_svg_plot(out_dir / "plots" / (std::string(aggregation) + "_" + slug + ".svg"),
                               std::string(dataset_name) + " / " + std::string(aggregation),
                               empirical,
                               bound);
            }
        }

        write_markdown_report(out_dir / "BUFF_ERROR_STUDY.md", segment_size, rows);
        std::cout << "Wrote study artifacts to " << out_dir << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "buff_error_study: " << error.what() << '\n';
        return 1;
    }
}
