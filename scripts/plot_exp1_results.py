#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _parse_run_name(run_dir_name: str) -> tuple[str, str, str]:
    # run_YYYYMMDD_HHMMSS_jobXXXXX_GPUTAG
    parts = run_dir_name.split("_")
    timestamp = "unknown"
    job = "unknown"
    gpu = "unknown"
    if len(parts) >= 5 and parts[0] == "run":
        timestamp = f"{parts[1]}_{parts[2]}"
        job = parts[3]
        gpu = "_".join(parts[4:])
    return timestamp, job, gpu


def _inventory(exp1_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_dir in sorted(exp1_root.glob("run_*")):
        if not run_dir.is_dir():
            continue

        timestamp, job, gpu_tag = _parse_run_name(run_dir.name)
        csv_path = run_dir / "exp1.csv"
        has_csv = csv_path.exists()

        row: dict[str, object] = {
            "run_dir": run_dir.name,
            "timestamp": timestamp,
            "job": job,
            "gpu_tag_from_dir": gpu_tag,
            "has_exp1_csv": has_csv,
            "row_count": 0,
            "k_min": None,
            "k_max": None,
            "n": None,
            "iters": None,
            "warmup": None,
            "strategy": None,
            "plane_bytes": None,
            "device": None,
            "run_kind": "incomplete_no_csv",
        }

        if has_csv:
            df = pd.read_csv(csv_path)
            row["row_count"] = int(len(df))
            if len(df) > 0:
                row["k_min"] = int(df["k"].min())
                row["k_max"] = int(df["k"].max())
                row["n"] = int(df["n"].iloc[0])
                row["iters"] = int(df["iters"].iloc[0])
                row["warmup"] = int(df["warmup"].iloc[0])
                row["strategy"] = str(df["strategy"].iloc[0])
                row["plane_bytes"] = int(df["plane_bytes"].iloc[0])
                row["device"] = str(df["device"].iloc[0])

                looks_full = (
                    len(df) >= 8
                    and int(df["k"].min()) == 1
                    and int(df["k"].max()) == 8
                    and int(df["n"].iloc[0]) >= 100_000_000
                    and int(df["iters"].iloc[0]) >= 100
                )
                row["run_kind"] = "full" if looks_full else "smoke_or_partial"

        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["run_dir", "has_exp1_csv", "run_kind"])
    return pd.DataFrame(rows)


def _load_full_runs(exp1_root: Path, inv_df: pd.DataFrame) -> pd.DataFrame:
    full_dirs = inv_df[inv_df["run_kind"] == "full"]["run_dir"].tolist()
    chunks: list[pd.DataFrame] = []
    for run_dir_name in full_dirs:
        run_dir = exp1_root / str(run_dir_name)
        df = pd.read_csv(run_dir / "exp1.csv")
        df["run_dir"] = run_dir_name
        df["run_label"] = run_dir_name.replace("run_", "")
        chunks.append(df)
    if not chunks:
        return pd.DataFrame()
    merged = pd.concat(chunks, ignore_index=True)
    merged = merged.sort_values(["run_dir", "k"]).reset_index(drop=True)
    return merged


def _plot_lines(
    df: pd.DataFrame, y_col: str, y_label: str, title: str, output_path: Path
) -> None:
    plt.figure(figsize=(10, 6))
    for run_dir, part in df.groupby("run_dir"):
        part = part.sort_values("k")
        plt.plot(part["k"], part[y_col], marker="o", linewidth=2, label=run_dir)
    plt.xlabel("k (number of byte-planes)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(alpha=0.25, linestyle="--")
    plt.xticks(sorted(df["k"].unique()))
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _plot_normalized(df: pd.DataFrame, output_path: Path) -> None:
    rows: list[pd.DataFrame] = []
    for run_dir, part in df.groupby("run_dir"):
        part = part.sort_values("k").copy()
        base = float(part[part["k"] == 1]["logical_GBps"].iloc[0])
        part["norm_eff"] = part["logical_GBps"] / base
        rows.append(part)
    norm_df = pd.concat(rows, ignore_index=True)
    _plot_lines(
        norm_df,
        y_col="norm_eff",
        y_label="logical_GBps / logical_GBps(k=1)",
        title="Exp1 Normalized Efficiency vs k (full runs)",
        output_path=output_path,
    )


def _plot_per_plane_rate(df: pd.DataFrame, output_path: Path) -> None:
    rate_df = df.copy()
    rate_df["gbps_per_plane"] = rate_df["logical_GBps"] / rate_df["k"]
    _plot_lines(
        rate_df,
        y_col="gbps_per_plane",
        y_label="logical_GBps / k",
        title="Exp1 Per-Plane Effective Rate vs k (full runs)",
        output_path=output_path,
    )


def _plot_speedup_bar(
    df: pd.DataFrame,
    baseline_run: str,
    target_run: str,
    output_path: Path,
    csv_output_path: Path,
) -> None:
    base = (
        df[df["run_dir"] == baseline_run][["k", "ms_per_iter", "logical_GBps"]]
        .rename(
            columns={
                "ms_per_iter": "ms_per_iter_baseline",
                "logical_GBps": "logical_GBps_baseline",
            }
        )
        .copy()
    )
    target = (
        df[df["run_dir"] == target_run][["k", "ms_per_iter", "logical_GBps"]]
        .rename(
            columns={
                "ms_per_iter": "ms_per_iter_target",
                "logical_GBps": "logical_GBps_target",
            }
        )
        .copy()
    )
    merged = pd.merge(base, target, on="k", how="inner").sort_values("k")
    merged["latency_speedup_baseline_over_target"] = (
        merged["ms_per_iter_baseline"] / merged["ms_per_iter_target"]
    )
    merged["throughput_ratio_target_over_baseline"] = (
        merged["logical_GBps_target"] / merged["logical_GBps_baseline"]
    )
    merged.to_csv(csv_output_path, index=False)

    plt.figure(figsize=(10, 5.5))
    plt.bar(
        merged["k"],
        merged["latency_speedup_baseline_over_target"],
        width=0.65,
        color="#2ca02c",
        label="latency speedup = baseline/target",
    )
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("k (number of byte-planes)")
    plt.ylabel("Speedup")
    plt.title(f"Exp1 Speedup by k: {target_run} vs {baseline_run}")
    plt.xticks(sorted(merged["k"].unique()))
    plt.grid(axis="y", alpha=0.25, linestyle="--")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot and summarize exp1 runs")
    parser.add_argument("--exp1-root", required=True, help="Path to results/exp1")
    args = parser.parse_args()

    exp1_root = Path(args.exp1_root).resolve()
    plots_dir = exp1_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    inv_df = _inventory(exp1_root)
    inv_csv = plots_dir / "exp1_run_inventory.csv"
    inv_df.to_csv(inv_csv, index=False)

    full_df = _load_full_runs(exp1_root, inv_df)
    if full_df.empty:
        print(f"Generated: {inv_csv}")
        print("No full runs found; skipped plotting.")
        return

    compare_csv = plots_dir / "exp1_full_run_comparison.csv"
    full_df.to_csv(compare_csv, index=False)

    p1 = plots_dir / "exp1_logical_gbps_vs_k_full_runs.png"
    p2 = plots_dir / "exp1_ms_per_iter_vs_k_full_runs.png"
    p3 = plots_dir / "exp1_normalized_efficiency_vs_k_full_runs.png"
    p4 = plots_dir / "exp1_gbps_per_plane_vs_k_full_runs.png"
    p5 = plots_dir / "exp1_speedup_job16675_vs_job16611.png"
    p5_csv = plots_dir / "exp1_speedup_job16675_vs_job16611.csv"

    _plot_lines(
        full_df,
        y_col="logical_GBps",
        y_label="logical throughput (GB/s)",
        title="Exp1 logical_GBps vs k (full runs)",
        output_path=p1,
    )
    _plot_lines(
        full_df,
        y_col="ms_per_iter",
        y_label="latency per iteration (ms)",
        title="Exp1 ms_per_iter vs k (full runs)",
        output_path=p2,
    )
    _plot_normalized(full_df, p3)
    _plot_per_plane_rate(full_df, p4)

    baseline_run = "run_20260410_144202_job16611_H200"
    target_run = "run_20260410_203754_job16675_H200"
    run_set = set(full_df["run_dir"].unique().tolist())
    if baseline_run in run_set and target_run in run_set:
        _plot_speedup_bar(full_df, baseline_run, target_run, p5, p5_csv)

    print(f"Generated: {inv_csv}")
    print(f"Generated: {compare_csv}")
    print(f"Generated: {p1}")
    print(f"Generated: {p2}")
    print(f"Generated: {p3}")
    print(f"Generated: {p4}")
    if p5.exists() and p5_csv.exists():
        print(f"Generated: {p5}")
        print(f"Generated: {p5_csv}")


if __name__ == "__main__":
    main()
