#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _read_mode_csv(run_dir: Path, mode: str) -> pd.DataFrame:
    path = run_dir / f"exp0_{mode}.csv"
    df = pd.read_csv(path)
    df["mode"] = mode
    return df


def _bytes_to_gib(values: pd.Series) -> pd.Series:
    return values.astype(float) / (1024.0**3)


def _build_ratio_df(df: pd.DataFrame) -> pd.DataFrame:
    seq = (
        df[df["mode"] == "seq"][["bytes", "GBps"]]
        .rename(columns={"GBps": "GBps_seq"})
        .copy()
    )
    gather = (
        df[df["mode"] == "gather"][["bytes", "GBps"]]
        .rename(columns={"GBps": "GBps_gather"})
        .copy()
    )
    merged = pd.merge(seq, gather, on="bytes", how="inner")
    merged["gather_vs_seq_ratio"] = merged["GBps_gather"] / merged["GBps_seq"]
    merged["bytes_GiB"] = _bytes_to_gib(merged["bytes"])
    return merged.sort_values("bytes")


def _plot_bandwidth(df: pd.DataFrame, run_title: str, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    for mode in ["seq", "masked", "gather"]:
        part = df[df["mode"] == mode].sort_values("bytes")
        plt.plot(
            part["bytes"],
            part["GBps"],
            marker="o",
            linewidth=2,
            label=mode,
        )

    plt.xscale("log", base=2)
    plt.xlabel("Bytes per iteration (log2)")
    plt.ylabel("Effective Bandwidth (GB/s)")
    plt.title(f"Exp0 Bandwidth vs Problem Size ({run_title})")
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _plot_latency(df: pd.DataFrame, run_title: str, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    for mode in ["seq", "masked", "gather"]:
        part = df[df["mode"] == mode].sort_values("bytes")
        plt.plot(
            part["bytes"],
            part["ms_per_iter"],
            marker="o",
            linewidth=2,
            label=mode,
        )

    plt.xscale("log", base=2)
    plt.yscale("log", base=10)
    plt.xlabel("Bytes per iteration (log2)")
    plt.ylabel("Latency per iteration (ms, log10)")
    plt.title(f"Exp0 Latency vs Problem Size ({run_title})")
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _plot_ratio(ratio_df: pd.DataFrame, run_title: str, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(
        ratio_df["bytes"],
        ratio_df["gather_vs_seq_ratio"],
        marker="o",
        linewidth=2,
        color="#d62728",
        label="gather / seq",
    )
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.xscale("log", base=2)
    plt.xlabel("Bytes per iteration (log2)")
    plt.ylabel("Bandwidth Ratio")
    plt.title(f"Exp0 Gather-vs-Seq Bandwidth Ratio ({run_title})")
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot exp0 CSV results")
    parser.add_argument("--run-dir", required=True, help="Path to exp0 run directory")
    parser.add_argument(
        "--run-title",
        default="H200 job16608",
        help="Title suffix used in plot titles",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    seq_df = _read_mode_csv(run_dir, "seq")
    masked_df = _read_mode_csv(run_dir, "masked")
    gather_df = _read_mode_csv(run_dir, "gather")

    all_df = pd.concat([seq_df, masked_df, gather_df], ignore_index=True)
    all_df = all_df.sort_values(["mode", "bytes"]).reset_index(drop=True)
    all_df["bytes_GiB"] = _bytes_to_gib(all_df["bytes"])

    ratio_df = _build_ratio_df(all_df)

    bandwidth_png = run_dir / "exp0_bandwidth_vs_size.png"
    latency_png = run_dir / "exp0_latency_vs_size.png"
    ratio_png = run_dir / "exp0_gather_vs_seq_ratio.png"
    summary_csv = run_dir / "exp0_summary_table.csv"
    ratio_csv = run_dir / "exp0_gather_vs_seq_ratio.csv"

    _plot_bandwidth(all_df, args.run_title, bandwidth_png)
    _plot_latency(all_df, args.run_title, latency_png)
    _plot_ratio(ratio_df, args.run_title, ratio_png)

    all_df.to_csv(summary_csv, index=False)
    ratio_df.to_csv(ratio_csv, index=False)

    print(f"Generated: {bandwidth_png}")
    print(f"Generated: {latency_png}")
    print(f"Generated: {ratio_png}")
    print(f"Generated: {summary_csv}")
    print(f"Generated: {ratio_csv}")


if __name__ == "__main__":
    main()
