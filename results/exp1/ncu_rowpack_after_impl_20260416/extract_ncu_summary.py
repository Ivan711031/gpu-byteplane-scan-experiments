import csv
import re
import subprocess
from pathlib import Path


ROOT = Path("results/exp1/ncu_rowpack_after_impl_20260416")
REPORTS = [
    ("byte_ilp4_k1", ROOT / "byte_ilp4_k1.ncu-rep"),
    ("byte_ilp4_k4", ROOT / "byte_ilp4_k4.ncu-rep"),
    ("byte_ilp4_k8", ROOT / "byte_ilp4_k8.ncu-rep"),
    ("rowpack4_k1", ROOT / "rowpack4_k1.ncu-rep"),
    ("rowpack4_k4", ROOT / "rowpack4_k4.ncu-rep"),
    ("rowpack4_k8", ROOT / "rowpack4_k8.ncu-rep"),
    ("rowpack16_k1", ROOT / "rowpack16_k1.ncu-rep"),
    ("rowpack16_k4", ROOT / "rowpack16_k4.ncu-rep"),
    ("rowpack16_k8", ROOT / "rowpack16_k8.ncu-rep"),
    ("contiguous64", ROOT / "contiguous64.ncu-rep"),
]


def ncu_import(report: Path, *args: str) -> str:
    cmd = ["ncu", "--import", str(report), *args]
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


def csv_rows(text: str):
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('"ID","Process ID"'))
    return list(csv.DictReader(lines[start:]))


def rule_text(rows, section, rule):
    for row in rows:
        if row.get("Section Name") == section and row.get("Rule Name") == rule:
            return row.get("Rule Description", "")
    return ""


summary_fields = [
    "report",
    "kernel",
    "duration_us",
    "memory_throughput_TBps",
    "dram_throughput_pct",
    "l2_throughput_pct",
    "compute_sm_pct",
    "sm_busy_pct",
    "issue_slots_busy_pct",
    "eligible_warps_per_scheduler",
    "issued_warp_per_scheduler",
    "active_warps_per_scheduler",
    "warp_cycles_per_issued_inst",
    "achieved_occupancy_pct",
    "registers_per_thread",
    "local_spill_requests",
    "executed_instructions",
    "cpi_stall_note",
]

summary = []
for name, report in REPORTS:
    rows = csv_rows(ncu_import(report, "--page", "details", "--csv", "--print-units", "base"))
    vals = {(r.get("Section Name"), r.get("Metric Name")): r.get("Metric Value", "") for r in rows}
    dur_ns = vals.get(("GPU Speed Of Light Throughput", "Duration"), "")
    mem_bps = vals.get(("Memory Workload Analysis", "Memory Throughput"), "")
    duration_us = float(dur_ns) / 1000.0 if dur_ns else ""
    mem_tbps = float(mem_bps) / 1e12 if mem_bps else ""
    summary.append(
        {
            "report": name,
            "kernel": rows[0].get("Kernel Name", "") if rows else "",
            "duration_us": f"{duration_us:.3f}" if duration_us != "" else "",
            "memory_throughput_TBps": f"{mem_tbps:.3f}" if mem_tbps != "" else "",
            "dram_throughput_pct": vals.get(("GPU Speed Of Light Throughput", "DRAM Throughput"), ""),
            "l2_throughput_pct": vals.get(("GPU Speed Of Light Throughput", "L2 Cache Throughput"), ""),
            "compute_sm_pct": vals.get(("GPU Speed Of Light Throughput", "Compute (SM) Throughput"), ""),
            "sm_busy_pct": vals.get(("Compute Workload Analysis", "SM Busy"), ""),
            "issue_slots_busy_pct": vals.get(("Compute Workload Analysis", "Issue Slots Busy"), ""),
            "eligible_warps_per_scheduler": vals.get(("Scheduler Statistics", "Eligible Warps Per Scheduler"), ""),
            "issued_warp_per_scheduler": vals.get(("Scheduler Statistics", "Issued Warp Per Scheduler"), ""),
            "active_warps_per_scheduler": vals.get(("Scheduler Statistics", "Active Warps Per Scheduler"), ""),
            "warp_cycles_per_issued_inst": vals.get(
                ("Warp State Statistics", "Warp Cycles Per Issued Instruction"), ""
            ),
            "achieved_occupancy_pct": vals.get(("Occupancy", "Achieved Occupancy"), ""),
            "registers_per_thread": vals.get(("Launch Statistics", "Registers Per Thread"), ""),
            "local_spill_requests": vals.get(("Memory Workload Analysis", "Local Memory Spilling Requests"), ""),
            "executed_instructions": vals.get(("Instruction Statistics", "Executed Instructions"), ""),
            "cpi_stall_note": rule_text(rows, "WarpStateStats", "CPIStall"),
        }
    )

with (ROOT / "ncu_summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary_fields)
    writer.writeheader()
    writer.writerows(summary)

load_rows = []
for name, report in REPORTS:
    text = ncu_import(report, "--page", "source", "--print-source", "sass")
    found = []
    for line in text.splitlines():
        if "Global Load" not in line or "LDG.E" not in line:
            continue
        op = re.search(r"\bLDG\.E(?:\.[A-Z0-9]+)*\b", line)
        size = re.search(r"Global Load\s+(\d+)", line)
        if not op or not size:
            continue
        item = f"{op.group(0)}:{size.group(1)}b"
        if item not in found:
            found.append(item)
    load_rows.append({"report": name, "load_opcodes": ";".join(found)})

with (ROOT / "sass_load_summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["report", "load_opcodes"])
    writer.writeheader()
    writer.writerows(load_rows)

print("NCU metric summary:")
print("report,duration_us,mem_TBps,dram_pct,l2_pct,sm_pct,occ_pct,regs,eligible,issued,warp_cycles")
for row in summary:
    print(
        ",".join(
            [
                row["report"],
                row["duration_us"],
                row["memory_throughput_TBps"],
                row["dram_throughput_pct"],
                row["l2_throughput_pct"],
                row["compute_sm_pct"],
                row["achieved_occupancy_pct"],
                row["registers_per_thread"],
                row["eligible_warps_per_scheduler"],
                row["issued_warp_per_scheduler"],
                row["warp_cycles_per_issued_inst"],
            ]
        )
    )

print("\nSASS load summary:")
for row in load_rows:
    print(f"{row['report']}: {row['load_opcodes']}")
