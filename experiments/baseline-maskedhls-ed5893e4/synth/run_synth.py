"""
Runs yosys (via the pip-installable `yowasp-yosys` WebAssembly build) on one
RTL source, producing a technology-independent gate-level netlist.

Why this flow and not a real ASIC/FPGA synthesis: the pod has no root (no
`apt-get install yosys/verilator`, no liberty/PDK), so synthesis targets a
generic 2-input-gate library via `techmap; simplemap` (skips ABC's logic
optimization -- ABC hung/stalled indefinitely on this HLS-generated netlist's
~150k pre-optimization RTL cells in this sandboxed WASM runtime, and its
result would not be reproducible without a real PDK anyway). The reported
"gates" count is therefore a naive-but-honest technology-independent gate
count, not an ABC- or PDK-optimized area number -- see report.md.
"""
import json
import subprocess
import sys
import os

YOSYS_SCRIPT = """
read_verilog {rtl}
hierarchy -check -top {top}
proc
opt
memory
opt
techmap
simplemap
opt -fast
stat
write_json {out_json}
"""


def run_synth(rtl_path, top, out_json, log_path, yosys_bin="yowasp-yosys"):
    script = YOSYS_SCRIPT.format(rtl=rtl_path, top=top, out_json=out_json)
    proc = subprocess.run(
        [yosys_bin, "-p", script],
        cwd=os.path.dirname(rtl_path) or ".",
        capture_output=True, text=True, timeout=600,
    )
    with open(log_path, "w") as f:
        f.write(proc.stdout)
        f.write(proc.stderr)
    if proc.returncode != 0 or not os.path.exists(out_json):
        raise RuntimeError(f"yosys synthesis failed for {rtl_path}, see {log_path}")
    return parse_stat(log_path)


def parse_stat(log_path):
    """Pull the cell-type histogram out of the `stat` pass output."""
    counts = {}
    in_stats = False
    with open(log_path) as f:
        for line in f:
            if "Printing statistics" in line:
                in_stats = True
                continue
            if not in_stats:
                continue
            line = line.rstrip("\n")
            parts = line.split()
            if len(parts) == 2 and parts[0].isdigit() and parts[1].startswith("$_"):
                counts[parts[1]] = int(parts[0])
            elif len(parts) == 2 and parts[1] == "cells" and parts[0].isdigit():
                counts["_total_cells"] = int(parts[0])
            elif "End of script" in line:
                break
    return counts


if __name__ == "__main__":
    rtl, top, out_json, log_path = sys.argv[1:5]
    counts = run_synth(rtl, top, out_json, log_path)
    print(json.dumps(counts, indent=2))
