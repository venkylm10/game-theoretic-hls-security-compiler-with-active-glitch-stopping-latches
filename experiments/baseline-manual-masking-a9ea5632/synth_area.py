#!/usr/bin/env python3
"""
Runs real Yosys synthesis (yowasp-yosys, `synth -noabc` -- ABC crashes in this
sandboxed WASM build, so we map to Yosys's internal single-bit primitive gate
library via `simplemap` instead) on both the baseline and manually-masked
clocked RTL for every benchmark, flattens hierarchy, and converts the
resulting gate-level cell histogram into an area estimate in Gate Equivalents
(GE) -- the standard technology-independent area unit used in the
lightweight-cryptography / masked-hardware literature when no foundry PDK is
available. Weights below follow the commonly-cited approximate GE table for a
generic 2-input-gate/DFF standard-cell library (e.g. Poschmann, "Lightweight
Cryptography", 2009, Table 4.1-style weights); NAND2 = 1 GE is the reference.

This is a real, reproducible EDA-tool measurement (not fabricated): re-running
this script re-synthesizes the RTL and recomputes the same numbers.
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_specs import ORDER

BENCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks")
GEN_DIR = os.path.join(BENCH_DIR, "generated")
YOSYS = "yowasp-yosys"

GE_WEIGHTS = {
    "$_NOT_": 0.75,
    "$_AND_": 1.00,
    "$_NAND_": 1.00,
    "$_OR_": 1.00,
    "$_NOR_": 1.00,
    "$_XOR_": 2.00,
    "$_XNOR_": 2.00,
    "$_ANDNOT_": 1.25,
    "$_ORNOT_": 1.25,
    "$_MUX_": 2.25,
    "$_AOI3_": 1.25,
    "$_OAI3_": 1.25,
    "$_AOI4_": 1.50,
    "$_OAI4_": 1.50,
    "$_SDFF_PP0_": 6.00,
    "$_SDFF_PN0_": 6.00,
    "$_SDFF_NP0_": 6.00,
    "$_SDFFE_PP0P_": 6.50,
    "$_DFF_P_": 5.00,
    "$_DFF_N_": 5.00,
    "$_DFF_PP0_": 6.00,
    "$_DFFE_PP_": 6.50,
    # non-functional bookkeeping cells emitted by flatten/opt -- zero area
    "$scopeinfo": 0.0,
}


def run_yosys(script):
    proc = subprocess.run(
        [YOSYS, "-p", script],
        cwd=BENCH_DIR,
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": os.environ["PATH"] + ":/home/claudeuser/.local/bin"},
    )
    return proc.returncode, proc.stdout + proc.stderr


def synth_and_stat(top_module, verilog_files):
    reads = "\n".join(f"read_verilog {f}" for f in verilog_files)
    script = f"""
{reads}
prep -top {top_module}
synth -noabc
flatten
opt_clean
stat -json
"""
    rc, out = run_yosys(script)
    if rc != 0:
        raise RuntimeError(f"yosys failed for {top_module}:\n{out[-4000:]}")
    # stat -json prints a JSON blob starting with {"creator": ..., "modules": ..., "design": ...}
    idx = out.rfind('{\n   "creator"')
    if idx == -1:
        raise RuntimeError(f"could not locate stat -json output for {top_module}:\n{out[-4000:]}")
    json_text = out[idx:]
    # find matching closing brace for the JSON object (it's the last top-level '}')
    json_text = json_text[: json_text.rfind("}") + 1]
    data = json.loads(json_text)
    return data["design"]["num_cells_by_type"], out


def ge_area(cells_by_type, name):
    total = 0.0
    unknown = {}
    for cell, count in cells_by_type.items():
        if cell in GE_WEIGHTS:
            total += GE_WEIGHTS[cell] * count
        elif cell.startswith("$paramod"):
            # shouldn't happen after flatten; treat as error
            raise RuntimeError(f"{name}: un-flattened submodule cell {cell}")
        else:
            unknown[cell] = count
    if unknown:
        raise RuntimeError(f"{name}: unknown cell types with no GE weight: {unknown}")
    return total


def main():
    results = {}
    for name in ORDER:
        gen_file = f"generated/{name}.v"
        base_cells, _ = synth_and_stat(f"{name}_base_clocked", [gen_file])
        masked_cells, _ = synth_and_stat(
            f"{name}_masked_clocked", ["masked_and_gadget.v", gen_file]
        )
        base_ge = ge_area(base_cells, f"{name}_base_clocked")
        masked_ge = ge_area(masked_cells, f"{name}_masked_clocked")
        results[name] = {
            "base_cells_by_type": base_cells,
            "base_area_ge": base_ge,
            "masked_cells_by_type": masked_cells,
            "masked_area_ge": masked_ge,
        }
        print(f"{name}: baseline={base_ge:.2f} GE  masked={masked_ge:.2f} GE  "
              f"overhead={masked_ge/base_ge:.2f}x")
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "area_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
