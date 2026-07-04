#!/usr/bin/env python3
"""Main experiment driver for main-game-theoretic-latches.

Compiles the pinned 6-benchmark suite (benchmarks.py) with the game-
theoretic multi-objective ILP scheduler (scheduler.py), generates RTL
(rtlgen.py), synthesizes with real Yosys and simulates with real Icarus
Verilog (toolchain.py), measures dynamic power from real switching activity
(power_model.py), and runs the 10M-trace TVLA leakage assessment (tvla.py).

Writes:
  synthesized_rtl/<bm>.v, <bm>_tb.v
  synthesis_reports/<bm>_synth.log
  simulation_vcd/<bm>/seed<k>.vcd
  figures/latency_vs_area_pareto.png, figures/tvla_t_statistic_distribution.png
  results.json, report.md (written by assemble_manifest.py, not this file)

This file only produces the raw measurement dict, pickled to
/workspace/output/code/raw_results.json, which assemble_manifest.py then
turns into the enriched results.json + report.md.
"""
import json
import os
import random
import sys
import time

import numpy as np

from benchmarks import all_benchmarks
from scheduler import schedule_dfg
from rtlgen import generate_rtl, generate_testbench
from dfg import golden_eval
from toolchain import synthesize, simulate
from power_model import compute_dynamic_power_mw, CLOCK_FREQ_HZ
from tvla import run_tvla

OUT = "/workspace/output"
RTL_DIR = f"{OUT}/synthesized_rtl"
SYNTH_DIR = f"{OUT}/synthesis_reports"
VCD_DIR = f"{OUT}/simulation_vcd"
W = 8
SEEDS = [0, 1, 2, 3, 4]
N_EQUIV_VECTORS = 50
TVLA_N_TRACES = 10_000_000
PARETO_W_LAT = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]

for d in (RTL_DIR, SYNTH_DIR, VCD_DIR):
    os.makedirs(d, exist_ok=True)


def progress(phase, current, total):
    with open(f"{OUT}/PROGRESS.json", "w") as f:
        json.dump({"phase": phase, "current": current, "total": total}, f)


def run_benchmark(name, dfg):
    print(f"[{name}] scheduling (primary weights)...", flush=True)
    sched = schedule_dfg(dfg, weights=(1.0, 1.0, 1.0))

    rtl = generate_rtl(dfg, sched, w=W)
    rtl_path = f"{RTL_DIR}/{name}.v"
    with open(rtl_path, "w") as f:
        f.write(rtl)

    print(f"[{name}] synthesizing (yosys, generic 4-LUT)...", flush=True)
    synth = synthesize(rtl_path, name, f"{SYNTH_DIR}/{name}_synth.log")

    ins = [n.id for n in dfg.nodes.values() if n.op == "in"]
    bm_vcd_dir = f"{VCD_DIR}/{name}"
    os.makedirs(bm_vcd_dir, exist_ok=True)

    per_seed = []
    for seed in SEEDS:
        rng = random.Random(1000 * seed + hash(name) % 997)
        vectors = []
        for _ in range(N_EQUIV_VECTORS):
            invals = {i: rng.randint(0, 255) for i in ins}
            exp = golden_eval(dfg, invals)
            v = dict(invals)
            v["expected"] = exp
            vectors.append(v)

        vcd_name = f"seed{seed}.vcd"
        tb = generate_testbench(dfg, sched, vectors, w=W, vcd_name=vcd_name)
        tb_path = f"{RTL_DIR}/{name}_tb_seed{seed}.v"
        with open(tb_path, "w") as f:
            f.write(tb)

        sim = simulate(rtl_path, tb_path, bm_vcd_dir, f"{name}_seed{seed}", vcd_name=vcd_name)
        power = compute_dynamic_power_mw(sim["vcd_path"], sched["storage"])

        tvla = run_tvla(dfg, sched, seed=seed, n_traces=TVLA_N_TRACES)

        per_seed.append({
            "seed": seed,
            "equiv_pass": sim["pass"],
            "equiv_fail": sim["fail"],
            "equiv_total": sim["total"],
            "dynamic_power_mw": power["dynamic_power_mw"],
            "tvla_max_t_critical": tvla["max_abs_t"],
            "tvla_max_t_post_recombination": tvla["max_abs_t_post_recombination"],
            "tvla_t_by_cycle_critical": tvla["t_by_cycle_critical"],
        })
        print(f"[{name}] seed={seed} equiv={sim['pass']}/{sim['total']} "
              f"power={power['dynamic_power_mw']:.4f}mW tvla_crit={tvla['max_abs_t']:.3f}", flush=True)

    # Pareto sweep: real RTL + real yosys synthesis at each weight point
    pareto_points = []
    for w_lat in PARETO_W_LAT:
        s = schedule_dfg(dfg, weights=(w_lat, 1.0, 1.0))
        r = generate_rtl(dfg, s, w=W)
        p = f"{RTL_DIR}/{name}_pareto_wlat{w_lat}.v"
        with open(p, "w") as f:
            f.write(r)
        sy = synthesize(p, name, f"{SYNTH_DIR}/{name}_pareto_wlat{w_lat}.log")
        pareto_points.append({
            "w_lat": w_lat,
            "latency_cycles": s["latency_cycles"],
            "area_luts": sy["area_luts"],
            "resource_alloc": s["resource_alloc"],
        })

    return {
        "name": name,
        "sched": {
            "latency_cycles": sched["latency_cycles"],
            "resource_alloc": sched["resource_alloc"],
            "n_latch": sched["n_latch"],
            "n_register": sched["n_register"],
            "n_storage_total": sched["n_latch"] + sched["n_register"],
        },
        "synth": synth,
        "per_seed": per_seed,
        "pareto_points": pareto_points,
    }


def main():
    t_start = time.time()
    benchmarks = all_benchmarks()
    results = {}
    for i, (name, dfg) in enumerate(benchmarks.items()):
        progress("evaluating", i, len(benchmarks))
        results[name] = run_benchmark(name, dfg)
    progress("computing_metrics", len(benchmarks), len(benchmarks))

    with open(f"{OUT}/code/raw_results.json", "w") as f:
        json.dump({
            "benchmarks": results,
            "config": {
                "clock_freq_hz": CLOCK_FREQ_HZ,
                "seeds": SEEDS,
                "n_equiv_vectors_per_seed": N_EQUIV_VECTORS,
                "tvla_n_traces_per_group": TVLA_N_TRACES,
                "pareto_w_lat_sweep": PARETO_W_LAT,
                "wallclock_s": time.time() - t_start,
            },
        }, f, indent=2)
    print(f"done in {time.time()-t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
