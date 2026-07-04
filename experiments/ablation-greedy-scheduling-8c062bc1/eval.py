#!/usr/bin/env python3
"""
Re-evaluation entry point. There are no trained weights to load in this
compiler experiment (see train.py docstring) -- the analog of "load
weights + re-evaluate" here is: re-run the deterministic schedule /
latch-insertion / area / power / TVLA pipeline from scratch (same seeded
benchmark suite, same solver formulation) and confirm the headline
numbers reproduce the persisted code/pipeline_output.json from train.py,
i.e. that the reported results are not an artifact of a single lucky run.

Usage:
    python eval.py [path/to/pipeline_output.json]
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hls.pipeline import run_all, aggregate

REL_TOL = 1e-6  # scheduling/area/power are deterministic given fixed seeds;
                 # ILP solve time can vary but the optimum is stable at this tolerance.


def main():
    ref_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "pipeline_output.json")
    with open(ref_path) as f:
        ref = json.load(f)

    print(f"Re-running pipeline for regression check against {ref_path} ...")
    records = run_all(out_dirs=None, verbose=lambda *a, **k: None)
    agg = aggregate(records)

    ref_agg = ref["aggregate"]
    checks = [
        ("greedy.latency_cycles_mean", agg["greedy"]["latency_cycles_mean"], ref_agg["greedy"]["latency_cycles_mean"]),
        ("greedy.area_ge_mean", agg["greedy"]["area_ge_mean"], ref_agg["greedy"]["area_ge_mean"]),
        ("greedy.power_mw_mean", agg["greedy"]["power_mw_mean"], ref_agg["greedy"]["power_mw_mean"]),
        ("ilp.latency_cycles_mean", agg["ilp"]["latency_cycles_mean"], ref_agg["ilp"]["latency_cycles_mean"]),
        ("ilp.area_ge_mean", agg["ilp"]["area_ge_mean"], ref_agg["ilp"]["area_ge_mean"]),
    ]
    all_ok = True
    for label, got, want in checks:
        ok = math.isclose(got, want, rel_tol=REL_TOL) or abs(got - want) < 1e-6
        all_ok &= ok
        print(f"  [{'OK' if ok else 'MISMATCH'}] {label}: got={got:.6f} ref={want:.6f}")

    print("PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
