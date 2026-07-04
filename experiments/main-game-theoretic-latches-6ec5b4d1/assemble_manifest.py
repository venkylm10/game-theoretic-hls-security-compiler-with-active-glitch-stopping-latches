#!/usr/bin/env python3
"""Turns raw_results.json (run_experiment.py's output) into the enriched
results.json manifest + figures/*.png. report.md is written separately
by hand (see /workspace/output/report.md) since it needs narrative judgment;
this script only emits the machine-checkable numeric contract.
"""
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/workspace/output"


def mean_ci(vals):
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    m = float(vals.mean())
    v = float(vals.var(ddof=1)) if n > 1 else 0.0
    se = math.sqrt(v / n) if n > 1 else 0.0
    return m, v, [m - 1.96 * se, m + 1.96 * se], n


def main():
    raw = json.load(open(f"{OUT}/code/raw_results.json"))
    bms = raw["benchmarks"]
    cfg = raw["config"]
    names = list(bms.keys())

    results = []

    # --- per-benchmark, per-metric entries ---
    lat_vals, area_vals, pow_means, tvla_means, tvla_posts = [], [], [], [], []
    equiv_pass_total, equiv_total_total = 0, 0
    latch_total, reg_total = 0, 0

    for name, b in bms.items():
        lat = b["sched"]["latency_cycles"]
        area = b["synth"]["area_luts"]
        lat_vals.append(lat)
        area_vals.append(area)
        latch_total += b["sched"]["n_latch"]
        reg_total += b["sched"]["n_register"]

        pw = [s["dynamic_power_mw"] for s in b["per_seed"]]
        tv = [s["tvla_max_t_critical"] for s in b["per_seed"]]
        tvp = [s["tvla_max_t_post_recombination"] for s in b["per_seed"]]
        for s in b["per_seed"]:
            equiv_pass_total += s["equiv_pass"]
            equiv_total_total += s["equiv_total"]

        pm, pv, pci, pn = mean_ci(pw)
        tm, tv_, tci, tn = mean_ci(tv)
        tpm, _, _, _ = mean_ci(tvp)
        pow_means.append(pm)
        tvla_means.append(tm)
        tvla_posts.append(tpm)

        results.append({
            "name": "latency_cycles", "method": name, "value": lat, "unit": "cycles",
            "provenance": "measured",
            "description": "ILP-scheduled makespan (cycles) for this benchmark at the primary "
                            "(1,1,1) latency/area/power weight -- deterministic given the DFG "
                            "and resource limits, so variance across seeds is 0 by construction "
                            "(scheduling does not depend on random input data).",
            "n_seeds": 1,
        })
        results.append({
            "name": "area_luts", "method": name, "value": area, "unit": "LUTs",
            "provenance": "measured",
            "description": "4-input-LUT count from real Yosys synthesis (generic synth + "
                            "abc -lut 4) of this benchmark's generated RTL; excludes the "
                            "flip-flop/latch storage cells themselves (reported separately as "
                            "n_dff/n_dlatch), counting only combinational mapped logic.",
            "n_seeds": 1,
        })
        results.append({
            "name": "dynamic_power", "method": name, "value": pm, "unit": "mW",
            "provenance": "measured",
            "description": "Mean dynamic power over 5 seeds, computed from real Icarus "
                            "Verilog switching activity (VCD) using a Hamming-distance "
                            "(bit-toggle) energy model at 100MHz -- see power_model.py for "
                            "the assumed per-toggle energy constants.",
            "variance": pv, "ci": pci, "n_seeds": pn,
        })
        results.append({
            "name": "tvla_max_t_stat", "method": name, "value": tm, "unit": "t-statistic",
            "provenance": "measured",
            "description": "Mean (over 5 seeds) max|Welch t| across all PRE-recombination "
                            "share-derived signals (the security-critical zone -- see "
                            "tvla.py classify_nodes), fixed-vs-random TVLA at N=10,000,000 "
                            "traces per group. Must stay <=4.5 to pass first-order leakage "
                            "assessment.",
            "variance": tv_, "ci": tci, "n_seeds": tn,
        })
        results.append({
            "name": "tvla_max_t_stat_post_recombination", "method": name, "value": tpm,
            "unit": "t-statistic", "provenance": "measured",
            "description": "Mean max|Welch t| across signals AT/AFTER a recombination point "
                            "(the design's legitimate computed output). Expected to be large "
                            "-- this is the circuit correctly revealing its computed result, "
                            "NOT a side-channel flaw -- and is excluded from tvla_max_t_stat "
                            "and the pass/fail threshold. Reported for transparency only.",
        })

    n_pairs = latch_total + reg_total
    latch_fraction = latch_total / n_pairs if n_pairs else 0.0

    lat_mean = float(np.mean(lat_vals))
    area_mean = float(np.mean(area_vals))
    pow_mean = float(np.mean(pow_means))
    tvla_mean = float(np.mean(tvla_means))
    equiv_rate = 100.0 * equiv_pass_total / equiv_total_total if equiv_total_total else None

    results.append({
        "name": "latency_cycles", "method": "mean_across_benchmarks", "value": lat_mean,
        "unit": "cycles", "provenance": "measured",
        "description": "Mean ILP-scheduled latency across the 6-benchmark suite at the "
                        "primary (1,1,1) weight vector. Headline latency metric.",
        "n_seeds": 1,
    })
    results.append({
        "name": "area_luts", "method": "mean_across_benchmarks", "value": area_mean,
        "unit": "LUTs", "provenance": "measured",
        "description": "Mean real-synthesized LUT count across the 6-benchmark suite. "
                        "Headline area metric.",
        "n_seeds": 1,
    })
    results.append({
        "name": "dynamic_power", "method": "mean_across_benchmarks", "value": pow_mean,
        "unit": "mW", "provenance": "measured",
        "description": "Mean measured dynamic power across the 6-benchmark suite (each "
                        "itself a 5-seed mean). Headline power metric.",
        "n_seeds": 5,
    })
    results.append({
        "name": "tvla_max_t_stat", "method": "mean_across_benchmarks", "value": tvla_mean,
        "unit": "t-statistic", "provenance": "measured",
        "description": "Mean critical-zone max|t| across the 6-benchmark suite (each itself "
                        "a 5-seed mean at N=10M traces/group). Headline TVLA metric -- must "
                        "stay <=4.5 per success_criteria.",
        "n_seeds": 5,
    })
    results.append({
        "name": "functional_equivalence_pass_rate", "method": "all_benchmarks", "value": equiv_rate,
        "unit": "%", "provenance": "measured",
        "description": "Percentage of (test vector, output port) pairs across all 6 "
                        "benchmarks x 5 seeds whose Icarus Verilog RTL simulation output "
                        "exactly matched the independent golden Python evaluation "
                        "(dfg.golden_eval) -- the functional-correctness / formal-equivalence "
                        "check the methodology review required.",
    })
    results.append({
        "name": "active_latch_adoption_fraction", "method": "all_benchmarks", "value": latch_fraction * 100.0,
        "unit": "%", "provenance": "measured",
        "description": "Fraction of all storage elements (primary-input registers + "
                        "functional-unit output registers) across the whole benchmark suite "
                        "that the ILP chose to implement as an active glitch-stopping latch "
                        "rather than a full register, at the primary (1,1,1) weight vector.",
    })

    # --- Pareto figure ---
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    pareto_inline = {}
    for c, name in zip(colors, names):
        pts = sorted(bms[name]["pareto_points"], key=lambda p: p["latency_cycles"])
        xs = [p["latency_cycles"] for p in pts]
        ys = [p["area_luts"] for p in pts]
        ax.plot(xs, ys, marker="o", color=c, label=name)
        pareto_inline[f"{name}_latency_cycles"] = xs
        pareto_inline[f"{name}_area_luts"] = ys
    ax.set_xlabel("Latency (cycles)")
    ax.set_ylabel("Area (LUTs, real Yosys synthesis)")
    ax.set_title("Latency vs. area trade-off from the ILP weight sweep\n(per benchmark, w_area=w_power=1, w_lat swept)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/latency_vs_area_pareto.png", dpi=130)
    plt.close(fig)

    # --- TVLA distribution figure ---
    all_crit_t = []
    all_post_t = []
    for name, b in bms.items():
        for s in b["per_seed"]:
            all_crit_t.extend(s["tvla_t_by_cycle_critical"].values())
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(all_crit_t, bins=30, color="tab:blue", alpha=0.8, label="critical (pre-recombination)")
    ax.axvline(4.5, color="red", linestyle="--", label="+-4.5 threshold")
    ax.axvline(-4.5, color="red", linestyle="--")
    ax.set_xlabel("Welch t-statistic")
    ax.set_ylabel("Count (per-node, per-seed samples)")
    ax.set_title("TVLA t-statistic distribution, critical (pre-recombination) zone\nN=10,000,000 traces/group, all 6 benchmarks x 5 seeds")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/tvla_t_statistic_distribution.png", dpi=130)
    plt.close(fig)

    figures = [
        {
            "name": "latency_vs_area_pareto",
            "renders": ["latency_cycles", "area_luts"],
            "inline_data": {**pareto_inline, "latency_cycles": lat_mean, "area_luts": area_mean},
        },
        {
            "name": "tvla_t_statistic_distribution",
            "renders": ["tvla_max_t_stat"],
            "inline_data": {"all_critical_t_samples": all_crit_t, "tvla_max_t_stat": tvla_mean},
        },
    ]

    manifest = {
        "manifest_version": 1,
        "config": {
            "clock_freq_hz": cfg["clock_freq_hz"],
            "technology": "generic FPGA, 4-input LUT (yosys synth + abc -lut 4)",
            "area_unit": "LUTs",
            "power_unit": "mW",
            "benchmarks": names,
            "seeds": cfg["seeds"],
            "n_equiv_vectors_per_seed": cfg["n_equiv_vectors_per_seed"],
            "tvla_n_traces_per_group": cfg["tvla_n_traces_per_group"],
            "pareto_w_lat_sweep": cfg["pareto_w_lat_sweep"],
            "ilp_solver": "PuLP/CBC, gapRel=0.001",
            "wallclock_s": cfg["wallclock_s"],
        },
        "results": results,
        "baselines": [],
        "figures": figures,
        "metrics": {},
        "latency_cycles": lat_mean,
        "area_luts": area_mean,
        "dynamic_power": pow_mean,
        "tvla_max_t_stat": tvla_mean,
        "wandb_run_url": None,
        "github_commit_sha": None,
        "validation_status": None,
    }
    for r in results:
        if isinstance(r.get("value"), (int, float)):
            manifest["metrics"][r["name"]] = r["value"]

    with open(f"{OUT}/results.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("wrote results.json")
    print(f"lat_mean={lat_mean:.2f} area_mean={area_mean:.1f} pow_mean={pow_mean:.4f} tvla_mean={tvla_mean:.3f} equiv_rate={equiv_rate}")


if __name__ == "__main__":
    main()
