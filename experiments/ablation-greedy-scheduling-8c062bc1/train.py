#!/usr/bin/env python3
"""
Main entry point for the ablation-greedy-scheduling experiment.

This is a compiler/EDA experiment, not a gradient-trained model -- there
is no neural network here (TASK.yaml.subject.hyperparameters is empty and
constraints.gpu_required is false). "Training" in the ML-profile output
contract's sense is: run the full compile -> schedule -> latch-insert ->
bind -> verify -> synthesize(-estimate) -> simulate -> TVLA pipeline for
every benchmark, for BOTH the greedy scheduler under test and the
game-theoretic ILP scheduler recomputed here as the paired reference (see
code/README.md for why the reference is recomputed rather than read from
a sibling pod).

Usage:
    python train.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hls.pipeline import run_all, aggregate
from hls.plotting import plot_pareto_tradeoff, plot_scheduling_overhead

OUTPUT_ROOT = "/workspace/output"


def write_progress(phase, current, total):
    path = os.path.join(OUTPUT_ROOT, "PROGRESS.json")
    with open(path, "w") as f:
        json.dump({"phase": phase, "current": current, "total": total}, f)


def main():
    out_dirs = {
        "rtl": os.path.join(OUTPUT_ROOT, "greedy_rtl"),
        "vcd": os.path.join(OUTPUT_ROOT, "simulation_vcd"),
        "synth": os.path.join(OUTPUT_ROOT, "synthesis_logs"),
    }
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)
    figures_dir = os.path.join(OUTPUT_ROOT, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    write_progress("training", 0, 8)
    t0 = time.time()

    wandb_run_url = None
    wandb_run = None
    try:
        if os.environ.get("WANDB_API_KEY"):
            import wandb
            wandb_run = wandb.init(
                project="prof-f9e1ad4b-plan-052536e6",
                name="ablation-greedy-scheduling",
                config={
                    "scheduler_under_test": "greedy-list-scheduling",
                    "scheduler_reference": "game-theoretic-ilp",
                    "width_bits": 16,
                    "clock_mhz": 200,
                },
            )
            wandb_run_url = wandb_run.url
            print("WANDB_RUN_URL:", wandb_run_url)
    except Exception as e:
        print(f"[warn] wandb init failed, continuing without tracking: {e}")

    def progress_cb(i, total):
        write_progress("training", i, total)
        print(f"=== benchmark {i}/{total} done | elapsed={time.time() - t0:.1f}s ===")

    records = run_all(out_dirs=out_dirs, verbose=print, progress_cb=progress_cb)
    write_progress("computing_metrics", 0, 1)
    agg = aggregate(records)

    print("\n=== aggregate (greedy, under test) ===")
    print(json.dumps(agg["greedy"], indent=2, default=str))
    print("\n=== aggregate (ILP, reference) ===")
    print(json.dumps(agg["ilp"], indent=2, default=str))

    if wandb_run is not None:
        wandb_run.log({
            "latency_cycles": agg["greedy"]["latency_cycles_mean"],
            "area_ge": agg["greedy"]["area_ge_mean"],
            "power_mw": agg["greedy"]["power_mw_mean"],
            "tvla_t_statistic": agg["greedy"]["tvla_max_abs_t_worst"],
            "ilp_latency_cycles": agg["ilp"]["latency_cycles_mean"],
            "ilp_area_ge": agg["ilp"]["area_ge_mean"],
            "ilp_power_mw": agg["ilp"]["power_mw_mean"],
        })

    write_progress("generating_figures", 0, 2)
    pareto_data = {
        "benchmarks": agg["benchmarks"],
        "greedy_latency_cycles": agg["greedy"]["latency_cycles_values"],
        "greedy_area_ge": agg["greedy"]["area_ge_values"],
        "ilp_latency_cycles": agg["ilp"]["latency_cycles_values"],
        "ilp_area_ge": agg["ilp"]["area_ge_values"],
    }
    plot_pareto_tradeoff(pareto_data, os.path.join(figures_dir, "pareto_tradeoff_plot.png"))

    overhead_data = {
        "benchmarks": agg["benchmarks"],
        "latency_inflation_cycles": agg["latency_inflation_cycles"],
    }
    plot_scheduling_overhead(overhead_data, os.path.join(figures_dir, "scheduling_overhead.png"))
    write_progress("writing_results", 0, 1)

    pipeline_output = {
        "records": records,
        "aggregate": agg,
        "wandb_run_url": wandb_run_url,
        "elapsed_seconds": time.time() - t0,
        "figures_inline_data": {
            "pareto_tradeoff_plot": pareto_data,
            "scheduling_overhead": overhead_data,
        },
    }
    out_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline_output.json")
    with open(out_json, "w") as f:
        json.dump(pipeline_output, f, indent=2, default=str)
    print(f"\nWrote {out_json}")
    print(f"Total elapsed: {time.time() - t0:.1f}s")

    if wandb_run is not None:
        wandb_run.finish()

    write_progress("writing_results", 1, 1)


if __name__ == "__main__":
    main()
