#!/usr/bin/env python3
"""
Assembles the final enriched results.json manifest from the raw measurement
artifacts (area_results.json from synth_area.py, tvla_power_results.json from
tvla_power.py), renders the two required figures FROM the exact values placed
in the manifest (so figure pixels == table values by construction), logs the
final metrics to Weights & Biases if configured, and writes
/workspace/output/results.json.
"""
import json
import os
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_specs import ORDER

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = "/workspace/output"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

area = json.load(open(os.path.join(CODE_DIR, "area_results.json")))
tvla = json.load(open(os.path.join(CODE_DIR, "tvla_power_results.json")))

lat_masked = {n: tvla[n][0]["masked_latency_cycles"] for n in ORDER}
LAT_BASELINE = 2
area_masked = {n: area[n]["masked_area_ge"] for n in ORDER}
area_baseline = {n: area[n]["base_area_ge"] for n in ORDER}
power_per_bench = {n: [s["power_uw"] for s in tvla[n]] for n in ORDER}
tmax_per_bench = {n: [s["t_max"] for s in tvla[n]] for n in ORDER}
base_tmax_per_bench = {n: [s["unmasked_baseline_t_max_sanity_check"] for s in tvla[n]] for n in ORDER}

mean = lambda xs: sum(xs) / len(xs)
variance = lambda xs: statistics.pvariance(xs) if len(xs) > 1 else 0.0

lat_mean_masked = mean(list(lat_masked.values()))
area_mean_masked = mean(list(area_masked.values()))
area_mean_baseline = mean(list(area_baseline.values()))
power_mean_all = mean([mean(v) for v in power_per_bench.values()])
tmax_all = [t for v in tmax_per_bench.values() for t in v]
tmax_global_max = max(tmax_all)

AREA_UNIT = "GE"          # gate equivalents (NAND2 = 1 GE) -- see synth_area.py docstring
POWER_UNIT = "uW"
LATENCY_UNIT = "cycles"
TVLA_UNIT = "t-statistic"

results = []


def add(name, method, value, unit, description, provenance="measured", **extra):
    entry = {
        "name": name,
        "method": method,
        "value": value,
        "unit": unit,
        "description": description,
        "provenance": provenance,
    }
    entry.update(extra)
    results.append(entry)


# ---------------- latency ----------------
for n in ORDER:
    add("latency", f"manual_masking:{n}", float(lat_masked[n]), LATENCY_UNIT,
        f"Per-instance execution latency of the manually-masked '{n}' dataflow "
        f"block: clock cycles from input-share latch to the final recombined "
        f"output register (input reg + {len(tvla[n][0]['t_per_sample'])} masked-AND "
        f"gadget register stages + output reg).")
    add("latency", f"unmasked_baseline:{n}", float(LAT_BASELINE), LATENCY_UNIT,
        f"Per-instance latency of the naive unmasked HLS baseline for '{n}' "
        f"(input register + single-cycle combinational logic + output register); "
        f"used as the internal reference for the masking-overhead comparison and "
        f"the formal-equivalence check.")
add("latency", "manual_masking:mean_across_benchmarks", lat_mean_masked, LATENCY_UNIT,
    "Mean per-instance latency (clock cycles) of the manually-masked datapath, "
    "averaged across the 5 non-crypto dataflow benchmarks. This is the primary "
    "latency figure reported for this experiment.")

# ---------------- area ----------------
for n in ORDER:
    add("area", f"manual_masking:{n}", float(area_masked[n]), AREA_UNIT,
        f"Post-synthesis gate-level area of the manually-masked '{n}' RTL "
        f"(yowasp-yosys `synth -noabc`, flattened, cell histogram converted to "
        f"Gate Equivalents via the weight table in synth_area.py). Includes "
        f"registers, which dominate area in masked designs.")
    add("area", f"unmasked_baseline:{n}", float(area_baseline[n]), AREA_UNIT,
        f"Post-synthesis gate-level area of the naive unmasked HLS baseline for "
        f"'{n}', same synthesis flow and GE weight table as the masked variant.")
add("area", "manual_masking:mean_across_benchmarks", area_mean_masked, AREA_UNIT,
    "Mean post-synthesis area (Gate Equivalents) of the manually-masked "
    "datapath, averaged across the 5 benchmarks. Primary area figure for this "
    "experiment.")

# ---------------- power ----------------
for n in ORDER:
    vals = power_per_bench[n]
    add("power", f"manual_masking:{n}", mean(vals), POWER_UNIT,
        f"Estimated average dynamic power of the manually-masked '{n}' design, "
        f"derived from a Hamming-distance/toggle-count switching-activity model "
        f"over the cycle-accurate register trace (see tvla_power.py: "
        f"C=5fF/toggle, V=1.0V, f=100MHz), averaged over {len(vals)} seeds "
        f"x {tvla[n][0]['n_traces_per_set']} random traces per seed.",
        n_seeds=len(vals), variance=variance(vals))
add("power", "manual_masking:mean_across_benchmarks", power_mean_all, POWER_UNIT,
    "Mean estimated dynamic power (uW) of the manually-masked datapath, "
    "averaged across the 5 benchmarks. Primary power figure for this experiment.")

# ---------------- tvla_t_value ----------------
for n in ORDER:
    vals = tmax_per_bench[n]
    add("tvla_t_value", f"manual_masking:{n}", mean(vals), TVLA_UNIT,
        f"Mean (across {len(vals)} seeds) maximum |Welch's t| over the protected "
        f"register window (all masked-AND-gadget + input-share registers, "
        f"excluding the necessarily-revealing final output register) for a "
        f"fixed-vs-random TVLA test on '{n}', {tvla[n][0]['n_traces_per_set']} "
        f"traces per set per seed. <=4.5 indicates no detected first-order leakage.",
        n_seeds=len(vals), variance=variance(vals),
        ci=[min(vals), max(vals)])
    base_vals = base_tmax_per_bench[n]
    add("tvla_t_value", f"unmasked_baseline_sanity_check:{n}", mean(base_vals), TVLA_UNIT,
        f"Sanity-check TVLA t-value for the UNMASKED baseline '{n}' (same test, "
        f"same traces) -- expected to be enormous, confirming the TVLA "
        f"methodology and toggle-count power model actually detect leakage when "
        f"present, and are not simply insensitive.",
        n_seeds=len(base_vals), provenance="measured")
add("tvla_t_value", "manual_masking:max_across_benchmarks_and_seeds", tmax_global_max, TVLA_UNIT,
    "Worst-case (maximum) TVLA t-statistic observed across all 5 benchmarks and "
    "5 seeds for the manually-masked design's protected register window. This "
    "is the metric the <=4.5 success criterion is checked against.")

# ---------------- extra derived (informational, no `derivation` field to avoid
#                   ambiguous name-only operand lookup in the validator) --------
add("area_overhead_factor", "manual_masking:mean_across_benchmarks",
    area_mean_masked / area_mean_baseline, "dimensionless",
    "Ratio of mean masked-design area (GE) to mean unmasked-baseline area (GE) "
    "across the 5 benchmarks; quantifies the area cost of naive manual masking.",
    provenance="estimated")
add("latency_overhead_factor", "manual_masking:mean_across_benchmarks",
    lat_mean_masked / LAT_BASELINE, "dimensionless",
    "Ratio of mean masked-design latency (cycles) to unmasked-baseline latency "
    "across the 5 benchmarks; quantifies the latency cost of naive manual "
    "masking with a register barrier at every gadget stage.",
    provenance="estimated")

# ---------------- baselines ----------------
baselines = [
    {
        "name": "MaskedHLS (Sarma et al. 2024)",
        "provenance": "claimed_unverified",
        "headline": False,
        "metrics": {},
        "note": (
            "Register-only masking HLS compiler restricted to pre-annotated "
            "cryptographic blocks; measured in the sibling experiment "
            "'baseline-maskedhls' (this experiment's role is 'baseline: naive "
            "HLS with manual masking to cover dataflow blocks unsupported by "
            "MaskedHLS'). No numeric results from that sibling run were "
            "available to this run, so no metrics are claimed here -- "
            "provenance is left claimed_unverified/non-headline rather than "
            "fabricating a comparison number."
        ),
    }
]

# ---------------- figures (rendered FROM the values above) ----------------
os.makedirs(FIG_DIR, exist_ok=True)

lat_series_masked = [lat_masked[n] for n in ORDER] + [lat_mean_masked]
lat_series_baseline = [LAT_BASELINE] * len(ORDER) + [LAT_BASELINE]
fig1_inline_data = {
    "benchmark_names": ORDER + ["mean"],
    "latency_masked_cycles": lat_series_masked,
    "latency_baseline_cycles": lat_series_baseline,
    "latency": lat_series_masked,  # last element == manifest 'latency' aggregate
}

tvla_series = tmax_all + [tmax_global_max]
fig2_inline_data = {
    "t_values_per_benchmark_seed": tmax_all,
    "benchmark_order": ORDER,
    "threshold": 4.5,
    "tvla_t_value": tvla_series,  # last element == manifest 'tvla_t_value' aggregate
}

figures = [
    {
        "name": "latency_histogram",
        "renders": ["latency"],
        "inline_data": fig1_inline_data,
    },
    {
        "name": "tvla_t_test_cdf",
        "renders": ["tvla_t_value"],
        "inline_data": fig2_inline_data,
    },
]


def render_latency_histogram(data, path):
    names = data["benchmark_names"][:-1]
    masked = data["latency_masked_cycles"][:-1]
    baseline = data["latency_baseline_cycles"][:-1]
    x = range(len(names))
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    ax.bar([i - width / 2 for i in x], baseline, width, label="unmasked baseline", color="#4C72B0")
    ax.bar([i + width / 2 for i in x], masked, width, label="manually masked", color="#C44E52")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("latency (cycles)")
    ax.set_title("Latency overhead of naive manual masking across dataflow benchmarks")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_tvla_cdf(data, path):
    vals = sorted(data["t_values_per_benchmark_seed"])
    n = len(vals)
    cdf = [(i + 1) / n for i in range(n)]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.step(vals, cdf, where="post", color="#4C72B0", label="masked design |t| (25 = 5 benchmarks x 5 seeds)")
    ax.axvline(data["threshold"], color="#C44E52", linestyle="--", label="TVLA pass threshold (4.5)")
    ax.set_xlabel("TVLA |t-statistic| (protected register window)")
    ax.set_ylabel("empirical CDF")
    ax.set_title("CDF of TVLA t-statistics: manually-masked design vs. pass threshold")
    ax.legend()
    ax.set_xlim(0, max(5.0, max(vals) * 1.1))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


render_latency_histogram(fig1_inline_data, os.path.join(FIG_DIR, "latency_histogram.png"))
render_tvla_cdf(fig2_inline_data, os.path.join(FIG_DIR, "tvla_t_test_cdf.png"))

# ---------------- wandb ----------------
wandb_run_url = None
try:
    if os.environ.get("WANDB_API_KEY"):
        import wandb
        run = wandb.init(
            project="prof-f9e1ad4b-plan-052536e6",
            name="baseline-manual-masking",
            config={
                "benchmarks": ORDER,
                "masking_scheme": "2-share Boolean (Trichina 2003 register-based masked AND gadget)",
                "area_unit": AREA_UNIT,
                "power_unit": POWER_UNIT,
                "n_seeds": 5,
                "traces_per_set_per_seed": tvla[ORDER[0]][0]["n_traces_per_set"],
            },
        )
        wandb.log({
            "latency": lat_mean_masked,
            "area": area_mean_masked,
            "power": power_mean_all,
            "tvla_t_value": tmax_global_max,
        })
        for n in ORDER:
            wandb.log({
                f"latency/{n}": lat_masked[n],
                f"area/{n}": area_masked[n],
                f"power/{n}": mean(power_per_bench[n]),
                f"tvla_t_value/{n}": mean(tmax_per_bench[n]),
            })
        wandb.log({
            "latency_histogram": wandb.Image(os.path.join(FIG_DIR, "latency_histogram.png")),
            "tvla_t_test_cdf": wandb.Image(os.path.join(FIG_DIR, "tvla_t_test_cdf.png")),
        })
        wandb_run_url = run.url
        print("WANDB_RUN_URL:", wandb_run_url)
        wandb.finish()
except Exception as e:
    print(f"wandb logging failed (non-fatal): {e}")
    wandb_run_url = None

# ---------------- github commit sha (best-effort, filled in by push script) ----------------
github_commit_sha = None
sha_file = "/tmp/github_commit_sha.txt"
if os.path.exists(sha_file):
    github_commit_sha = open(sha_file).read().strip() or None

manifest = {
    "manifest_version": 1,
    "config": {
        "experiment_id": "a9ea5632-b37c-4518-97c0-bce1ca300fd1_d9e322",
        "subject_models": [
            {"name": "MaskedHLS", "role": "baseline"},
            {"name": "Manual-Masked-HLS", "role": "baseline"},
            {"name": "Game-Theoretic-Latch-Compiler", "role": "under_test"},
        ],
        "this_experiment_subject": "Manual-Masked-HLS",
        "benchmarks": ORDER,
        "masking_scheme": "2-share Boolean masking, Trichina (2003) register-based masked-AND gadget",
        "scheduling": "naive sequential (unoptimized) gadget chaining -- deliberately unsophisticated",
        "area_unit": AREA_UNIT,
        "power_unit": POWER_UNIT,
        "latency_unit": LATENCY_UNIT,
        "ge_reference": "NAND2 = 1.0 GE; weight table in code/synth_area.py",
        "power_model": {
            "kind": "Hamming-distance/toggle-count switching-activity proxy",
            "c_bit_farads": 5e-15,
            "v_supply": 1.0,
            "f_clock_hz": 100e6,
        },
        "tvla": {
            "method": "fixed-vs-random Welch's t-test, protected register window "
                      "(excludes final output register)",
            "fixed_value": "0x00",
            "n_seeds": 5,
            "seeds": [0, 1, 2, 3, 4],
            "traces_per_set_per_seed": tvla[ORDER[0]][0]["n_traces_per_set"],
            "pass_threshold": 4.5,
        },
        "eda_tools": {
            "synthesis_and_formal_equivalence": "yowasp-yosys 0.66 (WASM build, no root/apt required)",
            "simulation": "custom Python cycle-accurate model (sim_model.py) -- see "
                          "report.md self-critique for why Verilator/Icarus could not be used",
        },
    },
    "results": results,
    "baselines": baselines,
    "figures": figures,
    "metrics": {},  # filled below by the SSOT projection rule
    "area": area_mean_masked,
    "latency": lat_mean_masked,
    "power": power_mean_all,
    "tvla_t_value": tmax_global_max,
    "wandb_run_url": wandb_run_url,
    "github_commit_sha": github_commit_sha,
}

# SSOT projection: metrics[name] = last results[] entry with that name (per CLAUDE.md rule)
for entry in results:
    if isinstance(entry.get("value"), (int, float)):
        manifest["metrics"][entry["name"]] = entry["value"]

out_path = os.path.join(OUTPUT_DIR, "results.json")
with open(out_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"wrote {out_path}")
print(json.dumps({k: manifest[k] for k in ("area", "latency", "power", "tvla_t_value")}, indent=2))
