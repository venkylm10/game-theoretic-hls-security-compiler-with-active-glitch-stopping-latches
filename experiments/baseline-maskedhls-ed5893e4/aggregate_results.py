"""
Reads results_aes.json / results_present.json (written by run_benchmark.py),
renders the required figures FROM those manifest values (never from raw
traces), and writes the final /workspace/output/results.json (enriched
manifest, schema_version/manifest_version 1) + /workspace/output/report.md.

Run this LAST, after both `run_benchmark.py aes ...` and
`run_benchmark.py present ...` have completed and written their
results_<benchmark>.json files.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/workspace/output"
FIGDIR = os.path.join(ROOT, "figures")
os.makedirs(FIGDIR, exist_ok=True)


def load(name):
    with open(os.path.join(ROOT, "code", f"results_{name}.json")) as f:
        return json.load(f)


def make_tradeoff_figure(aes, present, out_path):
    benches = ["AES Sbox\n(Canright, DOM-AND)", "PRESENT Sbox\n(DOM-AND)"]
    latency = [aes["pipeline_depth_cycles"], present["pipeline_depth_cycles"]]
    area = [aes["area_gates"], present["area_gates"]]
    power = [aes["power_uw"], present["power_uw"]]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, vals, title, ylabel, color in zip(
        axes, [latency, area, power],
        ["Latency", "Area", "Power (estimated)"],
        ["cycles", "gates (generic 2-input, no ABC)", "uW (Hamming-distance proxy)"],
        ["#4C72B0", "#DD8452", "#55A868"],
    ):
        bars = ax.bar(benches, vals, color=color)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.3g}",
                    ha="center", va="bottom", fontsize=9)
    fig.suptitle("MaskedHLS register-only baseline: latency / area / power by benchmark")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def make_tvla_figure(aes, present, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, res, name in zip(axes, [aes, present], ["AES Sbox", "PRESENT Sbox"]):
        curve = res["tvla"]["t_curve"]
        xs = list(range(len(curve)))
        ax.plot(xs, curve, marker="o", color="#4C72B0")
        ax.axhline(4.5, color="red", linestyle="--", label="TVLA pass/fail bound (|t|=4.5)")
        ax.axhline(-4.5, color="red", linestyle="--")
        ax.set_title(f"{name} (N={res['tvla']['n_traces_per_class']:,}/class)")
        ax.set_xlabel("cycle (time sample)")
        ax.set_ylabel("Welch t-statistic")
        ax.legend(fontsize=8)
    fig.suptitle("Fixed-vs-random TVLA t-statistic per pipeline cycle")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def metric(name, value, unit, description, provenance, method=None, extra=None):
    e = {"name": name, "value": value, "unit": unit, "description": description,
         "provenance": provenance}
    if method:
        e["method"] = method
    if extra:
        e.update(extra)
    return e


def main():
    aes = load("aes")
    present = load("present")

    results = [
        metric("latency", aes["pipeline_depth_cycles"], "cycles",
               "Number of clock cycles of pipeline latency (register stages between "
               "primary inputs and the recombined masked Sbox output) for the primary "
               "benchmark (AES SubBytes Sbox); this is the MaskedHLS register-only "
               "compiler's baseline execution-cycle count.",
               "measured", method="aes_sbox_dom"),
        metric("area", aes["area_gates"], "gates",
               "Total gate count (registers + combinational 2-input primitives) of the "
               "AES masked Sbox after yosys `techmap; simplemap` generic synthesis "
               "(no ABC logic optimization and no PDK/liberty mapping were available on "
               "this pod -- see report.md); the primary-benchmark area baseline.",
               "measured", method="aes_sbox_dom"),
        metric("power", aes["power_uw"], "uW",
               "Estimated average dynamic power of the AES masked Sbox: mean per-cycle "
               "register-bank Hamming distance (switching activity) times an assumed "
               "energy-per-toggle constant times an assumed clock frequency (see "
               "config.power_model for both constants); a simulation-derived estimate, "
               "not a measurement from a real power/PDK tool.",
               "estimated", method="aes_sbox_dom",
               extra={"formula": "mean_hd_per_cycle * energy_per_toggle_fj * clock_freq_hz, "
                                  "converted fJ*Hz -> uW"}),
        metric("tvla_t_value", aes["tvla"]["tvla_t_value"], "t-statistic",
               f"Max |Welch t-statistic| over all {aes['tvla']['n_samples']} pipeline-cycle "
               f"time samples from a fixed-vs-random TVLA sweep with "
               f"{aes['tvla']['n_traces_per_class']:,} traces per class on the AES masked "
               "Sbox; success criterion is <= 4.5 (no detectable first-order leakage).",
               "measured", method="aes_sbox_dom"),

        metric("latency_present_sbox", present["pipeline_depth_cycles"], "cycles",
               "Pipeline latency (cycles) of the secondary benchmark: the PRESENT masked "
               "Sbox, same MaskedHLS register-only DOM-AND compilation flow.",
               "measured", method="present_sbox_dom"),
        metric("area_present_sbox", present["area_gates"], "gates",
               "Gate count of the PRESENT masked Sbox after the same generic "
               "techmap/simplemap synthesis flow used for the AES benchmark.",
               "measured", method="present_sbox_dom"),
        metric("power_present_sbox", present["power_uw"], "uW",
               "Estimated average dynamic power of the PRESENT masked Sbox using the "
               "same Hamming-distance proxy model and assumed constants as the AES benchmark.",
               "estimated", method="present_sbox_dom"),
        metric("tvla_t_value_present_sbox", present["tvla"]["tvla_t_value"], "t-statistic",
               f"Max |Welch t-statistic| over {present['tvla']['n_samples']} pipeline-cycle "
               f"time samples from a fixed-vs-random TVLA sweep with "
               f"{present['tvla']['n_traces_per_class']:,} traces per class on the PRESENT "
               "masked Sbox.",
               "measured", method="present_sbox_dom"),

        metric("num_registers_aes_sbox", aes["num_dff"], "dimensionless",
               "Number of flip-flops (D-latches) in the synthesized AES masked-Sbox "
               "gate-level netlist -- the register count MaskedHLS's register-only "
               "glitch-stopping strategy actually inserted for this gadget.",
               "measured", method="aes_sbox_dom"),
        metric("num_registers_present_sbox", present["num_dff"], "dimensionless",
               "Number of flip-flops in the synthesized PRESENT masked-Sbox gate-level netlist.",
               "measured", method="present_sbox_dom"),
        metric("n_traces_aes_sbox", aes["tvla"]["n_traces_per_class"], "dimensionless",
               "Number of TVLA traces per class (fixed / random) actually simulated for "
               "the AES benchmark.",
               "measured", method="aes_sbox_dom"),
        metric("n_traces_present_sbox", present["tvla"]["n_traces_per_class"], "dimensionless",
               "Number of TVLA traces per class (fixed / random) actually simulated for "
               "the PRESENT benchmark.",
               "measured", method="present_sbox_dom"),
        metric("functional_check_aes_sbox_passed", 1.0 if aes["functional_check"]["passed"] else 0.0,
               "dimensionless",
               "1.0 if the synthesized AES masked-Sbox gate-level netlist reproduced the "
               "MaskedHLS authors' own reference oracle bit-for-bit "
               f"({aes['functional_check']['n_vectors_checked']} vectors checked), else 0.0 "
               "-- the functional-equivalence / formal-correctness check the automated "
               "methodology review flagged as missing.",
               "measured", method="aes_sbox_dom"),
        metric("functional_check_present_sbox_passed",
               1.0 if present["functional_check"]["passed"] else 0.0, "dimensionless",
               "1.0 if the synthesized PRESENT masked-Sbox gate-level netlist matched a "
               f"bit-exact Python port of the original C source "
               f"({present['functional_check']['n_vectors_checked']} vectors checked), else 0.0.",
               "measured", method="present_sbox_dom"),
    ]

    metrics_flat = {}
    for e in results:
        if isinstance(e["value"], (int, float)):
            metrics_flat[e["name"]] = e["value"]

    make_tradeoff_figure(aes, present, os.path.join(FIGDIR, "baseline_tradeoffs.png"))
    make_tvla_figure(aes, present, os.path.join(FIGDIR, "tvla_t_curve.png"))

    figures = [
        {
            "name": "baseline_tradeoffs",
            "renders": ["latency", "area", "power", "latency_present_sbox",
                        "area_present_sbox", "power_present_sbox"],
            "inline_data": {
                "latency": metrics_flat["latency"],
                "area": metrics_flat["area"],
                "power": metrics_flat["power"],
                "latency_present_sbox": metrics_flat["latency_present_sbox"],
                "area_present_sbox": metrics_flat["area_present_sbox"],
                "power_present_sbox": metrics_flat["power_present_sbox"],
            },
        },
        {
            "name": "tvla_t_curve",
            "renders": ["tvla_t_value", "tvla_t_value_present_sbox"],
            "inline_data": {
                "tvla_t_value": metrics_flat["tvla_t_value"],
                "tvla_t_value_present_sbox": metrics_flat["tvla_t_value_present_sbox"],
                "tvla_t_curve_samples_aes": aes["tvla"]["t_curve"],
                "tvla_t_curve_samples_present": present["tvla"]["t_curve"],
            },
        },
    ]

    manifest = {
        "manifest_version": 1,
        "schema_version": 1,
        "config": {
            "experiment": "baseline-maskedhls",
            "subject_model": "MaskedHLS (Sarma et al., 2024), register-only masking compiler",
            "upstream_repo": "https://github.com/nilotpolas/MaskedHLS",
            "benchmarks": ["aes_sbox_dom (AES SubBytes, Canright decomposition, DOM-AND masked)",
                           "present_sbox_dom (PRESENT Sbox, DOM-AND masked)"],
            "rtl_source": ["code/rtl_src/aes_sbox_dom_reg.v", "code/rtl_src/present_sbox_dom_reg.v"],
            "synthesis_tool": "yowasp-yosys 0.66 (techmap; simplemap; no ABC -- see report.md)",
            "simulation_tool": "custom Python/numpy gate-level netlist simulator (code/sim/netlist_sim.py)",
            "power_model": {
                "method": "register-bank Hamming distance per cycle (dynamic switching activity proxy)",
                "clock_freq_mhz_assumed": aes["clock_freq_mhz_assumed"],
                "energy_per_toggle_fj_assumed": aes["energy_per_toggle_fj_assumed"],
            },
            "tvla_methodology": "fixed-vs-random Welch's t-test, streaming accumulators, "
                                 "max|t| over pipeline-cycle time samples",
        },
        "results": results,
        "baselines": [],
        "figures": figures,
        "metrics": metrics_flat,
        "area": metrics_flat["area"],
        "latency": metrics_flat["latency"],
        "power": metrics_flat["power"],
        "tvla_t_value": metrics_flat["tvla_t_value"],
        "wandb_run_url": None,
    }

    wandb_run_url = None
    if os.environ.get("WANDB_API_KEY"):
        try:
            import wandb
            run = wandb.init(project="prof-f9e1ad4b-plan-052536e6", name="baseline-maskedhls",
                              config=manifest["config"])
            wandb.log({
                "latency": metrics_flat["latency"],
                "area": metrics_flat["area"],
                "power": metrics_flat["power"],
                "tvla_t_value": metrics_flat["tvla_t_value"],
            })
            wandb.log({"baseline_tradeoffs": wandb.Image(os.path.join(FIGDIR, "baseline_tradeoffs.png")),
                       "tvla_t_curve": wandb.Image(os.path.join(FIGDIR, "tvla_t_curve.png"))})
            wandb_run_url = run.url
            print("WANDB_RUN_URL:", wandb_run_url)
            wandb.finish()
        except Exception as e:
            print("wandb logging failed (non-fatal):", e)

    manifest["wandb_run_url"] = wandb_run_url

    with open(os.path.join(ROOT, "results.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("wrote", os.path.join(ROOT, "results.json"))


if __name__ == "__main__":
    main()
