"""
Main per-benchmark driver for the MaskedHLS baseline experiment.

For one masked-Sbox benchmark (AES or PRESENT, both real MaskedHLS/SecureHLS
DOM-AND register-balanced outputs from the upstream nilotpolas/MaskedHLS
repo), this:
  1. synthesizes the RTL to a generic gate-level netlist with yosys (area, DFF count)
  2. determines pipeline latency (cycles) from the synthesized netlist
  3. runs a functional-equivalence check against a golden reference model
  4. runs fixed-vs-random TVLA over N traces, simulating the gate-level
     netlist directly and using register Hamming-distance as a dynamic-power
     proxy (see sim/power_tvla.py for the full methodology + limitations)
  5. writes a JSON summary + a CSV of a representative trace sample

Designed to be invoked as `python3 run_benchmark.py aes 10000000` (or
`present ...`) and run to completion in one process -- for the multi-million
trace TVLA phase this can take tens of minutes, so it prints progress lines
periodically so it can be run under `nohup` and monitored via log tail
without ever needing a single foreground tool call to block that long.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sim"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "synth"))

import numpy as np
from netlist_sim import Netlist
from power_tvla import run_tvla, bits_n
from run_synth import run_synth

ROOT = "/workspace/output"
CLOCK_FREQ_MHZ = 100.0          # assumed clock frequency (documented estimate, no real PDK/timing report)
ENERGY_PER_TOGGLE_FJ = 0.5      # assumed dynamic energy per register-bit toggle at a generic small-feature node

BENCHMARKS = {
    "aes": {
        "rtl": "code/rtl_src/aes_sbox_dom_reg.v",
        "top": "sbox",
        "port_groups": {
            "share_ports": [("t0", "t1")],
            "plaintext_widths": [8],
            "mask_ports": [f"r{i}" for i in range(36)],
        },
        "extra_const_ports_fn": "aes_dec_constants",
        "gate_json": "compiled_rtl/aes_sbox_gatelevel.json",
        "synth_log": "synthesis_logs/aes_synth.log",
        "label": "AES SubBytes Sbox (Canright, DOM-AND masked)",
    },
    "present": {
        "rtl": "code/rtl_src/present_sbox_dom_reg.v",
        "top": "sbox",
        "port_groups": {
            "share_ports": [("x0_0", "x0_1"), ("x1_0", "x1_1"), ("x2_0", "x2_1"), ("x3_0", "x3_1")],
            "plaintext_widths": [1, 1, 1, 1],
            "mask_ports": ["r"],
        },
        "extra_const_ports_fn": None,
        "gate_json": "compiled_rtl/present_sbox_gatelevel.json",
        "synth_log": "synthesis_logs/present_synth.log",
        "label": "PRESENT Sbox (DOM-AND masked)",
    },
}


def aes_dec_constants():
    from aes_reference import REFERENCE_DEC
    return REFERENCE_DEC


def functional_check_aes(nl):
    from aes_reference import load_reference_table, REFERENCE_R, REFERENCE_DEC
    depth = nl.pipeline_depth()
    table = load_reference_table()
    rng = np.random.default_rng(42)
    n = 128
    xors = rng.choice(256, size=n, replace=False)
    t0v = rng.integers(0, 256, size=n)
    t1v = t0v ^ xors
    pi_bits = {"t0": bits_n(t0v.astype(np.uint32), 8), "t1": bits_n(t1v.astype(np.uint32), 8)}
    for k, v in REFERENCE_R.items():
        pi_bits[k] = bits_n(np.full(n, v, dtype=np.uint32), len(nl.ports[k]["bits"]))
    for k, v in REFERENCE_DEC.items():
        pi_bits[k] = bits_n(np.full(n, v, dtype=np.uint32), len(nl.ports[k]["bits"]))
    state = nl.new_state(n)
    for _ in range(depth + 2):
        frame = {"clk": np.ones(n, dtype=bool)}
        frame.update(pi_bits)
        _, state, outputs = nl.step(frame, state, n)

    def b2i(bl):
        v = np.zeros(len(bl[0]), dtype=np.uint32)
        for i, b in enumerate(bl):
            v |= (b.astype(np.uint32) << i)
        return v

    y0, y1 = b2i(outputs["y0"]), b2i(outputs["y1"])
    computed = (y0 ^ y1) & 0xFF
    expected = np.array([table[int(x)] for x in xors])
    ok = bool(np.array_equal(computed, expected))
    return {
        "method": "compare recombined RTL output against MaskedHLS authors' own "
                  "iverilog-generated reference oracle (output_original.txt), "
                  "128 randomly sampled plaintext-xor values, random share splits",
        "n_vectors_checked": n,
        "passed": ok,
    }


def functional_check_present(nl):
    from golden_present import sbox as golden_sbox
    depth = nl.pipeline_depth()
    rng = np.random.default_rng(43)
    n = 4000
    x0_0 = rng.integers(0, 2, n); x1_0 = rng.integers(0, 2, n)
    x2_0 = rng.integers(0, 2, n); x3_0 = rng.integers(0, 2, n)
    x0_1 = rng.integers(0, 2, n); x1_1 = rng.integers(0, 2, n)
    x2_1 = rng.integers(0, 2, n); x3_1 = rng.integers(0, 2, n)
    r = rng.integers(0, 2, n)
    pi_bits = {
        "x0_0": [x0_0.astype(bool)], "x1_0": [x1_0.astype(bool)],
        "x2_0": [x2_0.astype(bool)], "x3_0": [x3_0.astype(bool)],
        "x0_1": [x0_1.astype(bool)], "x1_1": [x1_1.astype(bool)],
        "x2_1": [x2_1.astype(bool)], "x3_1": [x3_1.astype(bool)],
        "r": [r.astype(bool)],
    }
    state = nl.new_state(n)
    for _ in range(depth + 3):
        frame = {"clk": np.ones(n, dtype=bool)}
        frame.update(pi_bits)
        _, state, outputs = nl.step(frame, state, n)
    names = ["Y0_0", "Y1_0", "Y2_0", "Y3_0", "Y0_1", "Y1_1", "Y2_1", "Y3_1"]
    sim_out = np.stack([outputs[nn][0].astype(int) for nn in names], axis=1)
    golden_out = np.zeros((n, 8), dtype=int)
    for i in range(n):
        golden_out[i] = golden_sbox(int(x0_0[i]), int(x1_0[i]), int(x2_0[i]), int(x3_0[i]),
                                     int(x0_1[i]), int(x1_1[i]), int(x2_1[i]), int(x3_1[i]), int(r[i]))
    ok = bool(np.array_equal(sim_out, golden_out))
    return {
        "method": "compare recombined RTL output against a bit-exact Python "
                  "transliteration of the original present_domand.c source "
                  "compiled by MaskedHLS, 4000 random (shares, mask) trials",
        "n_vectors_checked": n,
        "passed": ok,
    }


def main():
    bench_name = sys.argv[1]
    n_traces = int(sys.argv[2])
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    cfg = BENCHMARKS[bench_name]

    log = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] [{bench_name}] {msg}", flush=True)

    rtl_path = os.path.join(ROOT, cfg["rtl"])
    gate_json = os.path.join(ROOT, cfg["gate_json"])
    synth_log = os.path.join(ROOT, cfg["synth_log"])

    if not os.path.exists(gate_json):
        log("synthesizing...")
        stat = run_synth(rtl_path, cfg["top"], gate_json, synth_log)
        log(f"synth done: {stat}")
    else:
        log("gate-level netlist already present, skipping re-synthesis")

    log("loading netlist...")
    t0 = time.time()
    nl = Netlist(gate_json, top=cfg["top"])
    log(f"netlist loaded in {time.time()-t0:.1f}s: "
        f"{len(nl.dff_cells)} DFFs, {len(nl.comb_cells)} comb cells")

    depth = nl.pipeline_depth()
    log(f"pipeline depth = {depth} cycles")

    from run_synth import parse_stat
    gate_stat = parse_stat(synth_log)
    area_gates = gate_stat.get("_total_cells", len(nl.dff_cells) + len(nl.comb_cells))
    log(f"area = {area_gates} gates (generic techmap cell count): {gate_stat}")

    log("running functional-equivalence check...")
    if bench_name == "aes":
        func_check = functional_check_aes(nl)
    else:
        func_check = functional_check_present(nl)
    log(f"functional check: {func_check}")

    extra_const = None
    if cfg["extra_const_ports_fn"]:
        extra_const = globals()[cfg["extra_const_ports_fn"]]()

    log(f"running TVLA: n_traces={n_traces} batch={batch_size}")
    t0 = time.time()

    def progress(n_done, n_target, t_partial):
        elapsed = time.time() - t0
        rate = n_done / elapsed if elapsed > 0 else 0
        eta_min = (n_target - n_done) / rate / 60 if rate > 0 else float("nan")
        log(f"progress: {n_done}/{n_target} traces/class "
            f"({100*n_done/n_target:.1f}%), {rate:.0f} traces/sec, "
            f"eta {eta_min:.1f} min, running |t|max={t_partial:.3f}")

    final_res = run_tvla(nl, cfg["port_groups"], n_traces=n_traces, batch_size=batch_size,
                          seed=12345, extra_const_ports=extra_const, progress_cb=progress)
    n_done = final_res["n_traces_per_class"]
    log(f"TVLA total time: {time.time()-t0:.1f}s for {n_done} traces/class, "
        f"final tvla_t_value={final_res['tvla_t_value']:.4f}")

    # Power model: mean_hd_per_cycle = average number of register bits toggling
    # per cycle (both TVLA classes pooled). Dynamic energy per cycle = HD *
    # assumed energy-per-toggle; average power = energy/cycle * clock freq.
    energy_per_cycle_j = final_res["mean_hd_per_cycle"] * ENERGY_PER_TOGGLE_FJ * 1e-15
    power_w = energy_per_cycle_j * (CLOCK_FREQ_MHZ * 1e6)
    power_uw = power_w * 1e6
    log(f"power = {power_uw:.3f} uW (mean_hd_per_cycle={final_res['mean_hd_per_cycle']:.2f}, "
        f"assumed {ENERGY_PER_TOGGLE_FJ} fJ/toggle @ {CLOCK_FREQ_MHZ} MHz)")

    out = {
        "benchmark": bench_name,
        "label": cfg["label"],
        "pipeline_depth_cycles": depth,
        "num_dff": len(nl.dff_cells),
        "num_comb_cells": len(nl.comb_cells),
        "area_gates": area_gates,
        "gate_type_breakdown": gate_stat,
        "power_uw": power_uw,
        "functional_check": func_check,
        "tvla": {
            "t_curve": final_res["tvla_t_curve"],
            "tvla_t_value": final_res["tvla_t_value"],
            "n_traces_per_class": n_done,
            "n_samples": final_res["n_samples"],
            "mean_hd_per_cycle": final_res["mean_hd_per_cycle"],
        },
        "clock_freq_mhz_assumed": CLOCK_FREQ_MHZ,
        "energy_per_toggle_fj_assumed": ENERGY_PER_TOGGLE_FJ,
    }
    out_path = os.path.join(ROOT, f"code/results_{bench_name}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {out_path}")
    log("DONE")


if __name__ == "__main__":
    main()
