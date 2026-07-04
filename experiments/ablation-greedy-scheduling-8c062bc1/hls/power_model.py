"""
Custom Python-based power model tracking Hamming-weight switching activity
over the (simulated) register-write events of the scheduled datapath --
this stands in for "Hamming weight/distance transitions over VCD traces
generated from Verilator simulations" (TASK.yaml context.method): with no
physical simulator available in this sandbox, hls/cycle_sim.py generates
the VCD trace directly from this same simulated model, so the numbers and
the archived VCD are consistent by construction.

Since every result register in this architecture is written exactly once
(single-static-assignment; see hls/functional_sim.py's RAW-hazard check),
the dynamic switching energy at a write event equals the Hamming WEIGHT of
the newly written value (transition from the reset value 0). This is a
standard simplified proxy used in early HLS power/design-space-exploration
tools that lack a gate-level netlist.

Technology parameters below are ASSUMED (no measured silicon data exists
for a sandboxed cost-model estimate), matching the reviewer's flagged
"top-level parameter regime is completely empty" note -- declared once,
here, and reused by every benchmark/scheduler so comparisons are
apples-to-apples.
"""
import numpy as np

from .sim_core import simulate_batch, popcount

CLOCK_FREQ_HZ = 200e6           # 200 MHz target clock (assumed)
ENERGY_PER_BIT_TOGGLE_FJ = 0.5   # dynamic energy per bit toggle, generic 45nm-equivalent (assumed)
LEAKAGE_W_PER_GE = 50e-12        # static leakage power per gate-equivalent (assumed)


def estimate_power_mw(bench, cycle_of, area_ge, n_traces=20000, seed=7, width=16):
    rng = np.random.default_rng(seed)
    values, _secret = simulate_batch(bench, n_traces, rng, width=width, secret_mode="random")

    T = max(cycle_of.values())
    bits_per_cycle = np.zeros(T + 1, dtype=np.float64)  # mean over traces
    for nid in bench.order:
        node = bench.nodes[nid]
        if node.op == "IN":
            continue
        t = cycle_of[nid]
        bits_per_cycle[t] += popcount(values[nid]).mean()

    avg_bits_per_cycle = bits_per_cycle.sum() / T  # amortized over the whole schedule
    e_dyn_per_cycle_j = avg_bits_per_cycle * ENERGY_PER_BIT_TOGGLE_FJ * 1e-15
    p_dyn_w = e_dyn_per_cycle_j * CLOCK_FREQ_HZ
    p_static_w = area_ge * LEAKAGE_W_PER_GE
    p_total_mw = (p_dyn_w + p_static_w) * 1e3
    return {
        "power_mw": p_total_mw,
        "p_dyn_mw": p_dyn_w * 1e3,
        "p_static_mw": p_static_w * 1e3,
        "avg_bits_per_cycle": float(avg_bits_per_cycle),
        "clock_mhz": CLOCK_FREQ_HZ / 1e6,
    }
