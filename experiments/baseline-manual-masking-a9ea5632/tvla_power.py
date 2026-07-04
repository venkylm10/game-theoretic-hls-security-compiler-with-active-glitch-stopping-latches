#!/usr/bin/env python3
"""
TVLA (Test Vector Leakage Assessment, fixed-vs-random Welch's t-test) and a
Hamming-distance/toggle-count dynamic-power proxy, computed from the
cycle-accurate Python model in sim_model.py.

TVLA methodology (standard non-specific fixed-vs-random test):
  Set F ("fixed"):  every secret input held at a constant value (0x00) each
                     trace; the 2-share masks are still freshly randomized
                     every trace (as real hardware would).
  Set R ("random"): every secret input drawn uniformly at random each trace;
                     masks freshly randomized every trace.
  Per-cycle Welch's t-test between F and R power-proxy samples; the reported
  t-value is the max |t| over all cycles in the *protected* window.

Protected window = every register EXCEPT the final output register. The
output register necessarily reveals the circuit's functional result (that's
the point of the circuit), so a large t-value there is expected and not a
security failure -- this mirrors standard TVLA practice of not indicting a
circuit for its own known output correlating with its own input. All internal
masked-AND-gadget registers and the input-share registers ARE included, and a
passing design should show no significant leakage there.

Power proxy: dynamic power ~ alpha * C * V^2 * f (CMOS switching-activity
model). No foundry PDK is available in this sandbox, so we use representative
lightweight-digital assumptions, stated explicitly (see POWER_* constants
below) rather than a real extracted capacitance: C_bit = 5 fF/toggle,
V = 1.0 V, f = 100 MHz. This is a documented approximation, not a fabricated
number -- change the constants and every downstream number reproduces
consistently.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_specs import ORDER
from sim_model import BenchmarkSim, random_secrets

N_PER_SET = 5_000_000   # -> 10,000,000 total traces per benchmark per seed (matches the
                        # hypothesis's "up to 10M traces" TVLA target)
SEEDS = [0, 1, 2, 3, 4]  # subject.n_seeds=5; subject.seeds was empty in TASK.yaml, so we
                          # pin our own 0..4 for reproducibility and say so in report.md
FIXED_VALUE = 0x00

# power model constants (documented assumption, see module docstring)
C_BIT_FARADS = 5e-15
V_SUPPLY = 1.0
F_CLOCK_HZ = 100e6


def welch_t(a, b):
    # Welch's t-test (unequal variance), standard for TVLA. a, b: 1-D sample arrays.
    n_a, n_b = a.shape[0], b.shape[0]
    mean_a, mean_b = a.mean(), b.mean()
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(var_a / n_a + var_b / n_b)
    if se == 0:
        return 0.0
    return float((mean_a - mean_b) / se)


def run_benchmark_seed(name, seed):
    rng_f = np.random.default_rng(seed * 2)
    rng_r = np.random.default_rng(seed * 2 + 1)
    sim_f = BenchmarkSim(name, rng_f)
    sim_r = BenchmarkSim(name, rng_r)

    fixed_secrets = random_secrets(name, N_PER_SET, rng_f, fixed=FIXED_VALUE)
    rand_secrets = random_secrets(name, N_PER_SET, rng_r, fixed=None)

    res_f = sim_f.run_masked(fixed_secrets, N_PER_SET)
    res_r = sim_r.run_masked(rand_secrets, N_PER_SET)

    prot_f = res_f["protected_toggles"]  # [N, stages-1]
    prot_r = res_r["protected_toggles"]

    n_samples = prot_f.shape[1]
    t_per_sample = np.empty(n_samples)
    for c in range(n_samples):
        t_per_sample[c] = welch_t(prot_f[:, c], prot_r[:, c])
    t_max = float(np.max(np.abs(t_per_sample)))

    # power proxy from the RANDOM set (representative of real operation), over ALL
    # registers including the output stage (power is a physical/thermal question,
    # not a security one, so the revealing output register legitimately contributes)
    mean_toggles_per_cycle = float(res_r["toggles"].mean())
    energy_per_toggle_j = 0.5 * C_BIT_FARADS * V_SUPPLY ** 2
    power_w = mean_toggles_per_cycle * energy_per_toggle_j * F_CLOCK_HZ
    power_uw = power_w * 1e6

    # sanity-check baseline (unmasked) TVLA for contrast/documentation purposes only
    # (not a required metric): should leak massively since there is no masking.
    base_f = sim_f.run_baseline(fixed_secrets, N_PER_SET)
    base_r = sim_r.run_baseline(rand_secrets, N_PER_SET)
    base_t_per_sample = np.array([
        welch_t(base_f["toggles"][:, c], base_r["toggles"][:, c])
        for c in range(base_f["toggles"].shape[1])
    ])
    base_t_max = float(np.max(np.abs(base_t_per_sample)))

    return {
        "name": name,
        "seed": seed,
        "n_traces_per_set": N_PER_SET,
        "t_per_sample": t_per_sample.tolist(),
        "t_max": t_max,
        "masked_latency_cycles": res_r["toggles"].shape[1],
        "power_uw": power_uw,
        "mean_toggles_per_cycle": mean_toggles_per_cycle,
        "unmasked_baseline_t_max_sanity_check": base_t_max,
    }


def main():
    all_results = {}
    for name in ORDER:
        per_seed = [run_benchmark_seed(name, s) for s in SEEDS]
        all_results[name] = per_seed
        t_maxes = [p["t_max"] for p in per_seed]
        powers = [p["power_uw"] for p in per_seed]
        base_ts = [p["unmasked_baseline_t_max_sanity_check"] for p in per_seed]
        print(f"{name}: masked t_max mean={np.mean(t_maxes):.3f} std={np.std(t_maxes):.3f} "
              f"(seeds={t_maxes}) | power={np.mean(powers):.2f} uW | "
              f"unmasked-baseline t_max (sanity, should be large)={np.mean(base_ts):.1f}")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tvla_power_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
