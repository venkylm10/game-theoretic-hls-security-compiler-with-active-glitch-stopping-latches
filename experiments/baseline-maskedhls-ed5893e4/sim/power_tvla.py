"""
Hamming-distance dynamic-power proxy + streaming fixed-vs-random TVLA
(Welch's t-test), run directly on the yosys-synthesized gate-level netlist
via netlist_sim.Netlist.

Power model: at each clock cycle we know the full register-bank state (every
$_DFF_P_ output bit) before and after the cycle. Dynamic (switching) power in
CMOS is dominated by C*V^2*f*activity_factor, so the number of register bits
that toggle in a cycle (their Hamming distance from the previous cycle) is
used as the activity-factor proxy per cycle, matching "custom Python-based
power models tracking Hamming weight/distance transitions" from the
experiment's method description. This is a REGISTER-level (not full
transistor/SPICE-level) leakage model: it does not include intra-cycle
combinational glitches, since the simulator is a zero-delay logical
simulator, not a delay-annotated one (no SDF, no PDK is available on this
pod). See report.md for the resulting limitation.

TVLA: classic fixed-vs-random methodology (Goodwill et al.). Dataset "fixed"
drives the same plaintext every trace with fresh random shares+masks;
dataset "random" drives a fresh random plaintext every trace with fresh
random shares+masks. A per-time-sample (per-cycle) Welch's t-test is run
between the two classes; tvla_t_value = max |t| over all time samples.
Running (n, mean, M2) Welch accumulators are used so millions of traces can
be processed in fixed memory instead of buffering every trace.
"""
import numpy as np


class WelchAccumulator:
    """Streaming per-sample mean/variance (Welford's algorithm), batched."""

    def __init__(self, n_samples):
        self.n = 0
        self.mean = np.zeros(n_samples, dtype=np.float64)
        self.M2 = np.zeros(n_samples, dtype=np.float64)

    def update_batch(self, batch_values):
        """batch_values: shape (batch, n_samples)."""
        b = batch_values.shape[0]
        batch_mean = batch_values.mean(axis=0)
        batch_var = batch_values.var(axis=0)
        new_n = self.n + b
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (b / new_n)
        self.M2 = self.M2 + batch_var * b + delta ** 2 * self.n * b / new_n
        self.n = new_n

    def variance(self):
        return self.M2 / max(self.n - 1, 1)


def welch_t(acc_a, acc_b):
    va, vb = acc_a.variance(), acc_b.variance()
    na, nb = acc_a.n, acc_b.n
    denom = np.sqrt(va / na + vb / nb)
    denom = np.where(denom == 0, np.nan, denom)
    t = (acc_a.mean - acc_b.mean) / denom
    return np.nan_to_num(t, nan=0.0)


def bits_n(vec, n):
    return [((vec >> i) & 1).astype(bool) for i in range(n)]


def run_tvla(nl, port_groups, n_traces, batch_size, seed, extra_const_ports=None,
             progress_cb=None):
    """
    port_groups: dict describing how to derive per-trace stimulus:
        {
          'share_ports': [(port0, port1), ...] one pair per plaintext bit/byte
                          group, each a tuple of (share0_port, share1_port)
          'plaintext_widths': [width_of_each_group, ...] (bits)
          'mask_ports': [port_name, ...] fresh-random-every-trace ports (r*)
        }
    extra_const_ports: {port_name: constant_int_value} (e.g. dec_* for AES)

    Returns: dict with tvla_t_curve (per-sample t stats), tvla_t_value (max|t|),
             mean_hd_per_cycle (avg register Hamming distance, both classes
             pooled), n_traces_run, per-class WelchAccumulators.
    """
    depth = nl.pipeline_depth()
    n_samples = depth + 1
    rng = np.random.default_rng(seed)

    acc_fixed = WelchAccumulator(n_samples)
    acc_random = WelchAccumulator(n_samples)
    hd_sum = 0.0
    hd_count = 0

    dff_cnames = [cname for cname, _ in nl.dff_cells]

    # Fixed-class plaintext: drawn once, held constant for every "fixed" trace.
    fixed_plaintexts = {}
    for gi, width in enumerate(port_groups["plaintext_widths"]):
        fixed_plaintexts[gi] = int(rng.integers(0, 2 ** width))

    n_done = 0
    while n_done < n_traces:
        b = min(batch_size, n_traces - n_done)
        for cls, acc in (("fixed", acc_fixed), ("random", acc_random)):
            pi_bits = {}
            for gi, (p0, p1) in enumerate(port_groups["share_ports"]):
                width = port_groups["plaintext_widths"][gi]
                if cls == "fixed":
                    pt = np.full(b, fixed_plaintexts[gi], dtype=np.uint32)
                else:
                    pt = rng.integers(0, 2 ** width, size=b).astype(np.uint32)
                share0 = rng.integers(0, 2 ** width, size=b).astype(np.uint32)
                share1 = pt ^ share0
                pi_bits[p0] = bits_n(share0, width)
                pi_bits[p1] = bits_n(share1, width)
            for mp in port_groups["mask_ports"]:
                width = len(nl.ports[mp]["bits"])
                pi_bits[mp] = bits_n(rng.integers(0, 2 ** width, size=b).astype(np.uint32), width)
            for cp, val in (extra_const_ports or {}).items():
                width = len(nl.ports[cp]["bits"])
                pi_bits[cp] = bits_n(np.full(b, val, dtype=np.uint32), width)

            state = nl.new_state(b)
            prev_state_bits = np.zeros((b, len(dff_cnames)), dtype=bool)
            hd_samples = np.zeros((b, n_samples), dtype=np.float64)
            for cyc in range(n_samples):
                frame = {"clk": np.ones(b, dtype=bool)}
                frame.update(pi_bits)
                _, state, _ = nl.step(frame, state, b)
                cur_state_bits = np.stack([state[c] for c in dff_cnames], axis=1)
                hd = (cur_state_bits != prev_state_bits).sum(axis=1).astype(np.float64)
                hd_samples[:, cyc] = hd
                prev_state_bits = cur_state_bits

            acc.update_batch(hd_samples)
            hd_sum += hd_samples.sum()
            hd_count += hd_samples.size
        n_done += b
        if progress_cb is not None:
            t_curve_partial = welch_t(acc_fixed, acc_random)
            progress_cb(n_done, n_traces, float(np.max(np.abs(t_curve_partial))))

    t_curve = welch_t(acc_fixed, acc_random)
    return {
        "tvla_t_curve": t_curve.tolist(),
        "tvla_t_value": float(np.max(np.abs(t_curve))),
        "mean_hd_per_cycle": hd_sum / hd_count if hd_count else 0.0,
        "n_traces_per_class": n_done,
        "n_samples": n_samples,
        "acc_fixed_mean": acc_fixed.mean.tolist(),
        "acc_random_mean": acc_random.mean.tolist(),
    }
