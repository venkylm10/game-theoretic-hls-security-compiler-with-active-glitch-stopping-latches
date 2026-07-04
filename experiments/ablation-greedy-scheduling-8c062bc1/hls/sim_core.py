"""
Vectorized (numpy) batch functional simulator: computes every DAG node's
register value for a whole batch of `n_traces` independent random input
vectors at once. This underlies both the power model (hls/power_model.py)
and the TVLA leakage test (hls/tvla.py), so both draw on the exact same
functional semantics as hls/functional_sim.py's scalar reference (ADD/SUB/
MUL truncated to `width` bits), just batched for throughput -- 10M-trace
-scale TVLA campaigns (per TASK.yaml context.hypothesis) are only tractable
in a CPU-only sandbox if vectorized.

Masking convention: the sensitive value ("secret") is Boolean-shared as
s0 (uniform random) and s1 = s0 XOR secret. `secret_mode='fixed'` pins
the secret to `fixed_secret` for every trace (still re-randomizing s0/s1
per trace) -- the "fixed" group of a fixed-vs-random TVLA test; `'random'`
draws a fresh uniform secret per trace -- the "random" group.
"""
import numpy as np


def popcount(arr):
    return np.bitwise_count(arr.astype(np.uint64)).astype(np.float64)


def simulate_batch(bench, n_traces, rng, width=16, secret_mode="random", fixed_secret=0):
    mask = (1 << width) - 1
    s0 = rng.integers(0, 1 << width, size=n_traces, dtype=np.int64)
    if secret_mode == "fixed":
        secret = np.full(n_traces, fixed_secret, dtype=np.int64)
    else:
        secret = rng.integers(0, 1 << width, size=n_traces, dtype=np.int64)
    s1 = (s0 ^ secret) & mask

    pub = {pid: rng.integers(0, 1 << width, size=n_traces, dtype=np.int64)
           for pid in bench.primary_inputs}

    values = {}
    for nid in bench.order:
        node = bench.nodes[nid]
        if node.op == "IN":
            if nid == bench.share_inputs[0]:
                values[nid] = s0
            elif nid == bench.share_inputs[1]:
                values[nid] = s1
            else:
                values[nid] = pub[nid]
            continue
        a = values[node.inputs[0]]
        b = values[node.inputs[1]]
        if node.op == "ADD":
            v = (a + b) & mask
        elif node.op == "SUB":
            v = (a - b) & mask
        elif node.op == "MUL":
            v = (a * b) & mask
        else:
            raise ValueError(node.op)
        values[nid] = v
    return values, secret
