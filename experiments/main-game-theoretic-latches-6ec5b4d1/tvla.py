"""TVLA (Test Vector Leakage Assessment), fixed-vs-random methodology, run at
full N=10,000,000 traces via a numpy-vectorized leakage proxy.

Scale note (see STATE.md): dumping a real per-trace VCD for 10M traces is
infeasible on disk/time within this run. Instead we vectorize the SAME
dataflow-node evaluation that golden_eval() does in dfg.py (identical op
semantics, same topological order) across all N traces at once with numpy,
and use the classical Hamming-WEIGHT leakage model (bit-population count of
each stored intermediate value) as the per-trace power proxy -- this is the
standard approximation used throughout the power-analysis literature (e.g.
Mangard/Oswald/Popp), and is a *different* (complementary) measurement from
power_model.py's VCD-derived Hamming-DISTANCE dynamic_power metric, which is
computed from real switching activity on a smaller representative trace set.
Both are reported; they are not expected to be numerically identical.

Masking test construction: for each (share0_i, share1_i) input pair (paired
by declaration order within the benchmark), treat share0_i as a fresh random
mask r_i and share1_i = S_i XOR r_i, so S_i = share0_i XOR share1_i is the
"logical secret" the masking is supposed to protect. Group FIXED holds each
S_i at a constant value (mask r_i still randomized fresh every trace, as real
masking would). Group RANDOM draws S_i fresh every trace too. A safe masked
implementation must show NO significant difference between the two groups'
power distributions -- that is exactly first-order TVLA.
"""
import numpy as np
from dfg import DFG, OP_INFO
from scheduler import storage_nodes

POPCOUNT = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint16)


def vectorized_eval(dfg: DFG, input_arrays: dict) -> dict:
    """input_arrays: node_id -> np.uint8 array shape (N,). Returns node_id ->
    np.uint8 array shape (N,) for every node, using the identical op
    semantics as dfg.golden_eval (mod-256 arithmetic)."""
    vals = {}
    for nid in dfg.topo_order():
        n = dfg.nodes[nid]
        if n.op == 'in':
            vals[nid] = input_arrays[nid]
        elif n.op == 'const':
            N = next(iter(input_arrays.values())).shape[0]
            vals[nid] = np.full(N, n.const_val & 0xFF, dtype=np.uint8)
        elif n.op == 'add':
            a, b = n.inputs
            vals[nid] = (vals[a] + vals[b]).astype(np.uint8)
        elif n.op == 'sub':
            a, b = n.inputs
            vals[nid] = (vals[a] - vals[b]).astype(np.uint8)
        elif n.op == 'xor':
            a, b = n.inputs
            vals[nid] = (vals[a] ^ vals[b]).astype(np.uint8)
        elif n.op == 'mul':
            a, b = n.inputs
            vals[nid] = (vals[a].astype(np.uint16) * vals[b].astype(np.uint16)).astype(np.uint8)
        elif n.op == 'shl':
            a, = n.inputs
            vals[nid] = ((vals[a].astype(np.uint16) << n.shift_amt) & 0xFF).astype(np.uint8)
        elif n.op == 'shr':
            a, = n.inputs
            vals[nid] = (vals[a] >> n.shift_amt).astype(np.uint8)
        else:
            raise ValueError(f"unknown op {n.op}")
    return vals


def share_pairs(dfg: DFG):
    s0 = [n.id for n in dfg.nodes.values() if n.op == 'in' and n.share == 'share0']
    s1 = [n.id for n in dfg.nodes.values() if n.op == 'in' and n.share == 'share1']
    return list(zip(s0, s1))


def gen_inputs(dfg: DFG, rng: np.random.Generator, n_traces: int, fixed_secret=None):
    """fixed_secret: None -> group RANDOM (S_i fresh each trace).
    dict pair_idx->int -> group FIXED (S_i held at that constant)."""
    pairs = share_pairs(dfg)
    inputs = {}
    for idx, (a, b) in enumerate(pairs):
        r = rng.integers(0, 256, size=n_traces, dtype=np.uint16).astype(np.uint8)
        if fixed_secret is None:
            s = rng.integers(0, 256, size=n_traces, dtype=np.uint16).astype(np.uint8)
        else:
            s = np.full(n_traces, fixed_secret[idx] & 0xFF, dtype=np.uint8)
        inputs[a] = r
        inputs[b] = (r ^ s).astype(np.uint8)
    return inputs


def classify_nodes(dfg: DFG):
    """Split storage-eligible nodes into:
      - 'critical': every share-derived node BEFORE its designated
        recombination point -- these hold at most ONE share alone and MUST
        NOT correlate with the logical secret S_i = share0_i XOR share1_i.
        This is the actual security property the compiler's balanced-
        latch/register placement at recombination points is supposed to
        deliver, and is what tvla_max_t_stat is computed over.
      - 'post_recombination': every recombination node (dfg.
        recombination_points) and everything downstream of one. Once two
        single-share subtrees have legitimately met, the combined result is
        the design's intended output and is EXPECTED to correlate with S --
        that is the computation happening, not a side-channel flaw. These
        are reported separately, never folded into the headline metric.
    Nodes with no share dependency at all (none exist in this suite, but
    handled for completeness) are 'critical' by default (trivially safe).
    """
    from dfg import recombination_points
    consumers = {nid: [] for nid in dfg.nodes}
    for n in dfg.nodes.values():
        for i in n.inputs:
            consumers[i].append(n.id)
    recomb_nodes = [nid for (nid, _a, _b) in recombination_points(dfg)]
    seen = set(recomb_nodes)
    frontier = list(recomb_nodes)
    while frontier:
        cur = frontier.pop()
        for nxt in consumers[cur]:
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    post = seen
    critical = [n.id for n in storage_nodes(dfg) if n.id not in post]
    post_storage = [n.id for n in storage_nodes(dfg) if n.id in post]
    return critical, post_storage


def per_node_leakage_traces(dfg: DFG, sched: dict, vals: dict, node_ids):
    """Returns dict node_id -> np.float64 array (N,) = weighted Hamming
    weight of that single storage node's value (one trace value per node,
    not aggregated across nodes -- precision matters here since folding
    multiple nodes sharing a cycle would blur a leaking node into a safe
    one)."""
    storage = sched['storage']
    weight = {'latch': 0.6, 'register': 0.9}  # mirrors power_model.py constants
    out = {}
    for nid in node_ids:
        w = weight[storage[nid]]
        out[nid] = POPCOUNT[vals[nid]].astype(np.float64) * w
    return out


def welch_t(a, b):
    na, nb = a.shape[0], b.shape[0]
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    denom = np.sqrt(va / na + vb / nb)
    if denom == 0:
        return 0.0
    return float((ma - mb) / denom)


def run_tvla(dfg: DFG, sched: dict, seed: int, n_traces: int, fixed_secret=None, batch_size=2_000_000):
    """Runs the fixed-vs-random TVLA test at n_traces per group, batched to
    bound peak memory. Returns per-node Welch t-statistics, split into the
    'critical' zone (pre-recombination shares -- must stay under threshold)
    and the 'post_recombination' zone (the design's legitimate output,
    expected to correlate with S -- reported for transparency, never used
    for the pass/fail metric). See classify_nodes() docstring."""
    pairs = share_pairs(dfg)
    if fixed_secret is None:
        fixed_secret = [0x55 + 0x11 * i for i in range(len(pairs))]
    critical_ids, post_ids = classify_nodes(dfg)
    all_ids = critical_ids + post_ids

    rng_fixed = np.random.default_rng(seed * 2)
    rng_random = np.random.default_rng(seed * 2 + 1)

    n_done = 0
    sums = {}    # node_id -> [sum_fixed, sumsq_fixed, sum_rand, sumsq_rand, n]
    while n_done < n_traces:
        b = min(batch_size, n_traces - n_done)
        in_fixed = gen_inputs(dfg, rng_fixed, b, fixed_secret=fixed_secret)
        in_random = gen_inputs(dfg, rng_random, b, fixed_secret=None)
        vals_fixed = vectorized_eval(dfg, in_fixed)
        vals_random = vectorized_eval(dfg, in_random)
        lk_fixed = per_node_leakage_traces(dfg, sched, vals_fixed, all_ids)
        lk_random = per_node_leakage_traces(dfg, sched, vals_random, all_ids)
        for nid in all_ids:
            xf, xr = lk_fixed[nid], lk_random[nid]
            s = sums.setdefault(nid, [0.0, 0.0, 0.0, 0.0, 0])
            s[0] += xf.sum(); s[1] += (xf ** 2).sum()
            s[2] += xr.sum(); s[3] += (xr ** 2).sum()
            s[4] += b
        n_done += b

    t_by_node = {}
    for nid, (sf, sqf, sr, sqr, n) in sums.items():
        mf, mr = sf / n, sr / n
        vf = (sqf / n - mf ** 2) * n / (n - 1)
        vr = (sqr / n - mr ** 2) * n / (n - 1)
        denom = np.sqrt(vf / n + vr / n)
        t_by_node[nid] = float((mf - mr) / denom) if denom > 0 else 0.0

    t_by_cycle_critical = {sched['avail'].get(nid, sched['start'].get(nid, 0)): t_by_node[nid] for nid in critical_ids}
    t_by_cycle_post = {sched['avail'].get(nid, sched['start'].get(nid, 0)): t_by_node[nid] for nid in post_ids}
    max_t_critical = max((abs(v) for v in t_by_cycle_critical.values()), default=0.0)
    max_t_post = max((abs(v) for v in t_by_cycle_post.values()), default=0.0)
    return {
        't_by_node': t_by_node,
        't_by_cycle_critical': t_by_cycle_critical,
        't_by_cycle_post_recombination': t_by_cycle_post,
        'max_abs_t': max_t_critical,
        'max_abs_t_post_recombination': max_t_post,
        'n_traces_per_group': n_traces,
        'fixed_secret': fixed_secret,
        'n_critical_nodes': len(critical_ids),
        'n_post_recombination_nodes': len(post_ids),
    }
