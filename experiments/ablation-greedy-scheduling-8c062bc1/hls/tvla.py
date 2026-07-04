"""
Simulated Test Vector Leakage Assessment (TVLA), fixed-vs-random-secret
methodology (standard non-specific TVLA adapted from fixed-vs-random
plaintext to our masked-secret setting).

Threat model / simplifications (documented explicitly, not hidden):
  - The two masking shares (s0, s1 = s0 XOR secret) are the "sensitive"
    signals. A "recombination point" (hls/latches.py) is where both
    shares first jointly influence a node -- the classic masking
    "non-completeness" hazard: if such a node's two operands arrive at
    different schedule cycles with no synchronizing latch, physical
    glitches can transiently evaluate a function of BOTH shares before
    they are both stable, which is conservatively modeled here as full
    disclosure of the secret's Hamming weight at that cycle (the standard
    conservative argument for why non-completeness violations are
    treated as catastrophic in the masking literature).
  - If the recombination point IS synchronized (same cycle naturally, or
    an active glitch latch closes the gap -- both arms of this ablation
    keep the latch mechanism, only the scheduler differs), its physical
    switching activity is modeled as drawn from a secret-independent
    Uniform(0, 2^width) surrogate. This encodes the (external, assumed-
    given per TASK.yaml context.intuition) formal guarantee from the
    state-wise datapath tracking verification of Sarma et al. (2026) that
    a *synchronized* masked-gadget combine does not leak -- verifying
    that guarantee itself is a different research artifact, out of scope
    for this scheduling ablation (see report.md self-critique).
  - Nodes touched by only ONE share are provably secret-independent in
    THIS model regardless of the above assumption: s0's own marginal
    distribution never depends on secret by construction of the trace
    generator, so their real computed value is used directly.
  - Safety propagates forward through the mixed-domain subtree: once a
    first recombination point is judged safe (surrogate-modeled), every
    node built further on top of it is *also* surrogate-modeled (it is
    just more processing on an already-assumed-secure masked value, not
    a fresh hazard) -- only a node's OWN first crossing into the mixed
    domain is ever checked for a NEW hazard. Symmetrically, if a first
    recombination point is unprotected (the `protect=False` self-test
    only), every downstream node inherits the raw "full secret" leakage
    sample too, since real ADD/SUB/MUL of the two raw shares is not
    itself a verified masking gadget (see hls/dfg.py docstring) and
    would otherwise leak through the untreated real value.
  - The benchmark's single designated final output is excluded from the
    leakage window: it is the intended reveal of the answer, not a
    physical side-channel leak, matching standard practice of scoping
    TVLA to the internal processing window.

A `protect=False` mode is provided purely as an internal sanity check
(the "self-test" logged by train.py) to confirm the test has real
discriminative power: an intentionally UN-latched recombination hazard
must drive |t| far above the 4.5 threshold, not silently pass.
"""
import numpy as np

from .dfg import taints
from .latches import recombination_points
from .sim_core import simulate_batch, popcount


def classify_nodes(bench, cycle_of, latch_specs, protect):
    """
    Per-node leakage classification, propagated forward through the
    mixed-domain (both-shares-present) subtree -- see module docstring.
    Returns {node_id: 'real' | 'surrogate' | 'leak'}: 'real' = provably
    secret-independent (single-share or public-only), use its actual
    computed value; 'surrogate' = a protected mixed-domain node, modeled
    as secret-independent Uniform noise; 'leak' = an unprotected mixed-
    domain hazard, modeled as full disclosure of the secret's Hamming
    weight (only reachable with protect=False, the internal self-test).
    """
    t = taints(bench)
    recomb = set(recombination_points(bench))
    protected_consumers = {ls.consumer for ls in latch_specs}
    cls = {}
    for nid in bench.order:
        node = bench.nodes[nid]
        if node.op == "IN":
            continue
        mixed = {"A", "B"} <= t[nid]
        if not mixed:
            cls[nid] = "real"
            continue
        if nid in recomb:
            naturally_balanced = cycle_of[node.inputs[0]] == cycle_of[node.inputs[1]]
            is_protected = protect and (nid in protected_consumers or naturally_balanced)
            cls[nid] = "surrogate" if is_protected else "leak"
        else:
            parent_states = [cls[i] for i in node.inputs if ({"A", "B"} <= t[i])]
            cls[nid] = "leak" if "leak" in parent_states else "surrogate"
    return cls


def _leakage_traces(bench, cycle_of, latch_specs, n_traces, seed, width, secret_mode,
                     fixed_secret, protect, exclude_final):
    rng = np.random.default_rng(seed)
    values, secret = simulate_batch(bench, n_traces, rng, width=width,
                                     secret_mode=secret_mode, fixed_secret=fixed_secret)
    T = max(cycle_of.values())
    samples = np.zeros((n_traces, T + 1), dtype=np.float64)

    cls = classify_nodes(bench, cycle_of, latch_specs, protect)
    surrogate_rng = np.random.default_rng(seed * 7919 + 3)  # independent stream, deterministic

    for nid in bench.order:
        node = bench.nodes[nid]
        if node.op == "IN":
            continue
        t_cycle = cycle_of[nid]
        if exclude_final and nid == bench.output:
            continue
        state = cls[nid]
        if state == "surrogate":
            surrogate = surrogate_rng.integers(0, 1 << node.width, size=n_traces, dtype=np.int64)
            samples[:, t_cycle] += popcount(surrogate)
        elif state == "leak":
            samples[:, t_cycle] += popcount(secret & ((1 << node.width) - 1))
        else:
            samples[:, t_cycle] += popcount(values[nid])
    return samples


def welch_t(a, b):
    ma, mb = a.mean(axis=0), b.mean(axis=0)
    va, vb = a.var(axis=0, ddof=1), b.var(axis=0, ddof=1)
    na, nb = a.shape[0], b.shape[0]
    denom = np.sqrt(va / na + vb / nb)
    denom = np.where(denom == 0, 1e-12, denom)
    return (ma - mb) / denom


def run_tvla(bench, cycle_of, latch_specs, n_traces=20000, seed=11, width=16,
             fixed_secret=0, protect=True, exclude_final=True):
    fixed = _leakage_traces(bench, cycle_of, latch_specs, n_traces, seed, width,
                             "fixed", fixed_secret, protect, exclude_final)
    random_ = _leakage_traces(bench, cycle_of, latch_specs, n_traces, seed + 1, width,
                               "random", fixed_secret, protect, exclude_final)
    t_per_cycle = welch_t(fixed, random_)
    max_abs_t = float(np.nanmax(np.abs(t_per_cycle)))
    worst_cycle = int(np.nanargmax(np.abs(t_per_cycle)))
    return {
        "benchmark": bench.name,
        "n_traces_per_group": n_traces,
        "max_abs_t": max_abs_t,
        "worst_cycle": worst_cycle,
        "t_per_cycle": t_per_cycle.tolist(),
        "protect": protect,
    }
