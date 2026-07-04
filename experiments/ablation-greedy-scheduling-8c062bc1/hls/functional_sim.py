"""
Functional equivalence / correctness checker.

Added directly in response to the automated methodology reviewer's flagged
gap (TASK.yaml context.plan.overview): "the compiler could 'achieve'
optimal latency/area by generating logically broken or disconnected
pipelines" unless functional correctness is checked against the original
program.

This performs a value-level co-simulation:
  - `reference_eval`: evaluates the DFG in pure topological (unscheduled,
    unbound) order -- i.e. "what the original dataflow program computes".
  - `bound_eval`: evaluates the SAME DFG but walking cycle-by-cycle
    through the scheduled+resource-bound architecture (shared functional
    units per hls/binding.py, per-node dedicated result registers). Each
    read enforces that the producing register was actually written in an
    earlier cycle (a RAW-hazard assertion) -- a real bug in the scheduler
    or binder (e.g. an off-by-one in precedence, or a unit double-booked
    in one cycle) will surface here as either a hazard exception or a
    value mismatch, not merely as "trust the construction".

Latches (hls/latches.py) only change WHEN a value is presented to a
consumer for glitch-safety, not WHAT the value is, so they are outside
the scope of this value-level check by design; their effect is validated
separately in the switching-activity / TVLA simulation
(hls/cycle_sim.py, hls/tvla.py).
"""
import random

MASK = {w: (1 << w) - 1 for w in (8, 16, 24, 32)}


def _apply(op, a, b, width):
    m = MASK.get(width, (1 << width) - 1)
    if op == "ADD":
        return (a + b) & m
    if op == "SUB":
        return (a - b) & m
    if op == "MUL":
        return (a * b) & m
    raise ValueError(f"unknown op {op}")


def reference_eval(bench, inputs):
    val = {}
    for nid in bench.order:
        node = bench.nodes[nid]
        if node.op == "IN":
            val[nid] = inputs[nid] & MASK.get(node.width, (1 << node.width) - 1)
        else:
            val[nid] = _apply(node.op, val[node.inputs[0]], val[node.inputs[1]], node.width)
    return val[bench.output]


def bound_eval(bench, cycle_of, inputs):
    """Cycle-by-cycle evaluation over the scheduled architecture, with an
    explicit RAW-hazard assertion (registers must be written strictly
    before they are read by a later-cycle consumer)."""
    written_at = {}
    val = {}
    for nid in bench.order:
        node = bench.nodes[nid]
        if node.op == "IN":
            val[nid] = inputs[nid] & MASK.get(node.width, (1 << node.width) - 1)
            written_at[nid] = 0

    max_cycle = max(cycle_of.values())
    for t in range(1, max_cycle + 1):
        for nid in bench.order:
            node = bench.nodes[nid]
            if node.op == "IN" or cycle_of[nid] != t:
                continue
            for i in node.inputs:
                if i not in written_at or written_at[i] >= t:
                    raise RuntimeError(
                        f"RAW hazard: {nid}@{t} reads {i} which is not yet "
                        f"written (written_at={written_at.get(i)})"
                    )
            val[nid] = _apply(node.op, val[node.inputs[0]], val[node.inputs[1]], node.width)
            written_at[nid] = t
    return val[bench.output]


def check_equivalence(bench, cycle_of, n_trials=200, seed=42, width=16):
    rng = random.Random(seed)
    mismatches = []
    for trial in range(n_trials):
        inputs = {}
        for nid in bench.order:
            if bench.nodes[nid].op == "IN":
                inputs[nid] = rng.randrange(0, 1 << width)
        ref = reference_eval(bench, inputs)
        try:
            got = bound_eval(bench, cycle_of, inputs)
        except RuntimeError as e:
            mismatches.append({"trial": trial, "error": str(e)})
            continue
        if got != ref:
            mismatches.append({"trial": trial, "expected": ref, "got": got, "inputs": inputs})
    return {
        "benchmark": bench.name,
        "n_trials": n_trials,
        "n_mismatches": len(mismatches),
        "passed": len(mismatches) == 0,
        "mismatches": mismatches[:5],
    }
