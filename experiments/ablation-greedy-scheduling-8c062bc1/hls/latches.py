"""
Active glitch-stopping latch insertion pass.

Applied IDENTICALLY (same rule, same code) after either scheduler -- this
ablation only disables the ILP schedule, not the latch mechanism itself
(see TASK.yaml context.method ablations list: "1. Disabling active
glitch-stopping latches" is a DIFFERENT, sibling ablation; this one keeps
latches and only swaps the scheduler).

A "recombination point" is a node where both masking shares (A and B)
first become jointly present in the taint (see hls/dfg.py: taints()).
Every recombination point -- including the intended final unmasking, not
only accidental ones -- is a glitch hazard if its two operands are
issued in different schedule cycles: a transient glitch can propagate the
early-arriving share's value (or a function of it) through the gate
before the late operand is stable, which is first-order exploitable.

For every recombination point whose two operands are scheduled at
different cycles, we insert a register-free, dynamically-controlled
active latch on the earlier-arriving operand that opens when its value
becomes valid and closes (holds) until the later operand's cycle, so the
consumer only ever samples both operands synchronized. If both operands
already land on the same cycle (the scheduler happened to balance them),
no latch is needed -- this is what makes the latches "active"/dynamic
rather than static registers on every edge.
"""
from dataclasses import dataclass

from .dfg import taints


def recombination_points(bench):
    t = taints(bench)
    pts = []
    for nid in bench.order:
        node = bench.nodes[nid]
        if node.op == "IN":
            continue
        if not ({"A", "B"} <= t[nid]):
            continue
        if any({"A", "B"} <= t[i] for i in node.inputs):
            continue  # already mixed upstream; not a *first* recombination
        pts.append(nid)
    return pts


@dataclass
class LatchSpec:
    consumer: str      # node id whose operand is held
    held_input: str    # node id of the operand being latched
    other_input: str   # the sibling operand it must synchronize with
    open_cycle: int    # cycle the held operand becomes valid
    close_cycle: int   # cycle it must hold until (== other operand's cycle)
    width: int


def insert_latches(bench, cycle_of):
    specs = []
    for nid in recombination_points(bench):
        node = bench.nodes[nid]
        if len(node.inputs) != 2:
            continue
        a, b = node.inputs
        ca, cb = cycle_of[a], cycle_of[b]
        if ca == cb:
            continue
        early, late = (a, ca) if ca < cb else (b, cb)
        other = b if early == a else a
        specs.append(LatchSpec(
            consumer=nid,
            held_input=early,
            other_input=other,
            open_cycle=cycle_of[early],
            close_cycle=cycle_of[other],
            width=node.width,
        ))
    return specs
