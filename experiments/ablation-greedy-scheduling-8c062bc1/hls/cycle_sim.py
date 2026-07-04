"""
Cycle-accurate simulator + VCD (Value Change Dump) writer.

No Verilator is available in this sandbox (no root / apt access), so
this module IS the simulator: it walks a single representative,
deterministic random trial through the scheduled+bound+latched
architecture (same semantics as hls/functional_sim.py) and emits a
standard-format .vcd trace of every node register, every active latch,
and the clock, directly from that walk -- so the archived VCD is
consistent with the numeric power/TVLA model by construction (both are
driven from the same schedule/binding/latch data), not a separately
faked artifact.
"""
import random

from .binding import bind_units
from .latches import insert_latches


def _apply(op, a, b, width):
    m = (1 << width) - 1
    if op == "ADD":
        return (a + b) & m
    if op == "SUB":
        return (a - b) & m
    if op == "MUL":
        return (a * b) & m
    raise ValueError(op)


def _vcd_id(i):
    # short printable VCD identifiers: !, ", #, ...
    return chr(33 + i)


def write_vcd(path, bench, cycle_of, latch_specs, width=16, seed=99):
    rng = random.Random(seed)
    val = {}
    inputs = {}
    for nid in bench.order:
        if bench.nodes[nid].op == "IN":
            inputs[nid] = rng.randrange(0, 1 << width)
            val[nid] = inputs[nid]

    signals = ["clk"] + [nid for nid in bench.order] + [f"latch_{i}" for i in range(len(latch_specs))]
    vcd_id = {s: _vcd_id(i) for i, s in enumerate(signals)}

    lines = [
        "$timescale 1ns $end",
        "$scope module top $end",
    ]
    for s in signals:
        w = width if s != "clk" else 1
        lines.append(f"$var wire {w} {vcd_id[s]} {s} $end")
    lines.append("$upscope $end")
    lines.append("$enddefinitions $end")

    max_cycle = max(cycle_of.values())
    latched_value = {i: None for i in range(len(latch_specs))}

    def fmt(sig, value, width_):
        if width_ == 1:
            return f"{value}{vcd_id[sig]}"
        return f"b{value:0{width_}b} {vcd_id[sig]}"

    events = []  # (time, [fmt lines])

    t0 = []
    for nid in bench.order:
        if bench.nodes[nid].op == "IN":
            t0.append(fmt(nid, val[nid], width))
    t0.append(fmt("clk", 0, 1))
    events.append((0, t0))

    for cycle in range(1, max_cycle + 1):
        time = cycle * 10
        changes = [fmt("clk", 1, 1)]
        for nid in bench.order:
            node = bench.nodes[nid]
            if node.op == "IN" or cycle_of[nid] != cycle:
                continue
            a_val = val[node.inputs[0]]
            b_val = val[node.inputs[1]]
            val[nid] = _apply(node.op, a_val, b_val, width)
            changes.append(fmt(nid, val[nid], width))
        for i, ls in enumerate(latch_specs):
            if ls.open_cycle == cycle:
                latched_value[i] = val[ls.held_input]
                changes.append(fmt(f"latch_{i}", latched_value[i], width))
        events.append((time, changes))
        events.append((time + 5, [fmt("clk", 0, 1)]))

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
        f.write("$dumpvars\n")
        for c in events[0][1]:
            f.write(c + "\n")
        f.write("$end\n")
        for time, changes in events[1:]:
            f.write(f"#{time}\n")
            for c in changes:
                f.write(c + "\n")

    return val[bench.output]
