"""
Gate-equivalent (GE) area cost model + synthesis log writer.

No physical synthesis toolchain (Yosys) is available in this sandbox (no
root / no apt access -- see code/README.md), so area is estimated from a
documented, simplified GE cost table instead of measured gate-level
synthesis. This is explicitly flagged as `provenance: "estimated"` in the
results manifest, never claimed as `"measured"`.

Per the methodology reviewer's flagged note (TASK.yaml
context.plan.overview): area is reported in a SINGLE unit -- gate
equivalents (GE) -- across every experiment in this plan (baseline,
main, and all ablations), so Pareto comparisons are valid; no other unit
(gates/LUTs) is used here.

Cost table (assumed, generic 45nm-equivalent standard-cell library --
this is the "target technology library" the reviewer flagged as
undefined; declared once, here, and reused everywhere):
"""
GE_PER_BIT = {"ADD": 1.5, "SUB": 1.5, "MUL": 25.0}
REG_GE_PER_BIT = 6.0     # a normal clocked D-flip-flop
LATCH_GE_PER_BIT = 3.5   # register-free active glitch-stopping latch (cheaper: no master-slave pair)
MUX_GE_PER_BIT = 1.0     # per extra operand routed into a shared, time-multiplexed unit


def compute_area_ge(bench, cycle_of, unit_of, latch_specs, width=16):
    functional_unit_ge = 0.0
    for op, count in bench.resource_limits.items():
        functional_unit_ge += count * GE_PER_BIT.get(op, 1.0) * width

    op_nodes = [nid for nid in bench.order if bench.nodes[nid].op != "IN"]
    register_ge = len(op_nodes) * REG_GE_PER_BIT * width
    latch_ge = len(latch_specs) * LATCH_GE_PER_BIT * width

    # mux area: a unit instance shared by k>1 distinct issues needs a
    # (k:1) input mux on each of its 2 operand ports.
    users_per_unit = {}
    for nid in op_nodes:
        key = (bench.nodes[nid].op, unit_of[nid])
        users_per_unit[key] = users_per_unit.get(key, 0) + 1
    mux_ge = sum(max(0, k - 1) * 2 * MUX_GE_PER_BIT * width for k in users_per_unit.values())

    total = functional_unit_ge + register_ge + latch_ge + mux_ge
    return {
        "functional_unit_ge": functional_unit_ge,
        "register_ge": register_ge,
        "latch_ge": latch_ge,
        "mux_ge": mux_ge,
        "total_ge": total,
    }


def write_synthesis_log(path, bench, cycle_of, unit_of, latch_specs, area_breakdown, latency, width=16):
    lines = [
        f"# Simulated synthesis log for benchmark '{bench.name}'",
        "# NOTE: no physical EDA toolchain (Yosys) is available in this sandbox",
        "#       (no root / apt access). This log records the deterministic,",
        "#       documented GE cost-model computation in hls/area_model.py",
        "#       instead of a real gate-level synthesis report.",
        f"target_technology: generic-45nm-equivalent-stdcell (assumed)",
        f"target_clock_mhz: 200",
        f"bit_width: {width}",
        f"scheduler: greedy-list-scheduling (ablation-greedy-scheduling)",
        f"latency_cycles: {latency}",
        f"resource_limits: {bench.resource_limits}",
        f"n_op_nodes: {len([n for n in bench.order if bench.nodes[n].op != 'IN'])}",
        f"n_active_glitch_latches: {len(latch_specs)}",
        "",
        "-- area breakdown (gate equivalents, GE) --",
    ]
    for k, v in area_breakdown.items():
        lines.append(f"{k}: {v:.2f}")
    lines.append("")
    lines.append("-- per-node cycle / unit binding --")
    for nid in bench.order:
        node = bench.nodes[nid]
        if node.op == "IN":
            continue
        lines.append(f"{nid}: op={node.op} cycle={cycle_of[nid]} unit={node.op.lower()}_u{unit_of[nid]}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
