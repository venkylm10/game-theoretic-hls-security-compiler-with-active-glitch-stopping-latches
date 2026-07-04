"""Game-theoretic multi-objective ILP scheduler.

Formulation (time-indexed HLS resource-constrained scheduling + storage-
element selection):

  - Every functional-unit (FU) node (add/sub/xor/mul) gets a binary
    start-cycle indicator x[n][t]; exactly one t is chosen. Every op is
    separated from its consumers by >=1 clock cycle (no raw combinational
    chaining across dataflow nodes) -- this eliminates cross-op glitch
    propagation by construction. shl/shr are pure wires (zero latency, zero
    cost) and inherit their operand's availability; const is a tie-off.
  - Resource counts k_r (per FU class) are THEMSELVES decision variables,
    capped at MAX_UNITS -- fewer units costs less area but can force
    serialization (more latency), which is what makes the latency/area/
    power objective genuinely multi-dimensional rather than a fixed-
    resource single-objective schedule.
  - Every storage-eligible node (primary inputs 'in', and every FU node's
    result) needs exactly one storage element to hold its value from the
    cycle it becomes valid onward (values are single-assignment, so once
    written a storage element simply holds -- no extra hardware is needed
    regardless of how many idle cycles a downstream consumer waits, whether
    that element is an edge-triggered flip-flop or a level-sensitive latch).
    The ILP's free per-node binary `use_latch_n` chooses between:
      - register: an edge-triggered D flip-flop (heavier, always glitch-safe)
      - active glitch-stopping latch: a level-sensitive D-latch whose enable
        is dynamically gated by the FSM to be transparent ONLY during that
        node's own designated cycle and opaque otherwise -- so it is exactly
        as glitch-safe as a flip-flop (both only let the input through
        during their own write window) but a level-sensitive latch is the
        physically lighter storage primitive (this is the paper's actual
        mechanism: swap flops for actively-gated latches wherever the ILP
        can, since both are equally secure and the latch strictly dominates
        on area+power).
    "Unbalanced paths" (e.g. iir2's feedback vs. its fresh input tap, crc8's
    growing state vs. each fresh bit) are exactly the edges with slack>0
    that this per-node storage element must silently absorb by holding --
    no separate mechanism is needed because holding is free once you have
    *any* correctly-gated storage element, which is the point.

The objective's "area"/"power" terms are a synthesis-time PROXY cost table
(assumed constants, LUTs-equivalent units) used only to bias the ILP's
scheduling/storage choices toward the cheaper element. The metric actually
reported in results.json (`dynamic_power`, `area_luts`) is measured
independently downstream from real Yosys synthesis + Verilator switching
activity (toolchain.py / power_model.py) -- the two are not the same number
by construction, and this file does not claim otherwise.
"""
import pulp
from dfg import DFG, OP_INFO

FU_AREA_LUTS = {'add': 30, 'mul': 120, 'xor': 15}          # assumed per-instance synth proxy cost
FU_POWER_PROXY = {'add': 1.0, 'mul': 4.0, 'xor': 0.4}       # assumed per-instance synth proxy cost
STORE_AREA_LUTS = {'register': 8, 'latch': 5}                # assumed per-instance synth proxy cost
STORE_POWER_PROXY = {'register': 1.1, 'latch': 0.8}          # assumed per-instance synth proxy cost
MAX_UNITS = {'add': 4, 'mul': 4, 'xor': 4}


def fu_nodes(dfg: DFG):
    """Nodes that occupy a clock cycle and a resource slot -- i.e. latency>0
    ops. shl/shr have latency 0 and are pure wires (see module docstring),
    so they are excluded even though OP_INFO gives them a resource class."""
    return [n for n in dfg.nodes.values() if OP_INFO[n.op][0] > 0]


def storage_nodes(dfg: DFG):
    """Nodes that get their own storage element: primary inputs + every FU
    result. const/shl/shr are pure wires with no storage."""
    return [n for n in dfg.nodes.values() if n.op == 'in' or OP_INFO[n.op][0] > 0]


def schedule_dfg(dfg: DFG, weights=(1.0, 1.0, 1.0), resource_limits=None, time_limit_s=8):
    """Solve the ILP for one DFG under one weight vector.
    weights = (w_latency, w_area, w_power). Returns a result dict."""
    w_lat, w_area, w_power = weights
    fus = fu_nodes(dfg)
    Tmax = max(4, len(fus) + 5)

    prob = pulp.LpProblem(f"sched_{dfg.name}", pulp.LpMinimize)

    # x[n][t] : FU node n starts at cycle t
    x = {n.id: [pulp.LpVariable(f"x_{n.id}_{t}", cat="Binary") for t in range(Tmax)] for n in fus}
    for n in fus:
        prob += pulp.lpSum(x[n.id]) == 1

    start = {n.id: pulp.lpSum(t * x[n.id][t] for t in range(Tmax)) for n in fus}

    # AVAIL(node) as a linear expression for every node, in topo order
    avail = {}
    for nid in dfg.topo_order():
        n = dfg.nodes[nid]
        if n.op in ('in', 'const'):
            avail[nid] = 0
        elif n.op in ('shl', 'shr'):
            avail[nid] = avail[n.inputs[0]]
        else:
            avail[nid] = start[nid] + 1

    # precedence: FU node start >= AVAIL(each input)
    for n in fus:
        for p in n.inputs:
            prob += start[n.id] >= avail[p]

    # resource counts (decision vars) and per-cycle resource limits
    limits = resource_limits or MAX_UNITS
    classes = sorted(set(OP_INFO[n.op][1] for n in fus))
    k = {r: pulp.LpVariable(f"k_{r}", lowBound=1, upBound=limits.get(r, 4), cat="Integer") for r in classes}
    for r in classes:
        nodes_r = [n for n in fus if OP_INFO[n.op][1] == r]
        for t in range(Tmax):
            prob += pulp.lpSum(x[n.id][t] for n in nodes_r) <= k[r]

    # storage-element type choice: one binary per storage-eligible node.
    # Every such node always needs exactly one storage element (no "need"
    # gating), so the cost is a direct linear function of use_latch_n.
    store_nodes = storage_nodes(dfg)
    use_latch = {n.id: pulp.LpVariable(f"latch_{n.id}", cat="Binary") for n in store_nodes}

    # objective terms
    makespan = pulp.LpVariable("makespan", lowBound=0, cat="Integer")
    for o in dfg.outputs():
        prob += makespan >= avail[o]

    area_fu = pulp.lpSum(k[r] * FU_AREA_LUTS[r] for r in classes)
    power_fu = pulp.lpSum(k[r] * FU_POWER_PROXY[r] for r in classes)
    area_store = pulp.lpSum(
        STORE_AREA_LUTS['register'] - use_latch[n.id] * (STORE_AREA_LUTS['register'] - STORE_AREA_LUTS['latch'])
        for n in store_nodes
    )
    power_store = pulp.lpSum(
        STORE_POWER_PROXY['register'] - use_latch[n.id] * (STORE_POWER_PROXY['register'] - STORE_POWER_PROXY['latch'])
        for n in store_nodes
    )

    prob += w_lat * makespan + w_area * (area_fu + area_store) + w_power * (power_fu + power_store)

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit_s, gapRel=0.001)
    status = prob.solve(solver)
    ok = pulp.LpStatus[status] == 'Optimal'

    start_sol = {nid: int(round(pulp.value(start[nid]))) for nid in start}
    avail_sol = {}
    for nid in dfg.topo_order():
        n = dfg.nodes[nid]
        if n.op in ('in', 'const'):
            avail_sol[nid] = 0
        elif n.op in ('shl', 'shr'):
            avail_sol[nid] = avail_sol[n.inputs[0]]
        else:
            avail_sol[nid] = start_sol[nid] + 1

    storage = {}
    for n in store_nodes:
        latch = pulp.value(use_latch[n.id]) > 0.5
        storage[n.id] = 'latch' if latch else 'register'

    resource_alloc = {r: int(round(pulp.value(k[r]))) for r in classes}
    n_latch = sum(1 for v in storage.values() if v == 'latch')
    n_register = sum(1 for v in storage.values() if v == 'register')

    return {
        'status': pulp.LpStatus[status],
        'ok': ok,
        'weights': {'latency': w_lat, 'area': w_area, 'power': w_power},
        'start': start_sol,
        'avail': avail_sol,
        'storage': storage,
        'resource_alloc': resource_alloc,
        'latency_cycles': int(round(pulp.value(makespan))),
        'proxy_area': float(pulp.value(area_fu) + pulp.value(area_store)),
        'proxy_power': float(pulp.value(power_fu) + pulp.value(power_store)),
        'n_latch': n_latch,
        'n_register': n_register,
    }
