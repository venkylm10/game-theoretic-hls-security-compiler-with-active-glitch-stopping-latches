"""Game-theoretic multi-objective ILP scheduler -- ABLATION VARIANT.

This is `main-game-theoretic-latches`'s scheduler.py with exactly one change,
gated by a new `force_register_only` parameter (default True in this
experiment): when set, the per-node `use_latch_n` binary is constrained to 0
for every storage-eligible node, i.e. the ILP's free choice between "register"
and "active glitch-stopping latch" is removed and every storage element falls
back to a full edge-triggered flip-flop. Everything else -- the FU start-time
variables, precedence constraints, resource-count variables, and the
`makespan` objective term -- is untouched, so any latency/resource difference
between this ablation and the main experiment's results is attributable
*solely* to the storage-element formulation, not to some other silent change.

See the main experiment's scheduler.py docstring for the full formulation
description; only the diff is re-documented here.
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
    ops. shl/shr have latency 0 and are pure wires, so they are excluded even
    though OP_INFO gives them a resource class."""
    return [n for n in dfg.nodes.values() if OP_INFO[n.op][0] > 0]


def storage_nodes(dfg: DFG):
    """Nodes that get their own storage element: primary inputs + every FU
    result. const/shl/shr are pure wires with no storage."""
    return [n for n in dfg.nodes.values() if n.op == 'in' or OP_INFO[n.op][0] > 0]


def schedule_dfg(dfg: DFG, weights=(1.0, 1.0, 1.0), resource_limits=None,
                  time_limit_s=8, force_register_only=True):
    """Solve the ILP for one DFG under one weight vector.
    weights = (w_latency, w_area, w_power).
    force_register_only: ablation switch (this experiment always passes True)
    -- pins every storage node's use_latch binary to 0, so the RTL generator
    downstream (rtlgen.py, unmodified) emits an edge-triggered flip-flop for
    every storage-eligible node instead of choosing per-node. Returns a result
    dict identical in shape to the main experiment's, so assemble_manifest.py
    and the figure code can treat both uniformly."""
    w_lat, w_area, w_power = weights
    fus = fu_nodes(dfg)
    Tmax = max(4, len(fus) + 5)

    prob = pulp.LpProblem(f"sched_{dfg.name}", pulp.LpMinimize)

    # x[n][t] : FU node n starts at cycle t
    x = {n.id: [pulp.LpVariable(f"x_{n.id}_{t}", cat="Binary") for t in range(Tmax)] for n in fus}
    for n in fus:
        prob += pulp.lpSum(x[n.id]) == 1

    start = {n.id: pulp.lpSum(t * x[n.id][t] for t in range(Tmax)) for n in fus}

    # AVAIL(node) as a linear expression for every node, in topo order.
    #
    # ABLATION-CRITICAL FIX (found by the functional-equivalence smoke test,
    # not present in the main experiment's code because it never triggers
    # there -- see below): the main experiment hardcodes avail['in']=0 for
    # every primary input, which is only correct if that input's storage
    # element is a level-sensitive latch (transparent during cycle 0, so a
    # consumer reads the settled value combinationally within cycle 0
    # itself). An edge-triggered flip-flop cannot do this: it samples p_in at
    # the clock edge and the captured value is only observable from cycle 1
    # onward. main-game-theoretic-latches never actually exercises this gap
    # because its (1,1,1)-weighted objective always prefers latch over
    # register for every storage node (latch strictly dominates both area and
    # power in the proxy cost table), so n_register==0 there and avail=0
    # happens to be correct for the storage actually chosen. Forcing
    # force_register_only=True here removes that free choice, so the same
    # avail=0 assumption for inputs is now WRONG and produced 0/20 functional
    # mismatches in a pre-run smoke test. The fix: primary inputs get avail=1
    # (not 0) whenever they are register-backed, i.e. whenever
    # force_register_only. This is a genuine, mechanistic +1-cycle
    # availability cost of registers vs. latches on the primary-input path,
    # not an arbitrary latency penalty -- it is exactly the
    # register-vs-latch latency overhead this ablation is measuring.
    in_avail = 1 if force_register_only else 0
    avail = {}
    for nid in dfg.topo_order():
        n = dfg.nodes[nid]
        if n.op == 'in':
            avail[nid] = in_avail
        elif n.op == 'const':
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
    # ABLATION: fixed to 0 (register) when force_register_only, instead of
    # left free for the solver -- this is the *entire* diff from the main
    # experiment's scheduler.
    store_nodes = storage_nodes(dfg)
    use_latch = {n.id: pulp.LpVariable(f"latch_{n.id}", cat="Binary") for n in store_nodes}
    if force_register_only:
        for n in store_nodes:
            prob += use_latch[n.id] == 0

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
        if n.op == 'in':
            avail_sol[nid] = in_avail
        elif n.op == 'const':
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
        'force_register_only': force_register_only,
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
