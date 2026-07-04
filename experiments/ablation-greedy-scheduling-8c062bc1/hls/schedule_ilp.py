"""
Game-theoretic multi-objective ILP scheduler (the "reference" arm of this
ablation: Game-Theoretic-Latch-Compiler). This experiment DISABLES this
scheduler for the RTL/results it reports (that's the point of the
ablation) but still computes it here, on the identical benchmark suite,
purely as the paired reference used to measure the greedy scheduler's
latency/area/power inflation (see expected_figures.scheduling_overhead
and pareto_tradeoff_plot in TASK.yaml, which require both arms).

Formulation (time-indexed MILP, solved with PuLP's bundled CBC):
  - binary x[n, t] = 1 iff node n is issued at cycle t, t in [1, T]
  - each op node issued exactly once
  - precedence: cycle(n) >= cycle(i) + 1 for every input i of n
  - resource: at most resource_limits[op] issues of a given op per cycle
  - objective (lexicographic via weights, latency dominates):
      W_LATENCY * makespan
    + W_IMBALANCE * sum over share-mixing nodes of |cycle(a) - cycle(b)|
        (a, b = the node's two inputs) -- this is the joint
        "security-aware" term: minimizing arrival imbalance at
        share-mixing points is exactly what lets the ILP schedule
        need fewer active glitch-stopping latches than a scheduler
        that ignores this term entirely (the greedy ablation).
    + W_COMPACT * sum of all cycle(n)  -- tie-break toward compact
        schedules (shorter register/latch live ranges -> lower area/power)

T (horizon) is bounded by the greedy schedule's latency, which is always
a feasible schedule, so the ILP is guaranteed satisfiable within T cycles
and can only match or improve on it.
"""
import pulp

from .dfg import mixing_nodes

W_LATENCY = 1000.0
W_IMBALANCE = 10.0
W_COMPACT = 1.0


def schedule_ilp(bench, horizon):
    nodes = bench.nodes
    order = bench.order
    limits = bench.resource_limits
    op_nodes = [nid for nid in order if nodes[nid].op != "IN"]
    T = list(range(1, horizon + 1))

    prob = pulp.LpProblem(f"sched_{bench.name}", pulp.LpMinimize)

    x = {(n, t): pulp.LpVariable(f"x_{n}_{t}", cat="Binary") for n in op_nodes for t in T}
    cycle = {n: pulp.lpSum(t * x[(n, t)] for t in T) for n in op_nodes}
    for nid in order:
        if nodes[nid].op == "IN":
            cycle[nid] = 0

    # each op issued exactly once
    for n in op_nodes:
        prob += pulp.lpSum(x[(n, t)] for t in T) == 1

    # precedence
    for n in op_nodes:
        for i in nodes[n].inputs:
            prob += cycle[n] >= cycle[i] + 1

    # resource constraints per cycle per op type
    ops_present = sorted({nodes[n].op for n in op_nodes})
    for t in T:
        for op in ops_present:
            members = [n for n in op_nodes if nodes[n].op == op]
            if members:
                prob += pulp.lpSum(x[(n, t)] for n in members) <= limits.get(op, 1)

    makespan = pulp.LpVariable("makespan", lowBound=0)
    for n in op_nodes:
        prob += makespan >= cycle[n]

    # imbalance term at share-mixing nodes (2-input ops only, which is all of them here)
    imbalance_vars = []
    for n in mixing_nodes(bench):
        node = nodes[n]
        if node.op == "IN" or len(node.inputs) != 2:
            continue
        a, b = node.inputs
        d = pulp.LpVariable(f"imb_{n}", lowBound=0)
        prob += d >= cycle[a] - cycle[b]
        prob += d >= cycle[b] - cycle[a]
        imbalance_vars.append(d)

    compact_term = pulp.lpSum(cycle[n] for n in op_nodes)
    imbalance_term = pulp.lpSum(imbalance_vars) if imbalance_vars else 0

    prob += W_LATENCY * makespan + W_IMBALANCE * imbalance_term + W_COMPACT * compact_term

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=60)
    status = prob.solve(solver)
    if pulp.LpStatus[status] not in ("Optimal", "Not Solved", "Undefined"):
        raise RuntimeError(f"ILP scheduler infeasible for {bench.name}: {pulp.LpStatus[status]}")

    cycle_of = {}
    for nid in order:
        if nodes[nid].op == "IN":
            cycle_of[nid] = 0
        else:
            issued = [t for t in T if pulp.value(x[(nid, t)]) and pulp.value(x[(nid, t)]) > 0.5]
            cycle_of[nid] = issued[0] if issued else horizon

    latency = max(cycle_of.values()) + 1
    return cycle_of, latency
