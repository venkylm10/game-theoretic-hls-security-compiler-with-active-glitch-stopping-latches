"""
Resource binding: assigns each op node to a physical functional-unit
instance of its type (0-indexed, count == resource_limits[op]).

Binding is deliberately a simple deterministic greedy pass (assign the
lowest free unit index within each (cycle, op) group, ties broken by node
id) for BOTH the greedy and ILP-scheduled variants -- the ablation under
test is the scheduler, not the binder, so binding policy is held
constant to isolate the scheduling contribution per TASK.yaml's
objective.
"""


def bind_units(bench, cycle_of):
    """Return {node_id: unit_index} for every op (non-IN) node."""
    groups = {}
    for nid in bench.order:
        node = bench.nodes[nid]
        if node.op == "IN":
            continue
        key = (cycle_of[nid], node.op)
        groups.setdefault(key, []).append(nid)

    unit_of = {}
    for (cycle, op), members in groups.items():
        for idx, nid in enumerate(sorted(members)):
            unit_of[nid] = idx
    return unit_of


def unit_counts(bench):
    """Physical functional units instantiated per op type (== resource_limits)."""
    return dict(bench.resource_limits)
