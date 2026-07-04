"""
Standard greedy resource-constrained list scheduler.

This is the "under_test" subject of this ablation (Greedy-Latch-Compiler):
it keeps the active glitch-stopping latch mechanism (see hls/latches.py)
but replaces the game-theoretic multi-objective ILP schedule (used by the
sibling main-game-theoretic-latches / reference schedule) with textbook
ASAP list scheduling. It has NO look-ahead: nodes are offered to
functional units strictly in definition (program) order, one ready-list
pass per cycle, with no critical-path or slack-based priority and no
joint reasoning about the resulting area/power/security cost. This is
exactly the "naive greedy scheduler" the objective in TASK.yaml asks us
to isolate the contribution of the ILP against.

A schedule is a dict {node_id: cycle:int}. IN nodes are pinned to cycle 0.
Every op node's cycle must be > max(cycle of its inputs), and at most
resource_limits[op] nodes of a given op type may share a cycle.
"""


def schedule_greedy(bench):
    nodes = bench.nodes
    order = bench.order
    limits = bench.resource_limits

    cycle_of = {}
    for nid in order:
        if nodes[nid].op == "IN":
            cycle_of[nid] = 0

    remaining = [nid for nid in order if nodes[nid].op != "IN"]
    usage = {}  # (cycle, op) -> count

    cycle = 1
    # Greedy: repeatedly scan the remaining list IN PROGRAM ORDER (no
    # priority function) and pack whatever is ready into the current
    # cycle until a resource limit blocks it, then advance the cycle.
    guard = 0
    while remaining:
        guard += 1
        if guard > 100000:
            raise RuntimeError("greedy scheduler failed to converge")
        progressed = False
        still_remaining = []
        for nid in remaining:
            node = nodes[nid]
            inputs_ready = all(i in cycle_of and cycle_of[i] < cycle for i in node.inputs)
            if not inputs_ready:
                still_remaining.append(nid)
                continue
            key = (cycle, node.op)
            used = usage.get(key, 0)
            if used >= limits.get(node.op, 1):
                still_remaining.append(nid)
                continue
            cycle_of[nid] = cycle
            usage[key] = used + 1
            progressed = True
        remaining = still_remaining
        if remaining:
            cycle += 1
            if not progressed and cycle > len(order) + 5:
                # Nothing at all fit this cycle and we've well exceeded any
                # sane horizon -> a correctness bug, not legitimate stalling.
                raise RuntimeError("greedy scheduler stalled with no progress")

    latency = max(cycle_of.values()) + 1  # +1: last op's result settles one more cycle
    return cycle_of, latency
