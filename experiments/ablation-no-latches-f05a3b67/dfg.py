"""Dataflow-graph (DFG) core: node representation, generic golden evaluator,
and share-recombination analysis shared by the scheduler, RTL generator, and
Python cycle simulator. Every downstream stage (ILP scheduler, Verilog
codegen, behavioral simulator) derives its op semantics from OP_INFO here, so
they cannot silently drift apart.
"""
from dataclasses import dataclass, field

MASK8 = 0xFF


@dataclass
class Node:
    id: str
    op: str                      # 'in','const','add','sub','mul','xor','shl','shr'
    inputs: tuple = ()           # tuple of node ids (order matters for shl/shr amt, sub)
    bitwidth: int = 8
    const_val: int = None
    shift_amt: int = None        # only for shl/shr
    share: str = None            # 'share0' | 'share1' | 'pub' | None -- only meaningful for op=='in'
    is_output: bool = False


# op -> (latency_cycles, resource_class, arity)
OP_INFO = {
    'in':    (0, None,   0),
    'const': (0, None,   0),
    'add':   (1, 'add',  2),
    'sub':   (1, 'add',  2),
    'xor':   (1, 'xor',  2),
    'mul':   (1, 'mul',  2),
    'shl':   (0, 'shift', 1),
    'shr':   (0, 'shift', 1),
}


class DFG:
    def __init__(self, name, nodes):
        self.name = name
        self.nodes = {n.id: n for n in nodes}
        self.order = list(self.nodes.keys())  # insertion order; caller must supply topo order
        for n in nodes:
            for i in n.inputs:
                if i not in self.nodes:
                    raise ValueError(f"{name}: node {n.id} references undefined input {i}")

    def topo_order(self):
        """Return node ids in a valid topological order (Kahn's algorithm).
        Insertion order is expected to already be topological, but we verify."""
        visited = set()
        out = []

        def visit(nid):
            if nid in visited:
                return
            visited.add(nid)
            for i in self.nodes[nid].inputs:
                visit(i)
            out.append(nid)

        for nid in self.order:
            visit(nid)
        return out

    def outputs(self):
        return [n.id for n in self.nodes.values() if n.is_output]

    def inputs_of_kind(self, share):
        return [n.id for n in self.nodes.values() if n.op == 'in' and n.share == share]


def golden_eval(dfg: DFG, inputs: dict) -> dict:
    """Pure-Python reference evaluation of a DFG, independent of the RTL/
    scheduler code path. `inputs` maps 'in' node id -> int value."""
    vals = {}
    for nid in dfg.topo_order():
        n = dfg.nodes[nid]
        w = n.bitwidth
        mask = (1 << w) - 1
        if n.op == 'in':
            vals[nid] = inputs[nid] & mask
        elif n.op == 'const':
            vals[nid] = n.const_val & mask
        elif n.op == 'add':
            a, b = n.inputs
            vals[nid] = (vals[a] + vals[b]) & mask
        elif n.op == 'sub':
            a, b = n.inputs
            vals[nid] = (vals[a] - vals[b]) & mask
        elif n.op == 'xor':
            a, b = n.inputs
            vals[nid] = (vals[a] ^ vals[b]) & mask
        elif n.op == 'mul':
            a, b = n.inputs
            vals[nid] = (vals[a] * vals[b]) & mask
        elif n.op == 'shl':
            a, = n.inputs
            vals[nid] = (vals[a] << n.shift_amt) & mask
        elif n.op == 'shr':
            a, = n.inputs
            vals[nid] = (vals[a] >> n.shift_amt) & mask
        else:
            raise ValueError(f"unknown op {n.op}")
    return {o: vals[o] for o in dfg.outputs()}


def share_sets(dfg: DFG) -> dict:
    """node id -> frozenset of shares ({'share0','share1'} subset) reachable
    transitively upstream of that node. 'pub' inputs contribute nothing."""
    sets = {}
    for nid in dfg.topo_order():
        n = dfg.nodes[nid]
        if n.op == 'in':
            sets[nid] = frozenset({n.share}) if n.share in ('share0', 'share1') else frozenset()
        elif n.op == 'const':
            sets[nid] = frozenset()
        else:
            s = frozenset()
            for i in n.inputs:
                s = s | sets[i]
            sets[nid] = s
    return sets


def recombination_points(dfg: DFG):
    """Return list of (node_id, input_a, input_b) triples marking the FIRST
    point where two purely-single-share subtrees (share0-only and
    share1-only) meet as distinct inputs of the same node. This is the
    security-critical set: the ILP scheduler must force equal arrival cycles
    for (input_a, input_b) at each such node (path balancing), and the RTL
    generator must never let a combinational glitch window exist there."""
    sets = share_sets(dfg)
    crit = []
    for nid in dfg.topo_order():
        n = dfg.nodes[nid]
        if len(n.inputs) < 2:
            continue
        ins = list(n.inputs)
        for a_idx in range(len(ins)):
            for b_idx in range(a_idx + 1, len(ins)):
                a, b = ins[a_idx], ins[b_idx]
                sa, sb = sets[a], sets[b]
                if len(sa) == 1 and len(sb) == 1 and sa != sb:
                    crit.append((nid, a, b))
    return crit
