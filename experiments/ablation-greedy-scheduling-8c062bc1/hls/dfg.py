"""
Dataflow-graph (DFG) representation and a small suite of synthetic,
non-cryptographic benchmark programs used across the ablation.

Each benchmark is a DAG of arithmetic operations over a small set of
primary inputs, two of which ("s0", "s1") are the two Boolean-masking
shares of a single secret value (s0 XOR s1 == secret). The rest of the
inputs are public. A node's "taint" is the set of shares {A, B} whose
data has propagated into it. A node is a "mixing point" if both A and B
appear in its taint. Only the designated primary output is allowed to be
a mixing point "by design" (the final unmasking/recombination); every
other mixing point is a potential glitch-recombination hazard that the
compiler must neutralize with an active glitch-stopping latch on
whichever operand arrives at a different schedule cycle than its sibling
operand (see hls/latches.py).

This benchmark suite is deliberately synthetic (no external dataset is
pinned for this experiment; TASK.yaml.subject.datasets is empty), fixed
by a seed for reproducibility, and shared verbatim by both the greedy
scheduler (this experiment) and the ILP reference scheduler recomputed
here for a paired comparison (see code/README.md for why the reference
is recomputed in-run rather than read from a sibling pod's artifacts).
"""
import random
from dataclasses import dataclass, field


@dataclass
class Node:
    id: str
    op: str            # 'IN' | 'ADD' | 'SUB' | 'MUL'
    inputs: list        # list of node ids (empty for 'IN')
    width: int = 16     # bit width


@dataclass
class Benchmark:
    name: str
    nodes: dict          # id -> Node, in a valid topological (definition) order
    order: list          # topological order of node ids
    primary_inputs: list  # ids of 'IN' nodes that are NOT shares
    share_inputs: tuple  # (id_share0, id_share1)
    output: str          # id of the primary output node
    resource_limits: dict  # {'ADD': n, 'MUL': n} functional units available


def _build(name, n_ops, mul_fraction, resource_limits, seed):
    """Generate a random-but-deterministic DAG with two share sources."""
    rng = random.Random(seed)
    nodes = {}
    order = []

    def add_node(nid, op, inputs, width=16):
        nodes[nid] = Node(nid, op, inputs, width)
        order.append(nid)

    # Primary inputs: 2 masking shares + a handful of public operands.
    add_node("s0", "IN", [])
    add_node("s1", "IN", [])
    n_public = max(2, n_ops // 3)
    public_inputs = []
    for i in range(n_public):
        pid = f"pub{i}"
        add_node(pid, "IN", [])
        public_inputs.append(pid)

    available = ["s0", "s1"] + public_inputs
    for i in range(n_ops):
        nid = f"op{i}"
        op = "MUL" if rng.random() < mul_fraction else rng.choice(["ADD", "SUB"])
        # Pick 2 distinct operands, biased toward recently created nodes so
        # the graph has real depth instead of a flat fan-in fan-out star.
        a = rng.choice(available)
        b = rng.choice(available)
        tries = 0
        while b == a and tries < 5:
            b = rng.choice(available)
            tries += 1
        add_node(nid, op, [a, b])
        available.append(nid)
        # occasionally prune the pool so later ops build on recent results
        if len(available) > 6 and rng.random() < 0.5:
            available.pop(rng.randrange(len(available) - 3))

    output = available[-1]
    return Benchmark(
        name=name,
        nodes=nodes,
        order=order,
        primary_inputs=public_inputs,
        share_inputs=("s0", "s1"),
        output=output,
        resource_limits=dict(resource_limits),
    )


def load_benchmark_suite():
    """The fixed benchmark suite used for every (scheduler, run) pair."""
    specs = [
        ("dot_product4",    8,  0.35, {"ADD": 2, "MUL": 1}, 1001),
        ("fir4_tap",        10, 0.40, {"ADD": 2, "MUL": 1}, 1002),
        ("poly_eval5",      12, 0.30, {"ADD": 2, "MUL": 2}, 1003),
        ("matvec2x2",       14, 0.35, {"ADD": 2, "MUL": 2}, 1004),
        ("sad_block",       16, 0.15, {"ADD": 3, "MUL": 1}, 1005),
        ("iir2_section",    18, 0.30, {"ADD": 2, "MUL": 2}, 1006),
        ("moving_avg8",     20, 0.10, {"ADD": 3, "MUL": 1}, 1007),
        ("wgt_sum_mix",     22, 0.35, {"ADD": 3, "MUL": 2}, 1008),
    ]
    return [_build(*s) for s in specs]


def taints(bench):
    """Return {node_id: frozenset(subset of {'A','B'})} via forward propagation."""
    t = {}
    for nid in bench.order:
        node = bench.nodes[nid]
        if node.op == "IN":
            if nid == bench.share_inputs[0]:
                t[nid] = frozenset({"A"})
            elif nid == bench.share_inputs[1]:
                t[nid] = frozenset({"B"})
            else:
                t[nid] = frozenset()
        else:
            s = frozenset()
            for i in node.inputs:
                s = s | t[i]
            t[nid] = s
    return t


def mixing_nodes(bench):
    """Node ids where both shares (A and B) are present in the taint."""
    t = taints(bench)
    return {nid for nid, s in t.items() if {"A", "B"} <= s}
