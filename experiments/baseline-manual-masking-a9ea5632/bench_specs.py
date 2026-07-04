"""
Single source of truth for the 5 non-crypto dataflow benchmarks used in this
baseline. Both the Verilog codegen (codegen_verilog.py) and the Python
cycle-accurate simulator (sim_model.py) read these specs, so the RTL used for
synthesis/formal-equivalence and the model used for dynamic simulation cannot
drift apart.

Each benchmark is a small dataflow graph over 2-share-Boolean-masked secret
operands. The only nonlinear primitive is bitwise AND, protected with the
Trichina (2003) register-based masked-AND gadget (see benchmarks/
masked_and_gadget.v). All other operators used (XOR/XNOR/popcount-adder) are
GF(2)-affine and therefore require no gadget: an affine function of shares is
itself a valid (uniform, independent) sharing of the affine function of the
secret.

`and_nodes`: list of AND operations, each consuming either a primary secret
input or the output of an earlier AND node. `naive` scheduling chains
dependent AND nodes strictly sequentially (one gadget stage per node in
`chain` groups) -- deliberately unsophisticated, standing in for "naive HLS"
scheduling as opposed to the balanced/optimal scheduling the sibling
game-theoretic ILP experiment is expected to find.
"""

BENCHMARKS = {
    "and_reduce4": {
        "description": "4-way secret AND reduction (bitmask/permission-check dataflow block)",
        "width": 8,
        "inputs": ["a", "b", "c", "d"],
        # sequential (naive) chain: n1=a&b, n2=n1&c, n3=n2&d
        "and_stages": [
            [{"name": "n1", "lhs": ("in", "a"), "rhs": ("in", "b"), "width": 8}],
            [{"name": "n2", "lhs": ("node", "n1"), "rhs": ("in", "c"), "width": 8}],
            [{"name": "n3", "lhs": ("node", "n2"), "rhs": ("in", "d"), "width": 8}],
        ],
        "output": {"kind": "direct", "node": "n3", "width": 8},
        "baseline_expr": "a & b & c & d",
    },
    "mux_select": {
        "description": "secret-select 2:1 multiplexer dataflow block",
        "width": 8,
        "inputs": ["sel", "a", "b"],  # sel is broadcast to `width` bits
        "and_stages": [
            [
                {"name": "n1", "lhs": ("in", "sel"), "rhs": ("in", "a"), "width": 8},
                {"name": "n2", "lhs": ("inv", "sel"), "rhs": ("in", "b"), "width": 8},
            ],
        ],
        "output": {"kind": "xor", "nodes": ["n1", "n2"], "width": 8},
        "baseline_expr": "(sel & a) ^ (~sel & b)",
    },
    "parity_and_mask": {
        "description": "masked AND followed by parity/checksum reduction",
        "width": 8,
        "inputs": ["a", "b"],
        "and_stages": [
            [{"name": "n1", "lhs": ("in", "a"), "rhs": ("in", "b"), "width": 8}],
        ],
        "output": {"kind": "parity", "node": "n1", "width": 1},
        "baseline_expr": "^(a & b)",
    },
    "comparator_eq": {
        "description": "equality/bitmask-match comparator (naive 8-way sequential AND-reduce)",
        "width": 8,  # width of primary inputs a,b; the AND-reduce tree itself operates bitwise (node width 1)
        "inputs": ["a", "b"],  # each 8 bits wide; compared bitwise then AND-reduced
        "pre_xnor_width": 8,
        # 7 chained 1-bit gadgets: m1=e0&e1, m2=m1&e2, ..., m7=m6&e7
        "and_stages": [
            [{"name": "m1", "lhs": ("bit", 0), "rhs": ("bit", 1), "width": 1}],
            [{"name": "m2", "lhs": ("node", "m1"), "rhs": ("bit", 2), "width": 1}],
            [{"name": "m3", "lhs": ("node", "m2"), "rhs": ("bit", 3), "width": 1}],
            [{"name": "m4", "lhs": ("node", "m3"), "rhs": ("bit", 4), "width": 1}],
            [{"name": "m5", "lhs": ("node", "m4"), "rhs": ("bit", 5), "width": 1}],
            [{"name": "m6", "lhs": ("node", "m5"), "rhs": ("bit", 6), "width": 1}],
            [{"name": "m7", "lhs": ("node", "m6"), "rhs": ("bit", 7), "width": 1}],
        ],
        "output": {"kind": "direct", "node": "m7", "width": 1},
        "baseline_expr": "&(~(a ^ b))",
    },
    "popcount_and_gate": {
        "description": "Hamming-weight of secret AND-gated bits (revealed at gadget boundary, "
                        "then public popcount adder tree)",
        "width": 8,
        "inputs": ["a", "b"],
        "and_stages": [
            [{"name": "n1", "lhs": ("in", "a"), "rhs": ("in", "b"), "width": 8}],
        ],
        "output": {"kind": "popcount", "node": "n1", "width": 4},
        "baseline_expr": "popcount(a & b)",
    },
}

ORDER = ["and_reduce4", "mux_select", "parity_and_mask", "comparator_eq", "popcount_and_gate"]
