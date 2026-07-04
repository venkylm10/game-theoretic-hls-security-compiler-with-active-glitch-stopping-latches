"""Pinned benchmark suite: 6 arbitrary non-cryptographic dataflow kernels.

Each benchmark tags two primary inputs' worth of dataflow as 'share0' /
'share1' (a generic stand-in for any sensitive value split into two shares
for masking, per context.hypothesis — these are NOT cryptographic S-boxes,
just ordinary DSP/control dataflow with a masked sensitive input, which is
exactly the "arbitrary dataflow program" class this compiler targets).
Public/coefficient inputs are tagged 'pub'.

Pinned here (methodology-review fix: benchmark set was previously
unspecified/"arbitrary" -> not reproducible). Do not add/remove/rename
benchmarks without updating report.md.
"""
from dfg import Node, DFG

W = 8  # bitwidth pinned uniformly across the whole suite


def _in(id_, share):
    return Node(id_, 'in', bitwidth=W, share=share)


def _const(id_, val):
    return Node(id_, 'const', bitwidth=W, const_val=val)


def make_fir4():
    # y = h0*x0 + h1*x1 + h2*x2 + h3*x3 ; taps alternate share0/share1
    nodes = [
        _in('x0', 'share0'), _in('x1', 'share1'), _in('x2', 'share0'), _in('x3', 'share1'),
        _const('h0', 3), _const('h1', 5), _const('h2', 7), _const('h3', 11),
        Node('p0', 'mul', ('x0', 'h0'), W), Node('p1', 'mul', ('x1', 'h1'), W),
        Node('p2', 'mul', ('x2', 'h2'), W), Node('p3', 'mul', ('x3', 'h3'), W),
        Node('s01', 'add', ('p0', 'p1'), W), Node('s23', 'add', ('p2', 'p3'), W),
        Node('y', 'add', ('s01', 's23'), W, is_output=True),
    ]
    return DFG('fir4', nodes)


def make_dotprod8():
    # balanced binary-tree reduction of 8 products; shares alternate
    nodes = []
    for i in range(8):
        nodes.append(_in(f'x{i}', 'share0' if i % 2 == 0 else 'share1'))
        nodes.append(_const(f'h{i}', (i * 2 + 1) % 13 + 1))
        nodes.append(Node(f'p{i}', 'mul', (f'x{i}', f'h{i}'), W))
    level = [f'p{i}' for i in range(8)]
    lvl = 0
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            nid = f'r{lvl}_{i}'
            nodes.append(Node(nid, 'add', (level[i], level[i + 1]), W))
            nxt.append(nid)
        level = nxt
        lvl += 1
    nodes[-1] = Node(level[0], nodes[-1].op, nodes[-1].inputs, W, is_output=True) if nodes[-1].id == level[0] else nodes[-1]
    # mark the final reduction node as output
    for n in nodes:
        if n.id == level[0]:
            n.is_output = True
    return DFG('dotprod8', nodes)


def make_iir2():
    # y[n] = a1*y[n-1] + a2*y[n-2] + b0*x[n], unrolled 4 steps.
    # Feedback accumulates depth every step (unbalanced vs the fresh x[n] tap
    # of depth 1) -> the canonical "unbalanced path" benchmark.
    nodes = [_const('a1', 2), _const('a2', 3), _const('b0', 5)]
    y_hist = ['y_m2', 'y_m1']
    nodes += [_const('y_m2', 0), _const('y_m1', 0)]
    last_y = None
    for n_step in range(4):
        xin = _in(f'x{n_step}', 'share0' if n_step % 2 == 0 else 'share1')
        nodes.append(xin)
        t1 = f'fb1_{n_step}'
        t2 = f'fb2_{n_step}'
        tb = f'bx_{n_step}'
        nodes.append(Node(t1, 'mul', ('a1', y_hist[-1]), W))
        nodes.append(Node(t2, 'mul', ('a2', y_hist[-2]), W))
        nodes.append(Node(tb, 'mul', ('b0', xin.id), W))
        fbsum = f'fbsum_{n_step}'
        nodes.append(Node(fbsum, 'add', (t1, t2), W))
        yout = f'y{n_step}'
        last = Node(yout, 'add', (fbsum, tb), W, is_output=(n_step == 3))
        nodes.append(last)
        y_hist.append(yout)
        last_y = yout
    return DFG('iir2', nodes)


def make_matvec2x2():
    nodes = [
        _in('x0', 'share0'), _in('x1', 'share1'),
        _const('a00', 2), _const('a01', 3), _const('a10', 4), _const('a11', 5),
        Node('p00', 'mul', ('a00', 'x0'), W), Node('p01', 'mul', ('a01', 'x1'), W),
        Node('p10', 'mul', ('a10', 'x0'), W), Node('p11', 'mul', ('a11', 'x1'), W),
        Node('y0', 'add', ('p00', 'p01'), W, is_output=True),
        Node('y1', 'add', ('p10', 'p11'), W, is_output=True),
    ]
    return DFG('matvec2x2', nodes)


def make_crc8():
    # Bit-serial CRC-8 (poly 0x07) update over 8 input bits, alternating
    # share tags. State carries growing depth every step while each new
    # input bit is fresh (depth 0) -> unbalanced at every recombination.
    nodes = [_const('state0', 0xAB), _const('poly', 0x07)]
    state = 'state0'
    for i in range(8):
        bit = _in(f'b{i}', 'share0' if i % 2 == 0 else 'share1')
        nodes.append(bit)
        shifted = f'sh{i}'
        nodes.append(Node(shifted, 'shl', (state,), W, shift_amt=1))
        mixed = f'mix{i}'
        nodes.append(Node(mixed, 'xor', (shifted, bit.id), W))
        newstate = f'st{i}'
        nodes.append(Node(newstate, 'xor', (mixed, 'poly'), W, is_output=(i == 7)))
        state = newstate
    return DFG('crc8', nodes)


def make_adder_tree8():
    nodes = []
    for i in range(8):
        nodes.append(_in(f'x{i}', 'share0' if i % 2 == 0 else 'share1'))
    level = [f'x{i}' for i in range(8)]
    lvl = 0
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            nid = f'r{lvl}_{i}'
            nodes.append(Node(nid, 'add', (level[i], level[i + 1]), W))
            nxt.append(nid)
        level = nxt
        lvl += 1
    for n in nodes:
        if n.id == level[0]:
            n.is_output = True
    return DFG('adder_tree8', nodes)


BENCHMARKS = {
    'fir4': make_fir4,
    'dotprod8': make_dotprod8,
    'iir2': make_iir2,
    'matvec2x2': make_matvec2x2,
    'crc8': make_crc8,
    'adder_tree8': make_adder_tree8,
}


def get_benchmark(name):
    return BENCHMARKS[name]()


def all_benchmarks():
    return {name: fn() for name, fn in BENCHMARKS.items()}
