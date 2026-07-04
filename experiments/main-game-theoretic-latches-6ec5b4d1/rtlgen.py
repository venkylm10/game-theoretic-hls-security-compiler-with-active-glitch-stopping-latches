"""Verilog RTL generation from an ILP schedule (scheduler.schedule_dfg output).

Every storage-eligible node (dfg 'in' nodes and every FU result) gets exactly
one storage element, whose Verilog shape depends on scheduler.py's
`storage[node_id]` choice:

  - 'register': `always @(posedge clk) if (cycle_cnt==T) sig <= d;`
    -> edge-triggered flip-flop (yosys infers $dff/$adff).
  - 'latch':    `always @* if (cycle_cnt==T) sig = d;`
    -> level-sensitive latch, transparent ONLY during cycle T, opaque
    (holding) every other cycle (yosys infers $dlatch). This is the
    "dynamically controlled active glitch-stopping latch": its enable is
    derived from the FSM's cycle counter, so it is transparent for exactly
    one designated cycle per run, exactly like the paper's mechanism.

Functional units are time-multiplexed onto `resource_alloc[class]` physical
instances (greedy binding: same (class, cycle) group never collides, and
distinct cycles are free to reuse the same instance index — this is exactly
what the ILP's resource constraint guarantees). Each instance is a single
combinational case-muxed ALU selecting its operator+operands by cycle_cnt.
"""
from dfg import DFG, OP_INFO

OPSYM = {'add': '+', 'sub': '-', 'xor': '^', 'mul': '*'}


def bind_resources(dfg: DFG, sched: dict):
    """node_id -> physical instance index within its resource class."""
    by_class_cycle = {}
    for nid, t in sched['start'].items():
        n = dfg.nodes[nid]
        r = OP_INFO[n.op][1]
        by_class_cycle.setdefault((r, t), []).append(nid)
    bindings = {}
    for (_r, _t), ids in by_class_cycle.items():
        for idx, nid in enumerate(ids):
            bindings[nid] = idx
    return bindings


def generate_rtl(dfg: DFG, sched: dict, w=8):
    bindings = bind_resources(dfg, sched)
    storage = sched['storage']
    start = sched['start']

    lines = []
    ports_in = [n for n in dfg.nodes.values() if n.op == 'in']
    outs = dfg.outputs()

    lines.append(f"module {dfg.name} (")
    lines.append("    input wire clk,")
    lines.append("    input wire rst,")
    for n in ports_in:
        lines.append(f"    input wire [{w-1}:0] p_{n.id},")
    for i, o in enumerate(outs):
        sep = ',' if i < len(outs) - 1 else ''
        lines.append(f"    output wire [{w-1}:0] o_{o}{sep}")
    lines.append(");")
    lines.append(f"  reg [7:0] cycle_cnt;")
    lines.append("")

    # group FU nodes by (class, instance) for ALU mux generation
    alu_groups = {}
    for nid, t in start.items():
        n = dfg.nodes[nid]
        r = OP_INFO[n.op][1]
        u = bindings[nid]
        alu_groups.setdefault((r, u), []).append(nid)

    def sig(nid):
        return f"sig_{nid}"

    # --- pass 1: declare every signal up front (Icarus Verilog requires
    # declaration-before-use within a module scope, unlike some simulators
    # that build a full symbol table before elaboration) ---
    for nid in dfg.topo_order():
        n = dfg.nodes[nid]
        if n.op in ('const', 'shl', 'shr'):
            lines.append(f"  wire [{w-1}:0] {sig(nid)};")
        else:
            lines.append(f"  reg [{w-1}:0] {sig(nid)};")
    for (r, u) in alu_groups:
        lines.append(f"  reg [{w-1}:0] alu_{r}_{u};")
    lines.append("")

    # --- pass 2: drive every signal ---
    lines.append("  always @(posedge clk) if (rst) cycle_cnt <= 8'd0; else cycle_cnt <= cycle_cnt + 8'd1;")
    for nid in dfg.topo_order():
        n = dfg.nodes[nid]
        if n.op == 'const':
            lines.append(f"  assign {sig(nid)} = {w}'d{n.const_val};")
        elif n.op == 'in':
            stype = storage[nid]
            if stype == 'register':
                lines.append(f"  always @(posedge clk) if (rst) {sig(nid)} <= {w}'d0; else if (cycle_cnt == 8'd0) {sig(nid)} <= p_{nid};")
            else:
                lines.append(f"  always @* if (cycle_cnt == 8'd0) {sig(nid)} = p_{nid};  // active glitch-stopping latch: opaque except cycle 0")
        elif n.op in ('shl', 'shr'):
            opsym = '<<' if n.op == 'shl' else '>>'
            lines.append(f"  assign {sig(nid)} = {sig(n.inputs[0])} {opsym} {n.shift_amt};")
    lines.append("")

    # ALU instances (one comb mux per physical functional unit)
    for (r, u), ids in alu_groups.items():
        alu_name = f"alu_{r}_{u}"
        lines.append("  always @* begin")
        lines.append(f"    {alu_name} = {w}'d0;")
        lines.append("    case (cycle_cnt)")
        for nid in ids:
            n = dfg.nodes[nid]
            a, b = n.inputs
            opsym = OPSYM[n.op]
            lines.append(f"      8'd{start[nid]}: {alu_name} = {sig(a)} {opsym} {sig(b)};")
        lines.append("      default: ;")
        lines.append("    endcase")
        lines.append("  end")
    lines.append("")

    # storage element per FU node result
    for nid, t in start.items():
        n = dfg.nodes[nid]
        r = OP_INFO[n.op][1]
        u = bindings[nid]
        alu_name = f"alu_{r}_{u}"
        stype = storage[nid]
        if stype == 'register':
            lines.append(f"  always @(posedge clk) if (rst) {sig(nid)} <= {w}'d0; else if (cycle_cnt == 8'd{t}) {sig(nid)} <= {alu_name};")
        else:
            lines.append(f"  always @* if (cycle_cnt == 8'd{t}) {sig(nid)} = {alu_name};  // active glitch-stopping latch: opaque except cycle {t}")
    lines.append("")

    for o in outs:
        lines.append(f"  assign o_{o} = {sig(o)};")
    lines.append("endmodule")
    return "\n".join(lines) + "\n"


def generate_testbench(dfg: DFG, sched: dict, vectors, w=8, vcd_name=None):
    """Self-checking testbench: applies each vector (dict of input_id->int,
    plus vector['expected'] = dict of output_id->int computed independently
    by dfg.golden_eval), waits for the pipeline to drain, and compares. Emits
    one summary line `EQUIV_RESULT pass=.. fail=.. total=..` at $finish and
    dumps a VCD of the whole run (used both as the required simulation_vcd
    artifact and as ground truth for the Hamming-weight/distance power
    model)."""
    ins = [n.id for n in dfg.nodes.values() if n.op == 'in']
    outs = dfg.outputs()
    latency = sched['latency_cycles']
    vcd_name = vcd_name or f"{dfg.name}.vcd"

    lines = []
    lines.append("`timescale 1ns/1ps")
    lines.append(f"module tb_{dfg.name};")
    lines.append("  reg clk; reg rst;")
    for i in ins:
        lines.append(f"  reg [{w-1}:0] p_{i};")
    for o in outs:
        lines.append(f"  wire [{w-1}:0] o_{o};")
    lines.append("  integer pass_cnt, fail_cnt, total_cnt;")
    lines.append("")
    conns = ["    .clk(clk)", "    .rst(rst)"]
    conns += [f"    .p_{i}(p_{i})" for i in ins]
    conns += [f"    .o_{o}(o_{o})" for o in outs]
    lines.append(f"  {dfg.name} dut (")
    lines.append(",\n".join(conns))
    lines.append("  );")
    lines.append("")
    lines.append("  initial clk = 0;")
    lines.append("  always #5 clk = ~clk;")
    lines.append("")
    argdecls = [f"input [{w-1}:0] in_{i}" for i in ins] + [f"input [{w-1}:0] exp_{o}" for o in outs]
    lines.append("  task run_vector(")
    lines.append("    " + ",\n    ".join(argdecls))
    lines.append("  );")
    lines.append("  begin")
    lines.append("    rst = 1;")
    for i in ins:
        lines.append(f"    p_{i} = in_{i};")
    lines.append("    @(posedge clk); @(posedge clk);")
    lines.append("    rst = 0;")
    lines.append(f"    repeat ({latency + 2}) @(posedge clk);")
    lines.append("    #1;")
    for o in outs:
        lines.append(f"    if (o_{o} !== exp_{o}) fail_cnt = fail_cnt + 1; else pass_cnt = pass_cnt + 1;")
        lines.append("    total_cnt = total_cnt + 1;")
    lines.append("  end")
    lines.append("  endtask")
    lines.append("")
    lines.append("  initial begin")
    lines.append(f'    $dumpfile("{vcd_name}");')
    lines.append(f"    $dumpvars(0, tb_{dfg.name});")
    lines.append("    pass_cnt = 0; fail_cnt = 0; total_cnt = 0;")
    for vec in vectors:
        args = [str(vec[i]) for i in ins] + [str(vec['expected'][o]) for o in outs]
        lines.append(f"    run_vector({', '.join(args)});")
    lines.append('    $display("EQUIV_RESULT pass=%0d fail=%0d total=%0d", pass_cnt, fail_cnt, total_cnt);')
    lines.append("    $finish;")
    lines.append("  end")
    lines.append("endmodule")
    return "\n".join(lines) + "\n"
