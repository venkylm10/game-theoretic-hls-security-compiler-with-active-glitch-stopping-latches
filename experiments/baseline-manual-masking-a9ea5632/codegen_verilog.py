#!/usr/bin/env python3
"""
Generates Verilog RTL for all 5 benchmarks from bench_specs.BENCHMARKS:
  <name>_base_comb.v      combinational baseline (spec function of raw a,b,c,d,...)
  <name>_base_clocked.v   baseline with input/output pipeline registers (area synthesis)
  <name>_masked_comb.v    combinational 2-share masked datapath (no registers)
  <name>_masked_clocked.v masked datapath with a Trichina register barrier per AND
                          stage (area synthesis + reflects the real pipeline depth)
  <name>_eqcheck.v        wraps base_comb + masked_comb, exposes `mismatch`, used
                          with `sat -prove mismatch 0` to formally verify that
                          masking didn't change the function computed, for ALL
                          values of the random masks (the "formal equivalence
                          check" the methodology reviewer asked for).

Naive/sequential scheduling: dependent AND nodes are chained one gadget stage
at a time (see bench_specs.py docstring) -- this is deliberately unsophisticated,
representative of "naive HLS with manual masking" rather than an optimized
scheduler.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_specs import BENCHMARKS, ORDER

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks", "generated")


def resolve_ref(ref, node_names):
    """Return (share0_expr, share1_expr) for a lhs/rhs reference."""
    kind, val = ref
    if kind == "in":
        return f"{val}0", f"{val}1"
    if kind == "inv":
        return f"(~{val}0)", f"{val}1"
    if kind == "node":
        assert val in node_names
        return f"{val}0", f"{val}1"
    if kind == "bit":
        return f"E0[{val}]", f"E1[{val}]"
    raise ValueError(ref)


def all_and_nodes(spec):
    for stage in spec["and_stages"]:
        for node in stage:
            yield node


def gen_ports_inputs(spec):
    return spec["inputs"]


def gen_base_comb(name, spec):
    w = spec["width"]
    ins = spec["inputs"]
    port_list = ", ".join(f"input [{w-1}:0] {i}" for i in ins)
    lines = [f"module {name}_base_comb (", f"    {port_list},", f"    output [{spec['output']['width']-1}:0] y", ")", ";"]
    prelude = spec.get("_verilog_base_prelude", "")
    expr = spec["_verilog_base_expr"]
    body = f"{prelude}    assign y = {expr};\nendmodule\n"
    header = "\n".join(lines)
    return header + "\n" + body


def gen_base_clocked(name, spec):
    w = spec["width"]
    ins = spec["inputs"]
    ow = spec["output"]["width"]
    port_list = ", ".join(f"input [{w-1}:0] {i}" for i in ins)
    reg_decls = "\n".join(f"    reg [{w-1}:0] {i}_r;" for i in ins)
    reg_loads = "\n".join(f"            {i}_r <= {i};" for i in ins)
    # substitute raw input names with registered names in the expression/prelude
    expr_str = spec["_verilog_base_expr"]
    prelude_str = spec.get("_verilog_base_prelude", "")
    for i in ins:
        expr_str = expr_str.replace(i, f"{i}_r")
        prelude_str = prelude_str.replace(i, f"{i}_r")
    body = f"""module {name}_base_clocked (
    input clk, input rst,
    {port_list},
    output reg [{ow-1}:0] y
);
{reg_decls}
{prelude_str}    wire [{ow-1}:0] y_comb = {expr_str};
    always @(posedge clk) begin
        if (rst) begin
{chr(10).join(f"            {i}_r <= 0;" for i in ins)}
            y <= 0;
        end else begin
{reg_loads}
            y <= y_comb;
        end
    end
endmodule
"""
    return body


def gen_masked_shared_prelude(spec, comb):
    """Emit the pre-gadget wiring shared by comb/clocked masked variants:
    share ports, optional XNOR preprocessing (comparator_eq), gadget instances.
    Returns (port_decls, body_lines, node_names, extra_out_signal_for(output_node))
    """
    w = spec["width"]
    ins = spec["inputs"]
    gadget_mod = "masked_and_gadget_comb" if comb else "masked_and_gadget"
    lines = []
    node_names = []
    # primary share ports
    share_ports = []
    for i in ins:
        share_ports.append(f"input [{w-1}:0] {i}0, input [{w-1}:0] {i}1")

    # pre-XNOR for comparator_eq
    pre = ""
    if "pre_xnor_width" in spec:
        pw = spec["pre_xnor_width"]
        lines.append(f"    wire [{pw-1}:0] E0 = ~(a0 ^ b0);")
        lines.append(f"    wire [{pw-1}:0] E1 = a1 ^ b1;")

    r_ports = []
    inst_idx = 0
    for stage in spec["and_stages"]:
        for node in stage:
            inst_idx += 1
            nw = node["width"]
            lhs0, lhs1 = resolve_ref(node["lhs"], node_names)
            rhs0, rhs1 = resolve_ref(node["rhs"], node_names)
            rname = f"r_{node['name']}"
            r_ports.append((rname, nw))
            n0, n1 = f"{node['name']}0", f"{node['name']}1"
            if comb:
                lines.append(
                    f"    wire [{nw-1}:0] {n0}, {n1};\n"
                    f"    {gadget_mod} #(.W({nw})) g_{node['name']} "
                    f"({lhs0}, {lhs1}, {rhs0}, {rhs1}, {rname}, {n0}, {n1});"
                )
            else:
                lines.append(
                    f"    wire [{nw-1}:0] {n0}, {n1};\n"
                    f"    {gadget_mod} #(.W({nw})) g_{node['name']} "
                    f"(clk, rst, {lhs0}, {lhs1}, {rhs0}, {rhs1}, {rname}, {n0}, {n1});"
                )
            node_names.append(node["name"])
    return share_ports, lines, node_names, r_ports


def gen_output_combine(spec):
    o = spec["output"]
    ow = o["width"]
    if o["kind"] == "direct":
        n = o["node"]
        return f"    wire [{ow-1}:0] y_masked = {n}0 ^ {n}1;\n", ow
    if o["kind"] == "xor":
        parts = [f"({n}0 ^ {n}1)" for n in o["nodes"]]
        expr = " ^ ".join(parts)
        return f"    wire [{ow-1}:0] y_masked = {expr};\n", ow
    if o["kind"] == "parity":
        n = o["node"]
        return f"    wire [{spec['width']-1}:0] v_{n} = {n}0 ^ {n}1;\n    wire y_masked = ^v_{n};\n", 1
    if o["kind"] == "popcount":
        n = o["node"]
        pcw = spec["width"].bit_length()  # enough bits for 0..width
        return (
            f"    wire [{spec['width']-1}:0] v_{n} = {n}0 ^ {n}1;\n"
            f"    wire [{pcw-1}:0] y_masked = " + " + ".join(f"v_{n}[{k}]" for k in range(spec["width"])) + ";\n",
            pcw,
        )
    raise ValueError(o)


def gen_masked_comb(name, spec):
    share_ports, lines, node_names, r_ports = gen_masked_shared_prelude(spec, comb=True)
    combine_lines, ow = gen_output_combine(spec)
    r_port_decl = ", ".join(f"input [{w-1}:0] {rn}" for rn, w in r_ports)
    port_list = ",\n    ".join(share_ports) + f",\n    {r_port_decl}"
    body = f"""module {name}_masked_comb (
    {port_list},
    output [{ow-1}:0] y_masked
);
{chr(10).join(lines)}
{combine_lines}
    assign y_masked = y_masked;
endmodule
"""
    # fix: can't assign to a wire already declared with initializer + output port name clash;
    # rename internal wire to y_masked_i and drive the output port from it.
    body = body.replace("wire [", "wire [", 1)
    return body


def gen_masked_comb_fixed(name, spec):
    share_ports, lines, node_names, r_ports = gen_masked_shared_prelude(spec, comb=True)
    combine_lines, ow = gen_output_combine(spec)
    combine_lines = combine_lines.replace("y_masked", "y_masked_i")
    r_port_decl = ", ".join(f"input [{w-1}:0] {rn}" for rn, w in r_ports)
    port_list = ",\n    ".join(share_ports) + (f",\n    {r_port_decl}" if r_port_decl else "")
    body = f"""module {name}_masked_comb (
    {port_list},
    output [{ow-1}:0] y_masked
);
{chr(10).join(lines)}
{combine_lines}
    assign y_masked = y_masked_i;
endmodule
"""
    return body, ow, r_ports


def gen_masked_clocked(name, spec):
    w = spec["width"]
    ins = spec["inputs"]
    share_ports, lines, node_names, r_ports = gen_masked_shared_prelude(spec, comb=False)
    combine_lines, ow = gen_output_combine(spec)
    combine_lines = combine_lines.replace("y_masked", "y_masked_i")
    r_port_decl = ", ".join(f"input [{rw-1}:0] {rn}" for rn, rw in r_ports)
    in_share_decls = ",\n    ".join(share_ports)
    # input-stage registers latch the raw share ports into *_r versions used by the gadgets
    reg_in_decls = []
    reg_in_loads = []
    reg_in_resets = []
    for i in ins:
        for s in (0, 1):
            reg_in_decls.append(f"    reg [{w-1}:0] {i}{s}_r;")
            reg_in_loads.append(f"            {i}{s}_r <= {i}{s};")
            reg_in_resets.append(f"            {i}{s}_r <= 0;")
    # rewrite gadget instance lines to use the *_r registered inputs instead of raw ports,
    # EXCEPT references to earlier node outputs (already registered internally by the gadget).
    lines_r = []
    for l in lines:
        l2 = l
        for i in ins:
            l2 = l2.replace(f" {i}0,", f" {i}0_r,").replace(f" {i}1,", f" {i}1_r,")
            l2 = l2.replace(f"({i}0,", f"({i}0_r,").replace(f"({i}1,", f"({i}1_r,")
            l2 = l2.replace(f"(~{i}0)", f"(~{i}0_r)")
        lines_r.append(l2)
    body = f"""module {name}_masked_clocked (
    input clk, input rst,
    {in_share_decls},
    {r_port_decl},
    output reg [{ow-1}:0] y
);
{chr(10).join(reg_in_decls)}
    always @(posedge clk) begin
        if (rst) begin
{chr(10).join(reg_in_resets)}
        end else begin
{chr(10).join(reg_in_loads)}
        end
    end
{chr(10).join(lines_r)}
{combine_lines}
    always @(posedge clk) begin
        if (rst) y <= 0;
        else y <= y_masked_i;
    end
endmodule
"""
    n_gadget_stages = len(spec["and_stages"])
    total_stages = 1 + n_gadget_stages + 1  # input reg + gadget stages + output reg
    return body, total_stages


def gen_eqcheck(name, spec):
    w = spec["width"]
    ins = spec["inputs"]
    r_ports = []
    for stage in spec["and_stages"]:
        for node in stage:
            r_ports.append((f"r_{node['name']}", node["width"]))
    mask_ports = [f"m{i}" for i in ins]
    raw_port_decl = ", ".join(f"input [{w-1}:0] {i}" for i in ins)
    mask_port_decl = ", ".join(f"input [{w-1}:0] {m}" for m in mask_ports)
    r_port_decl = ", ".join(f"input [{rw-1}:0] {rn}" for rn, rw in r_ports)
    share_derivations = "\n".join(f"    wire [{w-1}:0] {i}0 = {i} ^ m{i}; wire [{w-1}:0] {i}1 = m{i};" for i in ins)
    share_args = ", ".join(f"{i}0, {i}1" for i in ins)
    r_args = ", ".join(rn for rn, rw in r_ports)
    ow = spec["output"]["width"]
    body = f"""module {name}_eqcheck (
    {raw_port_decl},
    {mask_port_decl},
    {r_port_decl},
    output mismatch
);
{share_derivations}
    wire [{ow-1}:0] y_base;
    {name}_base_comb u_base ({", ".join(ins)}, y_base);
    wire [{ow-1}:0] y_masked;
    {name}_masked_comb u_masked ({share_args}, {r_args}, y_masked);
    assign mismatch = |(y_base ^ y_masked);
endmodule
"""
    return body


def build_base_expr(name, spec):
    o = spec["output"]
    ins = spec["inputs"]
    if name == "and_reduce4":
        return "a & b & c & d"
    if name == "mux_select":
        return "(sel & a) ^ (~sel & b)"
    if name == "parity_and_mask":
        return "^(a & b)"
    if name == "comparator_eq":
        w = spec["width"]
        return "&(~(a ^ b))"
    if name == "popcount_and_gate":
        width = spec["width"]
        spec["_verilog_base_prelude"] = f"    wire [{width-1}:0] ab = a & b;\n"
        terms = " + ".join(f"ab[{k}]" for k in range(width))
        return terms
    raise ValueError(name)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = {}
    for name in ORDER:
        spec = dict(BENCHMARKS[name])
        spec["_verilog_base_expr"] = build_base_expr(name, spec)
        base_comb = gen_base_comb(name, spec)
        base_clocked = gen_base_clocked(name, spec)
        masked_comb, comb_ow, r_ports = gen_masked_comb_fixed(name, spec)
        masked_clocked, total_stages = gen_masked_clocked(name, spec)
        eqcheck = gen_eqcheck(name, spec)

        fname = os.path.join(OUT_DIR, f"{name}.v")
        with open(fname, "w") as f:
            f.write(f"// Auto-generated by codegen_verilog.py for benchmark `{name}`.\n")
            f.write(f"// {spec['description']}\n\n")
            f.write(base_comb + "\n")
            f.write(base_clocked + "\n")
            f.write(masked_comb + "\n")
            f.write(masked_clocked + "\n")
            f.write(eqcheck + "\n")
        manifest[name] = {
            "file": fname,
            "masked_stage_count": total_stages,
            "baseline_stage_count": 2,
            "r_ports": r_ports,
            "width": spec["width"],
            "output_width": comb_ow,
        }
        print(f"generated {fname}  (masked stages={total_stages}, baseline stages=2)")
    return manifest


if __name__ == "__main__":
    main()
