"""
Writes a small set of representative simulation artifacts into
simulation_traces/ for auditability:
  - a real VCD waveform (clk + primary I/O) for a handful of back-to-back
    traces, written directly from the same netlist_sim.Netlist state used
    for the TVLA sweep (no separate/second simulation code path)
  - a CSV of per-trace inputs/outputs/register-HD-power-sample for a larger
    (but still small, ~2000-trace) sample, for statistical spot-checking
    without needing to re-run the full multi-million-trace sweep
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from netlist_sim import Netlist

ROOT = "/workspace/output"


def write_vcd(nl, port_stimulus_fn, n_traces, out_path, watch_ports):
    depth = nl.pipeline_depth()
    n_cycles = depth + 1
    state = nl.new_state(1)
    lines = []
    lines.append("$timescale 10ns $end")
    lines.append("$scope module sbox $end")
    symbols = {}
    next_sym = ord("!")
    for p in watch_ports:
        width = len(nl.ports[p]["bits"])
        sym = chr(next_sym); next_sym += 1
        symbols[p] = (sym, width)
        lines.append(f"$var wire {width} {sym} {p} $end")
    lines.append("$upscope $end")
    lines.append("$enddefinitions $end")
    lines.append("$dumpvars")
    for p in watch_ports:
        sym, width = symbols[p]
        lines.append(("b" + "0" * width + " " + sym) if width > 1 else ("0" + sym))
    lines.append("$end")

    t = 0
    for trace_idx in range(n_traces):
        pi_bits = port_stimulus_fn(1, seed=trace_idx)
        for cyc in range(n_cycles):
            frame = {"clk": np.ones(1, dtype=bool)}
            frame.update(pi_bits)
            net_vals, state, outputs = nl.step(frame, state, 1)
            lines.append(f"#{t}")
            for p in watch_ports:
                sym, width = symbols[p]
                if p in outputs:
                    bits = outputs[p]
                elif p in pi_bits:
                    bits = pi_bits[p]
                else:
                    continue
                val = 0
                for i, b in enumerate(bits):
                    val |= (int(bool(b[0])) << i)
                bitstr = format(val, f"0{width}b")
                lines.append((f"b{bitstr} {sym}") if width > 1 else (f"{bitstr}{sym}"))
            t += 5
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def bits_n(vec, n):
    return [((vec >> i) & 1).astype(bool) for i in range(n)]


def aes_stimulus(batch, seed):
    from aes_reference import REFERENCE_R, REFERENCE_DEC
    rng = np.random.default_rng(seed)
    t0v = rng.integers(0, 256, size=batch).astype(np.uint32)
    t1v = rng.integers(0, 256, size=batch).astype(np.uint32)
    pi = {"t0": bits_n(t0v, 8), "t1": bits_n(t1v, 8)}
    for k, v in REFERENCE_R.items():
        pi[k] = bits_n(np.full(batch, v, dtype=np.uint32), 8)
    for k, v in REFERENCE_DEC.items():
        pi[k] = bits_n(np.full(batch, v, dtype=np.uint32), 8)
    return pi


def present_stimulus(batch, seed):
    rng = np.random.default_rng(seed)
    pi = {}
    for p in ["x0_0", "x1_0", "x2_0", "x3_0", "x0_1", "x1_1", "x2_1", "x3_1", "r"]:
        pi[p] = [rng.integers(0, 2, size=batch).astype(bool)]
    return pi


def export_csv(nl, stimulus_fn, n_traces, out_path, in_ports, out_ports):
    depth = nl.pipeline_depth()
    n_cycles = depth + 1
    batch = n_traces
    pi_bits = stimulus_fn(batch, seed=999)
    dff_cnames = [c for c, _ in nl.dff_cells]
    state = nl.new_state(batch)
    prev = np.zeros((batch, len(dff_cnames)), dtype=bool)
    hd_total = np.zeros(batch)
    for cyc in range(n_cycles):
        frame = {"clk": np.ones(batch, dtype=bool)}
        frame.update(pi_bits)
        _, state, outputs = nl.step(frame, state, batch)
        cur = np.stack([state[c] for c in dff_cnames], axis=1)
        hd_total += (cur != prev).sum(axis=1)
        prev = cur

    def to_int(bits):
        v = np.zeros(batch, dtype=np.uint32)
        for i, b in enumerate(bits):
            v |= (b.astype(np.uint32) << i)
        return v

    cols = {}
    for p in in_ports:
        cols[p] = to_int(pi_bits[p])
    for p in out_ports:
        cols[p] = to_int(outputs[p])
    cols["total_register_hd_all_cycles"] = hd_total

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = list(cols.keys())
        writer.writerow(header)
        for i in range(batch):
            writer.writerow([int(cols[h][i]) for h in header])


def main():
    aes_nl = Netlist(os.path.join(ROOT, "compiled_rtl/aes_sbox_gatelevel.json"), top="sbox")
    present_nl = Netlist(os.path.join(ROOT, "compiled_rtl/present_sbox_gatelevel.json"), top="sbox")

    out_dir = os.path.join(ROOT, "simulation_traces")
    os.makedirs(out_dir, exist_ok=True)

    write_vcd(aes_nl, aes_stimulus, n_traces=5,
              out_path=os.path.join(out_dir, "aes_sbox_sample.vcd"),
              watch_ports=["clk", "t0", "t1", "y0", "y1"])
    write_vcd(present_nl, present_stimulus, n_traces=5,
              out_path=os.path.join(out_dir, "present_sbox_sample.vcd"),
              watch_ports=["clk", "x0_0", "x1_0", "x2_0", "x3_0", "x0_1", "x1_1", "x2_1", "x3_1",
                           "r", "Y0_0", "Y1_0", "Y2_0", "Y3_0", "Y0_1", "Y1_1", "Y2_1", "Y3_1"])
    print("wrote sample VCDs")

    export_csv(aes_nl, aes_stimulus, 2000,
               os.path.join(out_dir, "aes_sbox_sample_traces.csv"),
               in_ports=["t0", "t1"], out_ports=["y0", "y1"])
    export_csv(present_nl, present_stimulus, 2000,
               os.path.join(out_dir, "present_sbox_sample_traces.csv"),
               in_ports=["x0_0", "x1_0", "x2_0", "x3_0", "x0_1", "x1_1", "x2_1", "x3_1", "r"],
               out_ports=["Y0_0", "Y1_0", "Y2_0", "Y3_0", "Y0_1", "Y1_1", "Y2_1", "Y3_1"])
    print("wrote sample trace CSVs")


if __name__ == "__main__":
    main()
