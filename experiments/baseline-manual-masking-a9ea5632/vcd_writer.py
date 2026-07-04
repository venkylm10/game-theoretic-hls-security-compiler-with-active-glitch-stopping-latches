#!/usr/bin/env python3
"""
Minimal VCD (Value Change Dump) writer + driver that runs the Python
cycle-accurate model (sim_model.py) for a handful of representative trace
instances per benchmark and dumps real per-cycle register waveforms to
/workspace/output/simulation_vcd/<name>_masked.vcd and <name>_baseline.vcd.

These are genuine VCD files (IEEE 1364 textual format) produced by actually
executing the model, one clock edge per line, back-to-back for several
trace instances (with a reset pulse between them) -- viewable in gtkwave or
any VCD reader. They are the same per-cycle register data the TVLA/power
analysis (tvla_power.py) consumes at scale; this script just also renders it
as a waveform for a small, human-inspectable sample.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_specs import BENCHMARKS, ORDER
from sim_model import BenchmarkSim, random_secrets

OUT_DIR = "/workspace/output/simulation_vcd"
N_SAMPLE_TRACES = 4


class VcdWriter:
    def __init__(self, f, timescale="1ns"):
        self.f = f
        self.ids = {}
        self._next_id = ord("!")
        f.write(f"$timescale {timescale} $end\n")
        f.write("$scope module top $end\n")

    def add_var(self, name, width):
        vid = chr(self._next_id)
        self._next_id += 1
        self.ids[name] = vid
        self.f.write(f"$var wire {width} {vid} {name} $end\n")
        return vid

    def end_definitions(self):
        self.f.write("$upscope $end\n$enddefinitions $end\n")

    def dump_values(self, t, values):
        """values: dict name -> (int_value, width)"""
        self.f.write(f"#{t}\n")
        for name, (val, width) in values.items():
            vid = self.ids[name]
            bits = format(int(val) & ((1 << width) - 1), f"0{width}b")
            self.f.write(f"b{bits} {vid}\n")


def dump_masked_vcd(name, spec, path):
    rng = np.random.default_rng(12345)
    sim = BenchmarkSim(name, rng)
    secrets = random_secrets(name, N_SAMPLE_TRACES, rng)
    total_stages = sim.masked_total_stages()

    with open(path, "w") as f:
        vcd = VcdWriter(f)
        var_names = []
        for inp in spec["inputs"]:
            for s in (0, 1):
                var_names.append((f"{inp}{s}", spec["width"]))
        for stage in spec["and_stages"]:
            for node in stage:
                var_names.append((f"{node['name']}0", node["width"]))
                var_names.append((f"{node['name']}1", node["width"]))
        out_width = {"direct": lambda: spec["and_stages"] and 8, }
        # determine output width the same way codegen/sim compute it
        from codegen_verilog import gen_output_combine
        _, ow = gen_output_combine(spec)
        var_names.append(("y", ow))
        var_names.append(("clk", 1))
        var_names.append(("rst", 1))
        for vn, w in var_names:
            vcd.add_var(vn, max(w, 1))
        vcd.end_definitions()

        t = 0
        for trace_i in range(N_SAMPLE_TRACES):
            trace_secrets = {k: v[trace_i:trace_i + 1] for k, v in secrets.items()}
            res = sim.run_masked(trace_secrets, 1, record_all_cycles=True)
            # reset pulse
            vcd.dump_values(t, {"clk": (0, 1), "rst": (1, 1)})
            t += 5
            vcd.dump_values(t, {"clk": (1, 1), "rst": (1, 1)})
            t += 5
            for cyc, snap in enumerate(res["snapshots"]):
                vals = {"clk": (0, 1), "rst": (0, 1)}
                for sig, arr in snap.items():
                    width = spec["width"] if not sig.startswith("y") else ow
                    for vn, w in var_names:
                        if vn == sig:
                            width = w
                    vals[sig] = (int(arr[0]), width if width else 1)
                vcd.dump_values(t, vals)
                t += 5
                vcd.dump_values(t, {"clk": (1, 1)})
                t += 5
    return path


def dump_baseline_vcd(name, spec, path):
    rng = np.random.default_rng(54321)
    sim = BenchmarkSim(name, rng)
    secrets = random_secrets(name, N_SAMPLE_TRACES, rng)

    with open(path, "w") as f:
        vcd = VcdWriter(f)
        var_names = [(inp, spec["width"]) for inp in spec["inputs"]]
        from codegen_verilog import gen_output_combine
        _, ow = gen_output_combine(spec)
        var_names.append(("y", ow))
        var_names.append(("clk", 1))
        var_names.append(("rst", 1))
        for vn, w in var_names:
            vcd.add_var(vn, max(w, 1))
        vcd.end_definitions()

        t = 0
        for trace_i in range(N_SAMPLE_TRACES):
            trace_secrets = {k: v[trace_i:trace_i + 1] for k, v in secrets.items()}
            res = sim.run_baseline(trace_secrets, 1)
            vcd.dump_values(t, {"clk": (0, 1), "rst": (1, 1)})
            t += 5
            vcd.dump_values(t, {"clk": (1, 1), "rst": (1, 1)})
            t += 5
            # cycle 1: inputs latch
            vals = {"clk": (0, 1), "rst": (0, 1)}
            for inp in spec["inputs"]:
                vals[inp] = (int(trace_secrets[inp][0]), spec["width"])
            vcd.dump_values(t, vals)
            t += 5
            vcd.dump_values(t, {"clk": (1, 1)})
            t += 5
            # cycle 2: output latch
            vcd.dump_values(t, {"clk": (0, 1), "y": (int(res["y"][0]), ow)})
            t += 5
            vcd.dump_values(t, {"clk": (1, 1)})
            t += 5
    return path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name in ORDER:
        spec = BENCHMARKS[name]
        mpath = os.path.join(OUT_DIR, f"{name}_masked.vcd")
        bpath = os.path.join(OUT_DIR, f"{name}_baseline.vcd")
        dump_masked_vcd(name, spec, mpath)
        dump_baseline_vcd(name, spec, bpath)
        print(f"wrote {mpath}, {bpath}")


if __name__ == "__main__":
    main()
