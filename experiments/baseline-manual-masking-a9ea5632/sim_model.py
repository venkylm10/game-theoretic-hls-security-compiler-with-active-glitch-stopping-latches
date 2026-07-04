#!/usr/bin/env python3
"""
Cycle-accurate Python model of the clocked RTL in benchmarks/generated/*.v.

Why a Python model instead of a Verilog simulator: this sandbox has no root
access (apt-get is blocked) and no C/C++ toolchain (no gcc/g++), so neither
Verilator nor Icarus Verilog can be installed, and the WASM build of Yosys
used for synthesis/equivalence-checking (yowasp-yosys) ships without the
`sim` command. To still get real per-cycle register activity for the power
proxy and TVLA analysis, this module re-implements the exact same register
pipeline described in bench_specs.py (input latch -> per-AND-stage masked
gadget registers -> output register) directly in vectorized numpy, so many
independent trace instances can be simulated at once. Functional correctness
of the *combinational* masked datapath (i.e. that masking didn't change the
function) is separately proven for real by Yosys formal equivalence checking
(scripts/run_eqcheck.sh) -- this module is only responsible for *timing and
per-cycle switching activity*, which the RTL's register-stage count fixes
unambiguously (see bench_specs docstring).

Power proxy: dynamic power is approximated as total register-bit toggle
count per cycle (Hamming distance between consecutive states of every
flip-flop that updates that cycle), i.e. a switching-activity/Hamming-distance
model -- the standard cheap proxy for CMOS dynamic power (P ~ alpha * C * V^2
* f, alpha = toggle rate) when no gate-level timing/capacitance data is
available.
"""
import numpy as np

from bench_specs import BENCHMARKS, ORDER


def popcount(x):
    return np.bitwise_count(x.astype(np.uint32)).astype(np.int64)


def resolve_ref_sim(ref, node_state, input_state, e0, e1):
    kind, val = ref
    if kind == "in":
        return input_state[f"{val}0"], input_state[f"{val}1"]
    if kind == "inv":
        return (~input_state[f"{val}0"]).astype(np.uint8), input_state[f"{val}1"]
    if kind == "node":
        return node_state[val][0], node_state[val][1]
    if kind == "bit":
        i = val
        return (e0 >> i) & 1, (e1 >> i) & 1
    raise ValueError(ref)


class BenchmarkSim:
    """Vectorized simulator for one benchmark, N independent trace instances at once."""

    def __init__(self, name, rng):
        self.name = name
        self.spec = BENCHMARKS[name]
        self.rng = rng

    def n_gadget_stages(self):
        return len(self.spec["and_stages"])

    def masked_total_stages(self):
        return 1 + self.n_gadget_stages() + 1

    def baseline_total_stages(self):
        return 2

    def _fresh_share(self, n, width):
        maxv = (1 << width) - 1
        return self.rng.integers(0, maxv + 1, size=n, dtype=np.uint16).astype(np.uint8 if width <= 8 else np.uint16)

    def run_masked(self, secrets, n, record_all_cycles=False):
        """secrets: dict input_name -> np.ndarray[n] true secret values (width per spec).
        Returns dict with:
          toggles: np.ndarray[n, total_stages]  toggle count per cycle per trace
          y: np.ndarray[n] final recombined output value
          cycle_snapshots (optional): list of dict signal->array for VCD dumping
        """
        spec = self.spec
        w = spec["width"]
        total_stages = self.masked_total_stages()
        toggles = np.zeros((n, total_stages), dtype=np.int64)
        snapshots = [] if record_all_cycles else None

        # --- stage 0: input share registers ---
        input_state = {}
        prev_regs = {}
        for name in spec["inputs"]:
            share1 = self._fresh_share(n, w)
            share0 = (secrets[name].astype(np.uint16) ^ share1.astype(np.uint16)).astype(np.uint8)
            input_state[f"{name}0"] = share0
            input_state[f"{name}1"] = share1
            prev_regs[f"{name}0"] = np.zeros(n, dtype=np.uint8)
            prev_regs[f"{name}1"] = np.zeros(n, dtype=np.uint8)
        stage_toggle = np.zeros(n, dtype=np.int64)
        for name in spec["inputs"]:
            stage_toggle += popcount(input_state[f"{name}0"] ^ prev_regs[f"{name}0"])
            stage_toggle += popcount(input_state[f"{name}1"] ^ prev_regs[f"{name}1"])
        toggles[:, 0] = stage_toggle
        if record_all_cycles:
            snapshots.append({k: v.copy() for k, v in input_state.items()})

        # optional linear preprocessing (comparator_eq's XNOR), computed combinationally
        # from the registered inputs -- not itself a register, so contributes no toggle
        e0 = e1 = None
        if "pre_xnor_width" in spec:
            e0 = (~(input_state["a0"] ^ input_state["b0"])).astype(np.uint8)
            e1 = (input_state["a1"] ^ input_state["b1"]).astype(np.uint8)

        node_state = {}
        cycle_idx = 1
        for stage in spec["and_stages"]:
            stage_toggle = np.zeros(n, dtype=np.int64)
            new_node_state = {}
            for node in stage:
                nw = node["width"]
                l0, l1 = resolve_ref_sim(node["lhs"], node_state, input_state, e0, e1)
                r0, r1 = resolve_ref_sim(node["rhs"], node_state, input_state, e0, e1)
                l0 = l0.astype(np.uint16); l1 = l1.astype(np.uint16)
                r0 = r0.astype(np.uint16); r1 = r1.astype(np.uint16)
                rand = self._fresh_share(n, nw).astype(np.uint16)
                z0 = ((l0 & r0) ^ rand).astype(np.uint8)
                z1 = ((l1 & r1) ^ (rand ^ (l0 & r1) ^ (l1 & r0))).astype(np.uint8)
                prev_z0 = node_state.get(node["name"], (np.zeros(n, dtype=np.uint8),))[0] \
                    if node["name"] in node_state else np.zeros(n, dtype=np.uint8)
                prev_z1 = node_state.get(node["name"], (None, np.zeros(n, dtype=np.uint8)))[1] \
                    if node["name"] in node_state else np.zeros(n, dtype=np.uint8)
                stage_toggle += popcount(z0 ^ prev_z0) + popcount(z1 ^ prev_z1)
                new_node_state[node["name"]] = (z0, z1)
            node_state.update(new_node_state)
            toggles[:, cycle_idx] = stage_toggle
            if record_all_cycles:
                snap = {}
                for nm, (a0v, a1v) in new_node_state.items():
                    snap[f"{nm}0"] = a0v.copy()
                    snap[f"{nm}1"] = a1v.copy()
                snapshots.append(snap)
            cycle_idx += 1

        # --- output combine + register ---
        y_masked = self._combine_output(spec, node_state)
        prev_y = np.zeros(n, dtype=np.uint16)
        out_toggle = popcount(y_masked.astype(np.uint32) ^ prev_y.astype(np.uint32))
        toggles[:, cycle_idx] = out_toggle
        if record_all_cycles:
            snapshots.append({"y": y_masked.copy()})

        return {
            "toggles": toggles,
            "y": y_masked,
            "protected_toggles": toggles[:, :-1],  # excludes the revealing output register
            "snapshots": snapshots,
        }

    def _combine_output(self, spec, node_state):
        o = spec["output"]
        if o["kind"] == "direct":
            n0, n1 = node_state[o["node"]]
            return (n0.astype(np.uint16) ^ n1.astype(np.uint16))
        if o["kind"] == "xor":
            acc = np.zeros_like(node_state[o["nodes"][0]][0], dtype=np.uint16)
            for nm in o["nodes"]:
                n0, n1 = node_state[nm]
                acc = acc ^ n0.astype(np.uint16) ^ n1.astype(np.uint16)
            return acc
        if o["kind"] == "parity":
            n0, n1 = node_state[o["node"]]
            v = n0.astype(np.uint16) ^ n1.astype(np.uint16)
            return popcount(v) & 1
        if o["kind"] == "popcount":
            n0, n1 = node_state[o["node"]]
            v = n0.astype(np.uint16) ^ n1.astype(np.uint16)
            return popcount(v).astype(np.uint16)
        raise ValueError(o)

    def run_baseline(self, secrets, n):
        spec = self.spec
        # stage0: input registers latch true (unmasked) values
        stage0_toggle = np.zeros(n, dtype=np.int64)
        for name in spec["inputs"]:
            stage0_toggle += popcount(secrets[name].astype(np.uint32))  # vs reset value 0
        y = self._baseline_function(spec, secrets)
        out_toggle = popcount(y.astype(np.uint32))
        toggles = np.stack([stage0_toggle, out_toggle], axis=1)
        return {"toggles": toggles, "y": y}

    def _baseline_function(self, spec, secrets):
        name = self.name
        if name == "and_reduce4":
            return (secrets["a"].astype(np.uint16) & secrets["b"] & secrets["c"] & secrets["d"])
        if name == "mux_select":
            sel = secrets["sel"].astype(np.uint16)
            a = secrets["a"].astype(np.uint16)
            b = secrets["b"].astype(np.uint16)
            return (sel & a) ^ ((~sel & 0xFF) & b)
        if name == "parity_and_mask":
            v = secrets["a"].astype(np.uint16) & secrets["b"]
            return popcount(v) & 1
        if name == "comparator_eq":
            a = secrets["a"].astype(np.uint16)
            b = secrets["b"].astype(np.uint16)
            eq_bits = (~(a ^ b)) & 0xFF
            return (popcount(eq_bits) == 8).astype(np.uint16)
        if name == "popcount_and_gate":
            v = secrets["a"].astype(np.uint16) & secrets["b"]
            return popcount(v).astype(np.uint16)
        raise ValueError(name)


def random_secrets(name, n, rng, fixed=None):
    spec = BENCHMARKS[name]
    w = spec["width"]
    secrets = {}
    for inp in spec["inputs"]:
        if fixed is not None:
            secrets[inp] = np.full(n, fixed, dtype=np.uint8)
        else:
            secrets[inp] = rng.integers(0, 256, size=n, dtype=np.uint16).astype(np.uint8)
    return secrets
