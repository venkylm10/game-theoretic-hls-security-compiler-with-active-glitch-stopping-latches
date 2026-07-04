"""
Generic bit-level netlist simulator for Yosys `write_json` output.

Supports the small set of technology-independent single-bit gate primitives
produced by `techmap; simplemap` (no ABC, no liberty): $_AND_, $_OR_, $_XOR_,
$_NOT_, $_MUX_, $_DFF_P_ (and the _N_ clock-polarity variant). Values for every
net are represented as numpy boolean arrays of shape (batch,), so an entire
batch of independent traces is evaluated in lockstep with vectorized numpy ops.

This exists because the pod has no verilator/iverilog/gcc toolchain available
(no root to apt-get install, no C/C++ compiler) but does have yosys (via the
yowasp WebAssembly build, pip-installable) and numpy. Simulating directly off
the synthesized gate-level JSON netlist means "compiled RTL" and "simulated
trace" refer to the exact same artifact, and lets millions of traces be
evaluated by vectorizing over the batch dimension instead of looping in
per-trace Python.
"""
import json
import numpy as np

COMB_GATE_EVAL = {
    "$_AND_": lambda a, b: a & b,
    "$_OR_": lambda a, b: a | b,
    "$_XOR_": lambda a, b: a ^ b,
    "$_NAND_": lambda a, b: ~(a & b),
    "$_NOR_": lambda a, b: ~(a | b),
    "$_XNOR_": lambda a, b: ~(a ^ b),
    "$_ANDNOT_": lambda a, b: a & (~b),
    "$_ORNOT_": lambda a, b: a | (~b),
}
UNARY_GATE_EVAL = {
    "$_NOT_": lambda a: ~a,
    "$_BUF_": lambda a: a,
}
DFF_TYPES = {"$_DFF_P_", "$_DFF_N_"}


class Netlist:
    """Loads one Yosys JSON module and precomputes a fixed topological
    evaluation order. Net values are supplied/read per call to `run_cycles`."""

    def __init__(self, json_path, top="sbox"):
        with open(json_path) as f:
            doc = json.load(f)
        mod = doc["modules"][top]
        self.ports = mod["ports"]
        self.cells = mod["cells"]

        # driver[net_id] = (cell_name, port) for the single cell that drives it;
        # 'PI' for a primary input net.
        self.driver = {}
        for pname, pdef in self.ports.items():
            if pdef["direction"] == "input":
                for b in pdef["bits"]:
                    self.driver[b] = ("PI", pname)

        self.dff_cells = []
        self.comb_cells = []
        for cname, cdef in self.cells.items():
            ctype = cdef["type"]
            if ctype in DFF_TYPES:
                self.dff_cells.append((cname, cdef))
            elif ctype in COMB_GATE_EVAL or ctype in UNARY_GATE_EVAL or ctype == "$_MUX_":
                self.comb_cells.append((cname, cdef))
            else:
                raise ValueError(f"Unsupported cell type {ctype} ({cname}); "
                                  f"synthesis must be restricted to techmap/simplemap primitives")
            q_port = "Q" if ctype in DFF_TYPES else "Y"
            for b in cdef["connections"].get(q_port, []):
                self.driver[b] = (cname, q_port)

        self._topo_sort()
        self._compile()

    def _topo_sort(self):
        # Kahn's algorithm over comb cells only. `known` seeds with values that
        # are available BEFORE any combinational cell runs this cycle: primary
        # inputs and DFF Q outputs (latched from the previous cycle). Every
        # other net must be reached by walking through comb cells in dependency
        # order — it must NOT be pre-seeded from self.driver (which includes
        # comb-cell outputs too and would make every net trivially "known").
        known = set()
        for pname, pdef in self.ports.items():
            if pdef["direction"] == "input":
                known.update(pdef["bits"])
        for cname, cdef in self.dff_cells:
            known.update(cdef["connections"]["Q"])

        remaining = list(self.comb_cells)
        order = []
        progress = True
        while remaining and progress:
            progress = False
            still = []
            for cname, cdef in remaining:
                ins = self._input_nets(cdef)
                if all(self._net_known(n, known) for n in ins):
                    order.append((cname, cdef))
                    for b in cdef["connections"]["Y"]:
                        known.add(b)
                    progress = True
                else:
                    still.append((cname, cdef))
            remaining = still
        if remaining:
            names = [c[0] for c in remaining[:5]]
            raise ValueError(f"Combinational loop or unresolved deps in netlist near cells: {names}")
        self.comb_order = order

    @staticmethod
    def _net_known(n, known):
        if isinstance(n, str):
            return True  # constant "0"/"1"/"x"/"z"
        return n in known

    @staticmethod
    def _input_nets(cdef):
        nets = []
        for pname, dirn in cdef["port_directions"].items():
            if dirn == "input":
                nets.extend(cdef["connections"][pname])
        return nets

    def input_ports(self):
        return [p for p, d in self.ports.items() if d["direction"] == "input"]

    def output_ports(self):
        return [p for p, d in self.ports.items() if d["direction"] == "output"]

    def clock_port(self):
        for p in self.ports:
            if p.lower() == "clk":
                return p
        raise ValueError("no clk port found")

    def new_state(self, batch):
        """All-False initial DFF register state."""
        return {cname: np.zeros(batch, dtype=bool) for cname, _ in self.dff_cells}

    def _const_array(self, bitchar, batch):
        val = bitchar == "1"
        return np.full(batch, val, dtype=bool)

    @staticmethod
    def _ref(bit):
        """A JSON connection bit is either a net id (int) or a constant
        character ('0'/'1'/'x'/'z'). Resolve constants to a plain Python bool
        once at compile time so the hot loop never touches strings; numpy
        broadcasts a scalar bool against an array natively so no per-call
        array materialization is needed for constant-driven inputs."""
        if isinstance(bit, str):
            return bit == "1"
        return bit

    def _compile(self):
        """Flatten self.comb_order into ONE ordered instruction list (kind tag
        + precomputed refs) so the per-cycle hot loop is tuple unpacking +
        dict lookups + a numpy call, with no per-cell function definitions or
        repeated dict/string indexing (the naive version spent most of its
        time re-defining an `rd()` closure and re-indexing
        `cdef["connections"][port]` 60k times/cycle). Instructions MUST stay
        in the single topological order from comb_order -- splitting into
        per-op-type passes would silently reorder dependent gates."""
        self._ops = []  # (kind, func_or_None, a_ref, b_ref_or_None, s_ref_or_None, y_net)
        for cname, cdef in self.comb_order:
            conn = cdef["connections"]
            ctype = cdef["type"]
            y_net = conn["Y"][0]
            if ctype in COMB_GATE_EVAL:
                self._ops.append(("bin", COMB_GATE_EVAL[ctype], self._ref(conn["A"][0]),
                                   self._ref(conn["B"][0]), None, y_net))
            elif ctype in UNARY_GATE_EVAL:
                self._ops.append(("un", UNARY_GATE_EVAL[ctype], self._ref(conn["A"][0]),
                                   None, None, y_net))
            elif ctype == "$_MUX_":
                self._ops.append(("mux", None, self._ref(conn["A"][0]), self._ref(conn["B"][0]),
                                   self._ref(conn["S"][0]), y_net))
            else:
                raise AssertionError(ctype)

    def eval_combinational(self, net_vals, batch):
        """net_vals: dict net_id(int) -> bool array (or bare bool for a const).
        Mutates net_vals in place, filling in every combinational net given PI
        + DFF-Q values already present. Instruction order matches the
        topological sort computed once in _compile()."""
        nv = net_vals
        for kind, func, a_ref, b_ref, s_ref, y_net in self._ops:
            a = nv[a_ref] if type(a_ref) is int else a_ref
            if kind == "bin":
                b = nv[b_ref] if type(b_ref) is int else b_ref
                nv[y_net] = func(a, b)
            elif kind == "un":
                nv[y_net] = func(a)
            else:  # mux
                b = nv[b_ref] if type(b_ref) is int else b_ref
                s = nv[s_ref] if type(s_ref) is int else s_ref
                nv[y_net] = np.where(s, b, a)

    def step(self, pi_values, state, batch):
        """One clock cycle: pi_values = {port_name: bool array}, state = dff state dict.
        Returns (net_vals, new_state, output_values dict)."""
        net_vals = {}
        for pname, arr in pi_values.items():
            bits = self.ports[pname]["bits"]
            if isinstance(arr, list):
                assert len(bits) == len(arr)
                for b, v in zip(bits, arr):
                    net_vals[b] = v
            else:
                assert len(bits) == 1
                net_vals[bits[0]] = arr
        for cname, cdef in self.dff_cells:
            q = cdef["connections"]["Q"][0]
            net_vals[q] = state[cname]

        self.eval_combinational(net_vals, batch)

        new_state = {}
        for cname, cdef in self.dff_cells:
            d = cdef["connections"]["D"][0]
            new_state[cname] = self._const_array(d, batch) if isinstance(d, str) else net_vals[d]

        outputs = {}
        for pname in self.output_ports():
            bits = self.ports[pname]["bits"]
            outputs[pname] = [net_vals[b] for b in bits]

        return net_vals, new_state, outputs

    def pipeline_depth(self):
        """Number of DFF stages on the longest PI->PO path (assumes a balanced,
        feed-forward pipeline as produced by the HLS retiming pass)."""
        # depth[net] = number of registers crossed so far to reach this net
        depth = {}
        for pname, pdef in self.ports.items():
            if pdef["direction"] == "input":
                for b in pdef["bits"]:
                    depth[b] = 0
        for cname, cdef in self.dff_cells:
            q = cdef["connections"]["Q"][0]
            d = cdef["connections"]["D"][0]
            depth.setdefault(q, 0)
        for cname, cdef in self.comb_order:
            conn = cdef["connections"]
            ins = self._input_nets(cdef)
            d = 0
            for n in ins:
                if isinstance(n, str):
                    continue
                d = max(d, depth.get(n, 0))
            depth[conn["Y"][0]] = d
        # relax through DFFs once (D depth -> Q depth+1); iterate a few times
        # since register chains are sequential in cell dict order already
        # captured through comb_order dependency, but D nets need one more hop
        for _ in range(len(self.dff_cells) + 1):
            changed = False
            for cname, cdef in self.dff_cells:
                d = cdef["connections"]["D"][0]
                q = cdef["connections"]["Q"][0]
                dv = depth.get(d, 0) if not isinstance(d, str) else 0
                if depth.get(q, -1) < dv + 1:
                    depth[q] = dv + 1
                    changed = True
            if changed:
                # re-propagate combinational levels since Q depths changed
                for cname, cdef in self.comb_order:
                    conn = cdef["connections"]
                    ins = self._input_nets(cdef)
                    d2 = 0
                    for n in ins:
                        if isinstance(n, str):
                            continue
                        d2 = max(d2, depth.get(n, 0))
                    y = conn["Y"][0]
                    if depth.get(y, -1) < d2:
                        depth[y] = d2
            else:
                break
        max_depth = 0
        for pname in self.output_ports():
            for b in self.ports[pname]["bits"]:
                max_depth = max(max_depth, depth.get(b, 0))
        return max_depth
