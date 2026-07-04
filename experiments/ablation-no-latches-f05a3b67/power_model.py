"""Dynamic power from real switching activity: parse a Verilator/Icarus VCD
and apply a Hamming-distance (bit-toggle) energy model.

Per-toggle energy constants (assumed, documented, NOT measured silicon
values -- provenance is explicitly "assumed" wherever these feed
results.json):
  - comb/LUT-driven nets (the case-muxed ALU logic): data-dependent
    switching only.
  - register (flip-flop) nets: pay a clock-network toggle on every cycle
    regardless of data, on top of data toggles -- heavier.
  - active glitch-stopping latch nets: only toggle on their own designated
    transfer cycle (no free-running clock input) -- lighter, consistent
    with the area proxy table in scheduler.py and the paper's hypothesis
    that latches reduce both area AND power relative to full registers.
"""
import re

C_LUT_PJ = 0.5       # pJ per bit-toggle, combinational ALU mux logic
C_REGISTER_PJ = 0.9  # pJ per bit-toggle, edge-triggered flip-flop
C_LATCH_PJ = 0.6     # pJ per bit-toggle, dynamically-gated active latch
CLOCK_FREQ_HZ = 100e6  # pinned target clock (methodology-review fix: was unspecified)


def parse_vcd(path):
    """Minimal VCD parser. Returns (signals, n_clk_edges) where signals maps
    leaf signal name -> {'toggle_bits': int (sum of Hamming distances across
    all value changes), 'width': int}."""
    id_to_name = {}
    id_to_width = {}
    signals = {}
    prev_val = {}
    n_clk_edges = 0
    clk_ids = set()

    with open(path, 'r', errors='ignore') as f:
        in_defs = True
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('$var'):
                # $var wire 8 ! sig_x0 [7:0] $end   (fields can vary in count)
                parts = line.split()
                width = int(parts[2])
                vid = parts[3]
                name = parts[4]
                id_to_name[vid] = name
                id_to_width[vid] = width
                signals.setdefault(name, {'toggle_bits': 0, 'width': width})
                prev_val.setdefault(vid, 0)
                if name == 'clk':
                    clk_ids.add(vid)
                continue
            if line.startswith('$enddefinitions'):
                in_defs = False
                continue
            if in_defs:
                continue
            if line.startswith('#'):
                continue
            if line.startswith('b'):
                # bNNNN <id>
                sp = line[1:].split()
                if len(sp) != 2:
                    continue
                bits, vid = sp
                if vid not in id_to_name:
                    continue
                try:
                    val = int(bits, 2)
                except ValueError:
                    continue
                old = prev_val.get(vid, 0)
                toggled = old ^ val
                signals[id_to_name[vid]]['toggle_bits'] += bin(toggled).count('1')
                prev_val[vid] = val
            elif line[0] in '01xz':
                val = 1 if line[0] == '1' else 0
                vid = line[1:]
                if vid not in id_to_name:
                    continue
                old = prev_val.get(vid, 0)
                if vid in clk_ids and old == 0 and val == 1:
                    n_clk_edges += 1
                if val in (0, 1):
                    toggled = old ^ val
                    signals[id_to_name[vid]]['toggle_bits'] += bin(toggled).count('1')
                    prev_val[vid] = val
    return signals, n_clk_edges


def classify_signal(name, storage):
    """name -> 'lut' | 'register' | 'latch' | None (ignored: clk/rst/ports)."""
    if name in ('clk', 'rst', 'cycle_cnt'):
        return 'register'  # counter is a plain register, toggles every cycle
    if name.startswith('alu_'):
        return 'lut'
    if name.startswith('sig_'):
        nid = name[len('sig_'):]
        return storage.get(nid, None)
    return None  # p_*/o_* ports mirror sig_ values, don't double-count


def compute_dynamic_power_mw(vcd_path, storage, clock_freq_hz=CLOCK_FREQ_HZ):
    """Real measured dynamic power (mW) from one VCD run, using the actual
    per-signal bit-toggle counts and the classification above."""
    signals, n_clk_edges = parse_vcd(vcd_path)
    energy_pj = 0.0
    breakdown = {'lut': 0, 'register': 0, 'latch': 0}
    for name, info in signals.items():
        kind = classify_signal(name, storage)
        if kind is None:
            continue
        c = {'lut': C_LUT_PJ, 'register': C_REGISTER_PJ, 'latch': C_LATCH_PJ}[kind]
        energy_pj += info['toggle_bits'] * c
        breakdown[kind] += info['toggle_bits']
    if n_clk_edges == 0:
        return {'dynamic_power_mw': None, 'energy_pj': energy_pj, 'n_clk_edges': 0, 'toggle_breakdown': breakdown}
    total_time_ns = n_clk_edges * (1e9 / clock_freq_hz)
    power_mw = energy_pj / total_time_ns  # pJ/ns == mW
    return {
        'dynamic_power_mw': power_mw,
        'energy_pj': energy_pj,
        'n_clk_edges': n_clk_edges,
        'toggle_breakdown': breakdown,
    }
