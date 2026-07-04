# baseline-manual-masking

Naive HLS with manual masking, applied to 5 non-crypto dataflow benchmarks.
Measures latency (primary), area, power, and TVLA pass/fail, establishing the
"Manual-Masked-HLS" baseline subject pinned in `TASK.yaml`.

## Hardware requirements

None. No GPU, no `torch`. Everything runs on CPU with a WASM-compiled Yosys
(no root/apt-get required) and pure-Python/numpy simulation.

## Install

```bash
pip install -r requirements.txt
export PATH=$PATH:$HOME/.local/bin   # yowasp-yosys installs its CLI wrapper here
```

## Reproduce everything

```bash
bash scripts/run_all.sh
```

This runs, in order: `codegen_verilog.py` (emit RTL) -> `scripts/run_eqcheck.sh`
(formal equivalence check) -> `synth_area.py` (gate-level area) ->
`tvla_power.py` (TVLA + power) -> `vcd_writer.py` (sample waveforms) ->
`assemble_results.py` (final `results.json` + figures + W&B logging).

Random seed: `[0, 1, 2, 3, 4]` (5 seeds, `TASK.yaml`'s `subject.n_seeds=5`;
`subject.seeds` was empty so we pinned our own 0..4 for reproducibility).

Dataset: none (`TASK.yaml.datasets: []`). The 5 "arbitrary dataflow" benchmarks
are self-contained synthetic dataflow blocks defined in `bench_specs.py`
(4-way AND-reduce / bitmask select-mux / parity-checksum / equality-comparator
/ Hamming-popcount), each a plausible non-crypto HLS kernel with a bitwise-AND
nonlinearity that needs masking.

## What's here

- `bench_specs.py` -- single source of truth: the 5 benchmark dataflow graphs.
- `codegen_verilog.py` -- emits Verilog RTL (`benchmarks/generated/*.v`):
  baseline (unmasked) and manually-masked (2-share Trichina AND gadget)
  variants, both combinational (for formal equivalence checking) and clocked
  (for gate-level area synthesis), plus an `_eqcheck` wrapper module per
  benchmark.
- `benchmarks/masked_and_gadget.v` / `masked_and_gadget_comb.v` -- the reusable
  Trichina (2003) register-based masked-AND gadget (clocked / combinational).
- `scripts/run_eqcheck.sh` -- formal equivalence check: Yosys SAT proves
  `mismatch == 0` for ALL values of the random masks, i.e. masking didn't
  change the function computed (the correctness/equivalence check the
  methodology reviewer asked for).
- `synth_area.py` -- real Yosys synthesis (`synth -noabc`, flattened) -> gate
  histogram -> Gate Equivalents (GE), for both variants of all 5 benchmarks.
- `sim_model.py` -- vectorized numpy cycle-accurate model of the exact same
  pipeline the RTL describes (see file docstring for why: no root/no C
  compiler in this sandbox rules out Verilator/Icarus, and the WASM Yosys
  build ships without the `sim` command). Verified to functionally match the
  baseline RTL for all 5 benchmarks (see `scripts/run_all.sh` output).
- `tvla_power.py` -- fixed-vs-random TVLA (Welch's t-test) over the protected
  register window, and a Hamming-distance/toggle-count power proxy; 5 seeds x
  5,000,000 traces/set/benchmark (10M total per benchmark per seed).
- `vcd_writer.py` -- dumps real VCD waveforms for a handful of sample traces
  per benchmark to `../simulation_vcd/`.
- `assemble_results.py` -- builds the enriched `results.json` manifest,
  renders both required figures FROM the manifest values, logs to W&B.

## Known limitations (see report.md for the full self-critique)

- Verilator/Icarus Verilog were not installable in this sandbox (no root, no
  apt-get, no C/C++ compiler) and the WASM Yosys build lacks the `sim`
  command, so dynamic simulation (VCD/power/TVLA/latency) uses a hand-written
  Python cycle-accurate model instead of a real Verilog simulator. Static
  synthesis (area) and formal equivalence checking DO use real Yosys.
- Trace count is 10,000,000 per benchmark per seed (matching the hypothesis's
  "up to 10M traces" target) but power is a toggle-count proxy, not a real
  gate-level SPICE/PrimeTime power measurement -- there is no foundry PDK
  available in this sandbox.
