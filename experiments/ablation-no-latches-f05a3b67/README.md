# ablation-no-latches

Ablation of `main-game-theoretic-latches`: disables the ILP's active
glitch-stopping latch option so every storage element falls back to a full
edge-triggered register, then compiles the same pinned 6-benchmark suite,
synthesizes with real Yosys, simulates with real Icarus Verilog, and runs the
10M-trace TVLA leakage assessment -- to isolate the latency/area/power
contribution of the latches, holding the scheduler, benchmark suite, and
security methodology fixed.

## Hardware/software requirements

- No GPU required.
- Python 3.11+.
- EDA toolchain: [Yosys](https://github.com/YosysHQ/yosys) (synthesis) and
  [Icarus Verilog](https://github.com/steveicarus/iverilog) (simulation). This
  pod had no root access, so the toolchain is the prebuilt
  [oss-cad-suite](https://github.com/YosysHQ/oss-cad-suite-build) tarball:
  ```bash
  curl -sL -o oss-cad-suite.tgz \
    https://github.com/YosysHQ/oss-cad-suite-build/releases/download/2026-07-03/oss-cad-suite-linux-x64-20260703.tgz
  tar xzf oss-cad-suite.tgz -C /workspace/tools/
  ```
  `toolchain.py` hardcodes `TOOLBIN = "/workspace/tools/oss-cad-suite/bin"`;
  edit that if you extract elsewhere.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
export PATH=/workspace/tools/oss-cad-suite/bin:$PATH
python3 run_experiment.py       # ~2.5 minutes: schedules (register-only),
                                 # synthesizes, simulates, runs TVLA for all
                                 # 6 benchmarks x 5 seeds + a 7-point Pareto
                                 # sweep per benchmark
python3 assemble_manifest.py    # builds results.json + figures/*.png from
                                 # run_experiment.py's raw_results.json,
                                 # comparing against main_experiment_raw_results.json
```

Random seed: `[0, 1, 2, 3, 4]` -- not pinned by `TASK.yaml.subject` for this
run (`seeds: []`), so we reuse the exact same 5 seeds the parent experiment
used, to keep per-seed power samples directly comparable rather than
introducing an extra source of variance. Benchmarks are defined in
`benchmarks.py` (copied unmodified from the parent experiment -- no external
dataset; do not add/remove/rename without updating report.md).

## Files

- `dfg.py`, `benchmarks.py`, `rtlgen.py`, `toolchain.py`, `power_model.py`,
  `tvla.py` -- copied **unmodified** from `main-game-theoretic-latches` so
  that any measured difference is attributable solely to the storage-element
  formulation, not to a silent change elsewhere in the pipeline.
- `scheduler.py` -- the one file that differs from the parent: adds a
  `force_register_only` parameter that (a) constrains every storage node's
  `use_latch` binary to 0, and (b) fixes a genuine timing bug this ablation
  exposed -- primary-input availability (`avail['in']`) was hardcoded to 0 in
  the parent, which is only correct for latch-backed inputs (transparent
  same-cycle); a register-backed input needs `avail=1` (available only from
  the cycle after it's captured). A pre-run smoke test caught this via 0/20
  functional-equivalence failures on `fir4` before the fix.
- `run_experiment.py` -- orchestrates scheduling (`force_register_only=True`),
  RTL generation, synthesis, simulation, power measurement, and TVLA; writes
  `raw_results.json`.
- `main_experiment_raw_results.json` -- verbatim copy of the parent
  experiment's own raw measurement dump (pulled from its code/ output in the
  shared repo), used as the comparison baseline in `assemble_manifest.py` and
  the Pareto figure.
- `assemble_manifest.py` -- turns `raw_results.json` +
  `main_experiment_raw_results.json` into the enriched `results.json` +
  `figures/*.png`.
