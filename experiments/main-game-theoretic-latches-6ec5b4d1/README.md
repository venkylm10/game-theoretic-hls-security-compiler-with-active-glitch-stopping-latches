# main-game-theoretic-latches

Game-theoretic multi-objective ILP scheduler for HLS that chooses, per
dataflow node, between a full register and a dynamically-gated active
glitch-stopping latch, compiles 6 pinned dataflow benchmarks to synthesizable
Verilog, and validates them with real synthesis, real simulation, and a
10M-trace TVLA leakage test.

## Hardware/software requirements

- No GPU required.
- Python 3.11+.
- EDA toolchain: [Yosys](https://github.com/YosysHQ/yosys) (synthesis) and
  [Icarus Verilog](https://github.com/steveicarus/iverilog) (simulation). The
  pod this ran on had no root access, so the toolchain was the prebuilt
  [oss-cad-suite](https://github.com/YosysHQ/oss-cad-suite-build) tarball:
  ```bash
  curl -sL -o oss-cad-suite.tgz \
    https://github.com/YosysHQ/oss-cad-suite-build/releases/download/2026-07-03/oss-cad-suite-linux-x64-20260703.tgz
  tar xzf oss-cad-suite.tgz
  export PATH=$PWD/oss-cad-suite/bin:$PATH
  ```
  If you have root, `apt install yosys iverilog` works too, in which case
  edit `TOOLBIN` in `toolchain.py` (or just leave the tools on `$PATH` and
  drop the `TOOLBIN`-prefixed calls).

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
export PATH=/path/to/oss-cad-suite/bin:$PATH
python3 run_experiment.py       # ~2 minutes: schedules, synthesizes, simulates,
                                 # runs TVLA for all 6 benchmarks x 5 seeds
python3 assemble_manifest.py    # builds results.json + figures/*.png from
                                 # run_experiment.py's raw_results.json
```

Random seed: `[0, 1, 2, 3, 4]` (not pinned by the platform for this run — see
`../STATE.md` for why these were chosen). Benchmarks are defined in
`benchmarks.py` (no external dataset — this experiment has none).

## Files

- `dfg.py` — dataflow-graph core: node representation, generic golden
  evaluator, share-recombination analysis.
- `benchmarks.py` — the 6 pinned benchmarks.
- `scheduler.py` — the ILP scheduler (PuLP/CBC).
- `rtlgen.py` — Verilog RTL + self-checking testbench generation.
- `toolchain.py` — Yosys/Icarus Verilog process wrappers.
- `power_model.py` — VCD-derived Hamming-distance dynamic power model.
- `tvla.py` — vectorized (numpy) 10M-trace TVLA leakage assessment.
- `run_experiment.py` — orchestrates everything above, writes
  `raw_results.json`.
- `assemble_manifest.py` — turns `raw_results.json` into `results.json` +
  `figures/*.png`.
