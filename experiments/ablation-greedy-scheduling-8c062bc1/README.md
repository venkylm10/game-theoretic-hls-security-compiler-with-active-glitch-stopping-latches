# ablation-greedy-scheduling

Ablation arm of the game-theoretic HLS security compiler research plan.
Disables the game-theoretic multi-objective ILP scheduler and replaces it
with textbook greedy resource-constrained list scheduling, while KEEPING
the active glitch-stopping latch mechanism (a different sibling ablation,
`ablation-no-latches`, is the one that removes latches). The
game-theoretic ILP scheduler is recomputed here on the identical
benchmark suite as the paired reference (see "Why the ILP reference is
recomputed in-run" below) so the two required figures
(`pareto_tradeoff_plot`, `scheduling_overhead`) can be produced directly.

## Hardware / software requirements

CPU only (`constraints.gpu_required: false` in TASK.yaml; no GPU or torch
used anywhere in this code). Any machine with Python 3.11 and ~1 GB RAM is
sufficient; the heaviest step (200k-trace vectorized TVLA simulation per
benchmark) runs in a few seconds with numpy.

```bash
pip install -r requirements.txt
```

## Commands

```bash
python train.py            # runs the full pipeline, writes greedy_rtl/,
                            # simulation_vcd/, synthesis_logs/, figures/,
                            # and code/pipeline_output.json
python eval.py              # re-runs the deterministic pipeline from
                            # scratch and checks it reproduces
                            # pipeline_output.json (regression check)
```

Random seed: all randomness (benchmark DAG generation in `hls/dfg.py`,
functional-equivalence fuzzing in `hls/functional_sim.py`, and the
power/TVLA trace generators in `hls/sim_core.py`) is driven by fixed,
hardcoded integer seeds, so both commands are fully deterministic.

Dataset: none (`TASK.yaml.subject.datasets` is empty). The benchmark
suite is 8 synthetic, non-cryptographic dataflow programs generated
in `hls/dfg.py::load_benchmark_suite()` (fixed seeds per benchmark),
each with two Boolean-masking-share primary inputs used to drive the
latch-placement and TVLA analysis.

## No physical EDA toolchain in this sandbox

This pod has no root/sudo and no `apt` access, so neither **Yosys**
(synthesis) nor **Verilator** (RTL simulation) could be installed. Rather
than silently fabricate "measured" synthesis/simulation numbers, this
experiment:

- Emits real, readable Verilog (`hls/rtl_gen.py`) from the internal
  schedule/binding/latch model, into `greedy_rtl/`.
- Emits a real VCD trace (`hls/cycle_sim.py`) from the SAME internal
  model (not from Verilator), into `simulation_vcd/`.
- Estimates area (gate-equivalents) and power (mW) from a documented,
  simplified cost model (`hls/area_model.py`, `hls/power_model.py`)
  instead of gate-level synthesis, marked `"provenance": "estimated"`
  throughout `results.json` (never `"measured"`).
- Runs a real, vectorized Welch's-t-test TVLA simulation
  (`hls/tvla.py`) over the same internal model.

This is the intended reading of TASK.yaml's own wording: "a simulated
TVLA metric" and "custom Python-based power models" -- both explicitly
named as simulation/estimation, not physical synthesis.

## Why the ILP reference is recomputed in-run

This experiment's `TASK.yaml.context.plan.this_experiment.depends_on`
lists `main-game-theoretic-latches`, but each experiment in this research
platform runs in its own ephemeral pod with no shared filesystem access
to a sibling pod's outputs. To produce the two required comparison
figures (Pareto area/latency plot, scheduling-overhead histogram) as a
fair, paired, apples-to-apples comparison, the game-theoretic ILP
scheduler (`hls/schedule_ilp.py`) is re-solved here, on the IDENTICAL
benchmark suite, using PuLP/CBC. It is reported in `results.json` as a
`baselines[]` entry with `provenance: "reproduced_run_id"` is NOT used
(no actual sibling run id is available) -- see report.md for the exact
provenance caveat recorded there.

## Units (fixing the reviewer-flagged unit drift)

Per the methodology reviewer's note in `TASK.yaml.context.plan.overview`,
area is reported ONLY in gate-equivalents (**GE**) and power ONLY in
**mW** everywhere in this experiment's outputs -- matching the units
pinned by `TASK.yaml.deliverables.results_json_schema` (`area_ge`,
`power_mw`), so this ablation's numbers are directly comparable to the
sibling experiments once assembled by the platform.

## Functional correctness check

Per the reviewer's second flagged gap ("without a correctness check, the
compiler could 'achieve' optimal latency/area by generating logically
broken or disconnected pipelines"), `hls/functional_sim.py` co-simulates
every benchmark's scheduled+bound architecture against a pure topological
(unscheduled) reference evaluator over 200 random input trials per
schedule, asserting bit-exact agreement AND asserting no read-before-write
(RAW) hazard on any shared register. `train.py` raises immediately if
this check fails for either scheduler variant -- results are never
reported for a benchmark that fails functional equivalence.

## Layout

```
code/
  train.py, eval.py           # entry points
  hls/
    dfg.py                    # benchmark suite + share-taint tracking
    schedule_greedy.py         # ablation-under-test scheduler
    schedule_ilp.py             # game-theoretic ILP reference scheduler
    latches.py                  # active glitch-stopping latch insertion
    binding.py                  # functional-unit resource binding
    functional_sim.py           # correctness / equivalence check
    rtl_gen.py                  # Verilog emission
    cycle_sim.py                # cycle-accurate sim + VCD writer
    area_model.py                # GE cost model + synthesis log writer
    power_model.py               # Hamming-weight power model
    tvla.py                       # simulated TVLA (Welch's t-test)
    plotting.py                   # figures, plotted from manifest values
    pipeline.py                   # orchestrates the above per benchmark
  pipeline_output.json           # raw per-benchmark + aggregate results
```
