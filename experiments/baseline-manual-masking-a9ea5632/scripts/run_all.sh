#!/usr/bin/env bash
# End-to-end reproduction of this experiment. Run from the code/ directory.
set -e
export PATH=$PATH:/home/claudeuser/.local/bin
cd "$(dirname "$0")/.."

echo "[1/6] generating Verilog RTL from bench_specs.py"
python3 codegen_verilog.py

echo "[2/6] formal equivalence checking (Yosys SAT, masked vs baseline)"
bash scripts/run_eqcheck.sh scripts/eqcheck_results.txt

echo "[3/6] synthesizing for area (Yosys, GE)"
python3 synth_area.py

echo "[4/6] TVLA + power analysis (Python cycle-accurate model, 5 seeds x 10M traces/benchmark)"
python3 tvla_power.py

echo "[5/6] generating sample VCD waveforms"
python3 vcd_writer.py

echo "[6/6] assembling results.json + figures (+ wandb logging if WANDB_API_KEY set)"
python3 assemble_results.py

echo "done. See /workspace/output/results.json, /workspace/output/report.md, /workspace/output/figures/"
