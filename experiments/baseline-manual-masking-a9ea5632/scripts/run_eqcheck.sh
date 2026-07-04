#!/usr/bin/env bash
# Formal equivalence check: prove masked_comb (recombined) == base_comb for ALL
# values of the random masks, for every benchmark. Uses yosys's SAT engine
# (no ABC dependency, which crashes in this sandboxed WASM build).
set -e
export PATH=$PATH:/home/claudeuser/.local/bin
cd "$(dirname "$0")/../benchmarks"
BENCHES="and_reduce4 mux_select parity_and_mask comparator_eq popcount_and_gate"
RESULTS_FILE="${1:-/tmp/eqcheck_results.txt}"
: > "$RESULTS_FILE"
for b in $BENCHES; do
  echo "=== equivalence check: $b ===" | tee -a "$RESULTS_FILE"
  yowasp-yosys -p "
    read_verilog masked_and_gadget_comb.v
    read_verilog generated/${b}.v
    prep -top ${b}_eqcheck
    flatten
    select -module ${b}_eqcheck
    sat -prove mismatch 0
  " > "/tmp/eqcheck_${b}.log" 2>&1
  if grep -q "SAT proof finished - no model found: SUCCESS" "/tmp/eqcheck_${b}.log"; then
    echo "PASS: ${b}: masked design is formally equivalent to baseline (mismatch=0 for all masks)" | tee -a "$RESULTS_FILE"
  else
    echo "FAIL: ${b}: formal equivalence NOT proven -- see /tmp/eqcheck_${b}.log" | tee -a "$RESULTS_FILE"
    tail -30 "/tmp/eqcheck_${b}.log" | tee -a "$RESULTS_FILE"
  fi
done
