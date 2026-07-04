"""Wrappers around the real EDA toolchain (Yosys synthesis, Icarus Verilog
simulation). No pod root access was available (apt install of yosys/
verilator failed on the dpkg lock), so the toolchain is the prebuilt
YosysHQ `oss-cad-suite` tarball extracted to /workspace/tools/oss-cad-suite
(see STATE.md). That bundle has no bundled C/C++ compiler+make, which
Verilator's --binary flow needs, so simulation uses Icarus Verilog
(iverilog+vvp) instead -- a self-contained bytecode simulator that needs no
external compiler. Yosys itself needs no external compiler either (it's a
single static-ish binary), so synthesis is unaffected by that gap.
"""
import os
import re
import subprocess

TOOLBIN = "/workspace/tools/oss-cad-suite/bin"


def _env():
    e = os.environ.copy()
    e['PATH'] = f"{TOOLBIN}:{e.get('PATH','')}"
    return e


def synthesize(rtl_path, top_module, report_path, lut_size=4):
    """Run yosys generic synth -> LUT-N mapping. Returns dict with area_luts
    and sequential-cell breakdown (dff vs dlatch counts), parsed from the
    `stat` output. Writes the full yosys log to report_path."""
    script = (
        f"read_verilog {rtl_path}; "
        f"synth -flatten -top {top_module}; "
        f"abc -lut {lut_size}; "
        f"stat"
    )
    proc = subprocess.run(
        [f"{TOOLBIN}/yosys", "-p", script],
        capture_output=True, text=True, env=_env(), timeout=120,
    )
    with open(report_path, 'w') as f:
        f.write(proc.stdout)
        f.write(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"yosys failed for {top_module}: {proc.stderr[-2000:]}")

    text = proc.stdout
    def count(cellname):
        m = re.search(rf"^\s*(\d+)\s+{re.escape(cellname)}\s*$", text, re.MULTILINE)
        return int(m.group(1)) if m else 0

    n_lut = count("$lut")
    n_dff = sum(count(c) for c in ("$_SDFF_PP0_", "$_DFF_P_", "$_DFFE_PP_", "$_SDFFE_PP0P_"))
    n_dlatch = sum(count(c) for c in ("$_DLATCH_P_", "$_DLATCH_N_"))
    m_cells = re.search(r"^\s*(\d+)\s+cells\s*$", text, re.MULTILINE)
    n_cells = int(m_cells.group(1)) if m_cells else None

    return {
        'area_luts': n_lut,
        'n_dff': n_dff,
        'n_dlatch': n_dlatch,
        'n_cells_total': n_cells,
        'raw_log_path': report_path,
    }


def simulate(dut_path, tb_path, work_dir, sim_name, vcd_name=None, timeout_s=60):
    """Compile with Icarus Verilog and run. Returns dict with pass/fail/total
    parsed from the testbench's EQUIV_RESULT line, plus the VCD path (the
    testbench itself calls $dumpfile/$dumpvars into work_dir/vcd_name).
    `vcd_name` MUST be given explicitly (matching the name passed to
    generate_testbench) whenever work_dir may already contain other .vcd
    files (e.g. from a previous seed) -- globbing for "the" .vcd in the
    directory is ambiguous and silently returns the wrong file."""
    os.makedirs(work_dir, exist_ok=True)
    sim_bin = os.path.join(work_dir, f"{sim_name}.vvp")
    compile_proc = subprocess.run(
        [f"{TOOLBIN}/iverilog", "-g2012", "-o", sim_bin, dut_path, tb_path],
        capture_output=True, text=True, env=_env(), timeout=timeout_s, cwd=work_dir,
    )
    if compile_proc.returncode != 0:
        raise RuntimeError(f"iverilog compile failed for {sim_name}: {compile_proc.stderr[-2000:]}")

    run_proc = subprocess.run(
        [f"{TOOLBIN}/vvp", sim_bin],
        capture_output=True, text=True, env=_env(), timeout=timeout_s, cwd=work_dir,
    )
    m = re.search(r"EQUIV_RESULT pass=(\d+) fail=(\d+) total=(\d+)", run_proc.stdout)
    if not m:
        raise RuntimeError(f"no EQUIV_RESULT line for {sim_name}: stdout={run_proc.stdout[-2000:]} stderr={run_proc.stderr[-1000:]}")
    if vcd_name is not None:
        vcd_path = os.path.join(work_dir, vcd_name)
        if not os.path.exists(vcd_path):
            raise RuntimeError(f"expected VCD {vcd_path} not produced by {sim_name}")
    else:
        vcd_candidates = [f for f in os.listdir(work_dir) if f.endswith('.vcd')]
        vcd_path = os.path.join(work_dir, vcd_candidates[0]) if vcd_candidates else None
    return {
        'pass': int(m.group(1)),
        'fail': int(m.group(2)),
        'total': int(m.group(3)),
        'vcd_path': vcd_path,
        'stdout_tail': run_proc.stdout[-500:],
    }
