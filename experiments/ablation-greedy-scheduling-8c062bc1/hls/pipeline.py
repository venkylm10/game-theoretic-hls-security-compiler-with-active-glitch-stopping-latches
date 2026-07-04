"""
End-to-end per-benchmark compilation pipeline shared by code/train.py and
code/eval.py: schedule (greedy AND the ILP reference) -> insert active
glitch-stopping latches -> bind functional units -> check functional
equivalence against the unscheduled reference -> estimate area (GE) and
power (mW) -> run simulated TVLA -> emit RTL / VCD / synthesis-log
artifacts for the greedy (under-test) arm.
"""
import json
import os

from .area_model import compute_area_ge, write_synthesis_log
from .binding import bind_units
from .dfg import load_benchmark_suite
from .functional_sim import check_equivalence
from .latches import insert_latches
from .power_model import estimate_power_mw
from .rtl_gen import generate_verilog
from .schedule_greedy import schedule_greedy
from .schedule_ilp import schedule_ilp
from .cycle_sim import write_vcd
from .tvla import run_tvla

WIDTH = 16
N_TRACES_POWER = 20_000
N_TRACES_TVLA = 200_000


def run_benchmark(bench, out_dirs=None, verbose=print):
    cycle_g, lat_g = schedule_greedy(bench)
    cycle_i, lat_i = schedule_ilp(bench, horizon=lat_g)

    eq_g = check_equivalence(bench, cycle_g, n_trials=200, seed=42, width=WIDTH)
    eq_i = check_equivalence(bench, cycle_i, n_trials=200, seed=43, width=WIDTH)
    if not eq_g["passed"]:
        raise RuntimeError(f"greedy schedule functional check FAILED for {bench.name}: {eq_g}")
    if not eq_i["passed"]:
        raise RuntimeError(f"ILP schedule functional check FAILED for {bench.name}: {eq_i}")

    latches_g = insert_latches(bench, cycle_g)
    latches_i = insert_latches(bench, cycle_i)

    unit_g = bind_units(bench, cycle_g)
    unit_i = bind_units(bench, cycle_i)

    area_g = compute_area_ge(bench, cycle_g, unit_g, latches_g, width=WIDTH)
    area_i = compute_area_ge(bench, cycle_i, unit_i, latches_i, width=WIDTH)

    power_g = estimate_power_mw(bench, cycle_g, area_g["total_ge"], n_traces=N_TRACES_POWER, width=WIDTH)
    power_i = estimate_power_mw(bench, cycle_i, area_i["total_ge"], n_traces=N_TRACES_POWER, width=WIDTH)

    tvla_g = run_tvla(bench, cycle_g, latches_g, n_traces=N_TRACES_TVLA, width=WIDTH, protect=True)
    tvla_g_unprotected_selftest = run_tvla(bench, cycle_g, latches_g, n_traces=N_TRACES_TVLA,
                                            width=WIDTH, protect=False)

    if out_dirs:
        rtl = generate_verilog(bench, cycle_g, unit_g, latches_g, width=WIDTH)
        with open(os.path.join(out_dirs["rtl"], f"{bench.name}.v"), "w") as f:
            f.write(rtl)
        write_vcd(os.path.join(out_dirs["vcd"], f"{bench.name}.vcd"), bench, cycle_g, latches_g, width=WIDTH)
        write_synthesis_log(os.path.join(out_dirs["synth"], f"{bench.name}.log"),
                             bench, cycle_g, unit_g, latches_g, area_g, lat_g, width=WIDTH)

    verbose(f"[{bench.name}] greedy: lat={lat_g} area={area_g['total_ge']:.1f}GE "
             f"power={power_g['power_mw']:.4f}mW latches={len(latches_g)} "
             f"tvla={tvla_g['max_abs_t']:.2f} | ilp: lat={lat_i} area={area_i['total_ge']:.1f}GE "
             f"latches={len(latches_i)}")

    return {
        "name": bench.name,
        "greedy": {
            "latency_cycles": lat_g, "area_ge": area_g["total_ge"], "area_breakdown": area_g,
            "power_mw": power_g["power_mw"], "power_detail": power_g,
            "n_latches": len(latches_g), "tvla_max_abs_t": tvla_g["max_abs_t"],
            "tvla_worst_cycle": tvla_g["worst_cycle"],
            "tvla_unprotected_selftest_max_abs_t": tvla_g_unprotected_selftest["max_abs_t"],
            "functional_check": eq_g,
        },
        "ilp": {
            "latency_cycles": lat_i, "area_ge": area_i["total_ge"], "area_breakdown": area_i,
            "power_mw": power_i["power_mw"], "power_detail": power_i,
            "n_latches": len(latches_i), "functional_check": eq_i,
        },
    }


def run_all(out_dirs=None, verbose=print, progress_cb=None):
    benches = load_benchmark_suite()
    records = []
    for idx, bench in enumerate(benches):
        records.append(run_benchmark(bench, out_dirs=out_dirs, verbose=verbose))
        if progress_cb:
            progress_cb(idx + 1, len(benches))
    return records


def aggregate(records):
    import statistics as st
    g_lat = [r["greedy"]["latency_cycles"] for r in records]
    g_area = [r["greedy"]["area_ge"] for r in records]
    g_pow = [r["greedy"]["power_mw"] for r in records]
    g_tvla = [r["greedy"]["tvla_max_abs_t"] for r in records]
    g_selftest = [r["greedy"]["tvla_unprotected_selftest_max_abs_t"] for r in records]
    i_lat = [r["ilp"]["latency_cycles"] for r in records]
    i_area = [r["ilp"]["area_ge"] for r in records]
    i_pow = [r["ilp"]["power_mw"] for r in records]

    return {
        "n_benchmarks": len(records),
        "greedy": {
            "latency_cycles_mean": st.mean(g_lat), "latency_cycles_values": g_lat,
            "area_ge_mean": st.mean(g_area), "area_ge_values": g_area,
            "power_mw_mean": st.mean(g_pow), "power_mw_values": g_pow,
            "tvla_max_abs_t_worst": max(g_tvla), "tvla_max_abs_t_values": g_tvla,
            "tvla_unprotected_selftest_worst": max(g_selftest),
        },
        "ilp": {
            "latency_cycles_mean": st.mean(i_lat), "latency_cycles_values": i_lat,
            "area_ge_mean": st.mean(i_area), "area_ge_values": i_area,
            "power_mw_mean": st.mean(i_pow), "power_mw_values": i_pow,
        },
        "latency_inflation_cycles": [g - i for g, i in zip(g_lat, i_lat)],
        "benchmarks": [r["name"] for r in records],
    }
