import sys, os
import pprint
import pytest
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
try:
    import tabla
except:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from tabla.compiler.backend import Component, Schedule, TablaTemplate
from tabla.simulation.simulator import Simulator
from pathlib import Path
from tabla.compiler.compiler import compile
import shutil
import argparse
DFG_ROOT = f"{Path(f'{__file__}').parent}/../benchmarks/dfgs/polymath_generated"
BENCH_ROOT = f"{Path(f'{__file__}').parent}/../benchmarks/compilation_output"
CFG_ROOT = f"{Path(f'{__file__}').parent}/../simulation/configs"

def test():
    import numpy as np
    m = 325
    n = 139
    x = np.arange(m)
    y = np.arange(n)
    out = np.arange(m*n).reshape(m, n)

    for i in range(m):
        for j in range(n):
            out[i, j] = x[i] * y[j]


def simulate_benchmark(bench_name, cfg_name, debug=False):

    benchmark_path = Path(f"{BENCH_ROOT}/{bench_name}").resolve()
    cfg_path = Path(f"{CFG_ROOT}/{cfg_name}").resolve()
    simulator = Simulator(benchmark_path, cfg_path, debug)

    # simulator.only_debug_pu(3)

    #simulator.run_cycles(51)
    simulator.run()

    simulator.print_statistics()

def run_benchmark(package_name, gen_schedule, gen_mem_instr, is_training_algorithm, cfg_path=None):
    Component.reset_ids()

    dfg_name = f"{package_name}.json"

    optimizations = {'reorder_instr': True, 'unused_ni_opt': True, 'apply_reuse': True}
    if cfg_path is None:
        cfg_path = f'config.json'


    if Path(f"{DFG_ROOT}/../../../compilation_output/{package_name}").exists():
        shutil.rmtree(f"{DFG_ROOT}/../../../compilation_output/{package_name}")
    compile(Path(f"{DFG_ROOT}/{dfg_name}").resolve(), cfg_path,
            f"{package_name}_input_data.txt",
            f"{package_name}_input_weights.txt",
            "meta.txt", sort_alg="custom",
            gen_sched_file=gen_schedule,
            gen_mem_instr=gen_mem_instr,
            save_data=True,
            debug=False,
            optimizations=optimizations,
            show_ns_utilization=["NI", "NW", "ND"],
            is_training_algorithm=is_training_algorithm)  #

@pytest.mark.parametrize('benchmark, feature_size, pus, pes',[
    # ('linear', [55], 8, 8),
    ('svm_wifi', [10, 5], 8, 8)
    # ('reco', [54, 54, 3], 8, 8)
])
def test_get_graph_attr(benchmark, feature_size, pus, pes):
    # Reset components in case of multipel tests
    Component.reset_ids()

    # Create file names
    feature_size = [str(f) for f in feature_size]
    package_name = f"{benchmark}_{'_'.join(feature_size)}"
    cfg_path = f"{Path(f'{__file__}').parent}/configs/config_{pus}_{pes}.json"

    with open(cfg_path, "r") as read_file:
        config_data = json.load(read_file)
    dfg_name = f"{package_name}.json"
    package_path = f"{DFG_ROOT}/{dfg_name}"

    # Create Architecture
    new_arch = TablaTemplate(config_data)

    # Create schedule
    sched = Schedule(new_arch, debug=False, progress_bar=False, is_training_algorithm=True)

    # Get graph attributes
    sched.get_dfg_attrs(package_path)



@pytest.mark.parametrize('benchmark, feature_size, pus, pes, simulate',[
    # ('fft', [8], 8, 8, True),
    ('svm_wifi_inf', [30, 20], 8, 8, True)
    # ('reco', [54, 54, 3], 8, 8)
])
def test_benchmark(benchmark, feature_size, pus, pes, simulate):
    gen_sched = True
    no_mem = True
    inf_alg = False

    feature_size = [str(f) for f in feature_size]
    package_name = f"{benchmark}_{'_'.join(feature_size)}"
    cfg_name = f"config_{pus}_{pes}.json"
    cfg_path = f"{Path(f'{__file__}').parent}/configs/{cfg_name}"
    run_benchmark(package_name, gen_sched, no_mem, not inf_alg, cfg_path)
    if simulate:
        simulate_benchmark(package_name, cfg_name, debug=False)


if __name__ == "__main__":
    gen_sched = False
    no_mem = True
    inf_alg = False
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '/'))

    argparser = argparse.ArgumentParser(description='Memory Interface Instructino Generator')
    argparser.add_argument('-b', '--benchmark', required=True,
                           help='Name of the benchmark to create. One of "logistic", "linear", "reco",'
                                'or "svm".')
    argparser.add_argument('-fs', '--feature_size', nargs='+', required=True,
                           help='Feature size to use for creating the benchmark')
    argparser.add_argument('-gs', '--generate_sched_json', default=False, action='store_true',
                           help='Generate schedule json file')
    argparser.add_argument('-nm', '--no_mem', default=True, action='store_false',
                           help='Generate schedule json file')
    argparser.add_argument('-cfg', '--config', nargs='+', required=True,
                           help='PE/PU config')
    argparser.add_argument('-ia', '--inference_algorithm', default=False, action='store_true',
                           help='Whether or not the benchmark is an inference algorithm')

    args = argparser.parse_args()
    assert len(args.config) == 2
    pus = args.config[0]
    pes = args.config[1]
    feature_size = [str(f) for f in args.feature_size]
    package_name = f"{args.benchmark}_{'_'.join(feature_size)}"
    cfg_path = f"{Path(f'{__file__}').parent}/configs/config_{pus}_{pes}.json"
    run_benchmark(package_name, gen_sched, no_mem, not inf_alg, cfg_path)



