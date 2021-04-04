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
DFG_ROOT = f"{Path(f'{__file__}').parent}/dfgs/polymath_generated"
BENCH_ROOT = f"{Path(f'{__file__}').parent}/compilation_output"
CFG_ROOT = f"{Path(f'{__file__}').parent}/../simulation/configs"



def simulate_benchmark(bench_name, cfg_name, debug=False):

    benchmark_path = Path(f"{BENCH_ROOT}/{bench_name}").resolve()
    cfg_path = Path(f"{CFG_ROOT}/{cfg_name}").resolve()
    simulator = Simulator(benchmark_path, cfg_path, debug)

    simulator.run()

    simulator.print_statistics()


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
    cfg_name = f"config_{pus}_{pes}.json"
    cfg_path = f"{Path(f'{__file__}').parent}/configs/config_{pus}_{pes}.json"
    simulate_benchmark(package_name, cfg_name, debug=False)
