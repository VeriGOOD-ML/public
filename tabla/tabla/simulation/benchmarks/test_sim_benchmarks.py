import os, sys

try:
    import simulation
except ModuleNotFoundError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
import pytest
from simulation.simulator import Simulator
from simulation.instruction import InstructionLoader
from pathlib import Path
import argparse
BENCH_ROOT = f"{Path(f'{__file__}').parent}/../../compilation_output"
CFG_ROOT = f"{Path(f'{__file__}').parent}/../configs"

def simulate_benchmark(bench_name, cfg_name, debug=False):

    benchmark_path = Path(f"{BENCH_ROOT}/{bench_name}").resolve()
    cfg_path = Path(f"{CFG_ROOT}/{cfg_name}").resolve()
    simulator = Simulator(benchmark_path, cfg_path, debug)

    # simulator.only_debug_pu(3)

    #simulator.run_cycles(51)
    simulator.run()

    simulator.print_statistics()

@pytest.mark.parametrize('benchmark, feature_size, pus, pes', [
    # ('linear', [784], 8, 8),
    ('svm_wifi', [325, 139], 4, 64)
    # ('reco', [54, 54, 3], 8, 8)
])
def test_sim(benchmark, feature_size, pus, pes):

    debug = False
    feature_size = [str(f) for f in feature_size]
    if benchmark == "svm_wifi":
        package_name = f"{benchmark}_{'_'.join(feature_size)}_{pus}PU_{pes}PE"
    else:
        package_name = f"{benchmark}_{'_'.join(feature_size)}"
    # package_name = f"{benchmark}_{'_'.join(feature_size)}"
    cfg_path = f"config_{pus}_{pes}.json"
    simulate_benchmark(package_name, cfg_path, debug=debug)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description='Simulator testing')
    argparser.add_argument('-b', '--benchmark', required=True,
                           help='Name of the benchmark to create. One of "logistic", "linear", "reco",'
                                'or "svm".')
    argparser.add_argument('-fs', '--feature_size', nargs='+', required=True,
                           help='Feature size to use for creating the benchmark')

    argparser.add_argument('-cfg', '--config', nargs='+', required=True,
                           help='PE/PU config')

    args = argparser.parse_args()
    assert len(args.config) == 2
    pus = args.config[0]
    pes = args.config[1]
    feature_size = [str(f) for f in args.feature_size]
    if args.benchmark == "svm_wifi":
        package_name = f"{args.benchmark}_{'_'.join(feature_size)}_{pus}PU_{pes}PE"
    else:
        package_name = f"{args.benchmark}_{'_'.join(feature_size)}"
    cfg_path = f"config_{pus}_{pes}.json"
    simulate_benchmark(package_name, cfg_path)