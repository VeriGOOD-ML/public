import sys, os
import pprint

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from backend import Component

from pathlib import Path
from compiler import compile
import shutil
import argparse
DFG_ROOT = f"{Path(f'{__file__}').parent}/dfgs/polymath_generated"

def run_benchmark(package_name, gen_schedule, gen_mem_instr, is_training_algorithm):
    Component.reset_ids()

    dfg_name = f"{package_name}.json"

    package_path = f"{DFG_ROOT}/{package_name}"
    optimizations = {'reorder_instr': True, 'unused_ni_opt': True, 'apply_reuse': True}
    file_path = f"{dfg_name}"
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


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='TABLA Benchmark Runner')
    argparser.add_argument('-b', '--benchmark', required=True,
                           help='Name of the benchmark to create. One of "logistic", "linear", "reco",'
                                'or "svm".')
    argparser.add_argument('-fs', '--feature_size', nargs='+', required=True,
                           help='Feature size to use for creating the benchmark')
    argparser.add_argument('-gs', '--generate_sched_json', default=False, action='store_true',
                           help='Generate schedule json file')
    argparser.add_argument('-nm', '--no_mem', default=True, action='store_false',
                           help='Generate schedule json file')
    argparser.add_argument('-ia', '--inference_algorithm', default=False, action='store_true',
                           help='Whether or not the benchmark is an inference algorithm')
    args = argparser.parse_args()
    package_name = f"{args.benchmark}_{'_'.join(args.feature_size)}"
    run_benchmark(package_name, args.generate_sched_json, args.no_mem, not args.inference_algorithm)
