from codelets.examples.genesys import genesys_instructions, define_genesys, GENESYS_CFG
import polymath as pm
from codelets import initialize_program, tile, hoist, pad_operands
from collections import namedtuple
import json
from pathlib import Path

CWD = Path(f"{__file__}").parent
BENCH_DIR = f"{CWD}/../benchmarks"
MODEL_DIR = f"{BENCH_DIR}/models/srdfg"
LAYER_DIR = f"{BENCH_DIR}/layers/srdfg"

TestDfgNode = namedtuple('TestDfgNode', ['input_components', 'input_shapes', 'attrs'])
GENESYS_CFG_PATH = f"{CWD}/scratch/genesys_cfg.json"


def run_benchmark(benchmark_name, output_type="simulation"):
    assert benchmark_name in ['resnet18', 'resnet50', 'maskrcnn']
    graph = pm.pb_load(f"{MODEL_DIR}/{benchmark_name}.srdfg")
    genesys = define_genesys(GENESYS_CFG)
    program = initialize_program(graph, genesys)
    program.add_compilation_step("pad_operands", pad_operands, preproc=True, stage_kwargs={'shaped_nodes': []})
    program.add_compilation_step("tile", tile)
    program.add_compilation_step("hoist", hoist, dependencies=["tile"])
    program.compile()
    if output_type == 'simulation':
        res = program.emit("json")
        with open(f"{BENCH_DIR}/compilation_output/{benchmark_name}.json", 'w') as outfile:
                json.dump(res, outfile)
    else:
        res = program.emit("string_final")
        with open(f"{BENCH_DIR}/compilation_output/{benchmark_name}.txt", 'w') as outfile:
                outfile.write(res)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='GeneSys compiler')
    argparser.add_argument('-b', '--benchmark', required=True,
                           help='Name of the benchmark to create. One of "resnet18", "resnet50", or "maskrcnn".')
    argparser.add_argument('-ot', '--output_type', required=True,
                           help='Output type. Must be one of "simulation" or "instructios".')

    args = argparser.parse_args()
    run_benchmark(args.benchmark, args.output_type)
