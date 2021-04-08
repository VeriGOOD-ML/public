from axiline.axiline.run_passes import VerilogGenerateFlexBitwidth,VerilogGenerateFixedBitwidth
import polymath.polymath as pm
import argparse
from pathlib import Path
import json
import os

def run_benchmark(path_to_benchmark,path_to_config,path_to_output):
    # cwd = Path(f"{__file__}").parent
    # onnx_path = f"{cwd}/benchmarks/{benchmark}"

    graph = pm.from_onnx(path_to_benchmark)
    shapes = {}
    shape_pass = pm.NormalizeGraph(shapes)
    transformed_graph = shape_pass(graph)
    lower_pass = pm.Lower({})
    lowered_graph = lower_pass(transformed_graph, {})
    import json
    with open(path_to_config) as f:
        config = json.load(f)

    if (set(["algo","input_bitwidth","internal_bitwidth","param_list"]).issubset(set(config.keys()))):
        verilog_pass = VerilogGenerateFixedBitwidth(config)
        # # file path of
        new_graph = verilog_pass(lowered_graph)
        verilog_pass.create_verilog_nonembedded(path_to_output)



def main():
    parser = argparse.ArgumentParser(description="Axiline compile Framework")
    parser.add_argument("action",
                        type=Text,
                        help="One of the following: 'pmlang', 'onnx', or 'instructions' which generates a"
                             " serialized CMstack graph_name from either "
                             "a CMLang file or an ONNX protobuf file, or generates instructions "
                             "code from a CMStack file.",
                        choices=["pmlang", "onnx", "instructions", "visualize", "c", "tabla", "translate"])
    parser.add_argument("--benchmark", "-b",
                        type=Text, required=True,
                        help="select a benchmark")
    parser.add_argument("--config","-c",
                        type=Text, required=False,
                        help="choose a json config, or use default")
    parser.add_argument("--output","-o",
                        type=Text, required=False,
                        help="output directory")
    args = parser.parse_args()


if __name__ == '__main__':
    run_benchmark(1,1,1)