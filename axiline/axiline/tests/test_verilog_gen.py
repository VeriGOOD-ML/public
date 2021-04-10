from polymath.polymath.srdfg.passes.compiler_passes import NormalizeGraph,Lower
from axiline.axiline.run_passes import VerilogGenerateFixedBitwidth, VerilogGenerateFlexBitwidth
import polymath.polymath as pm
from pathlib import Path



def test_svm():
    cwd = Path(f"{__file__}").parent
    onnx_path=f"{cwd}/benchmarks/svm54.onnx"
    print(onnx_path)
    graph = pm.from_onnx(onnx_path)
    # #print(graph)
    #
    # Create a dictionary of feature sizes to variable names
    shapes = {'m': 4}
    # # # # Initialize the Normalize pass with dictionary
    shape_pass = pm.NormalizeGraph(shapes)
    # # # Now that shapes are known, generate the scalar sub-graphs using the normalize pass
    transformed_graph = shape_pass(graph)

    lower_pass = Lower({})
    lowered_graph = lower_pass(transformed_graph, {})
    # #Apply transformations and/or generate verilog using 'transformed_graph'
    rtl_param = {
        "algo": "svm",
        "input_bitwidth": 8,
        "internal_bitwidth": 16,
        "param_list": {"mu:0":1},
        "m": 1
    }

    verilog_pass = VerilogGenerateFlexBitwidth(rtl_param)
    # # file path of
    new_graph = verilog_pass(lowered_graph)
    verilog_pass.create_verilog_nonembedded('./test.v')