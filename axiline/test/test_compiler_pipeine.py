from axiline.compiler.compiler_flat import VerilogGenerateFlexBitwidth, VerilogCreateGraphBitwidth
from polymath.polymath.srdfg.passes.compiler_passes import NormalizeGraph, Lower, VerilogGenerateFlexBitwidth, TestGraph, VerilogCreateGraphBitwidth
from axiline.compiler.compiler_pipeline import PrintNodes
import polymath.polymath as pm
from axiline.compiler.compiler_pipeline import AddSuccessors, Pipeline,PrintNodes
from axiline.compiler.compiler_pipeline_backpro import PipelineBackpro
import pprint
import numpy as np
from pathlib import Path
from polymath.polymath.tests.util import count_nodes, linear, reco
import warnings

def test_svm():
    cwd = Path(f"{__file__}").parent.parent.parent
    onnx_path=f"{cwd}/benchmarks/onnx_files/svm54.onnx"
    # onnx_path = f"{cwd}/tests/onnx_examples/backprop3_4_2.onnx"
    print(onnx_path)
    graph = pm.from_onnx(onnx_path)
    shapes = {}
    # # # # Initialize the Normalize pass with dictionary
    # shape_pass = pm.NormalizeGraph(shapes)
    # #
    # # # # Now that shapes are known, generate the scalar sub-graphs using the normalize pass
    # transformed_graph = shape_pass(graph)
    # # test=TestGraph()
    # #new_graph=test(graph)
    #
    # lower_pass = Lower({})
    # lowered_graph = lower_pass(graph, {})
    pre=AddSuccessors()
    graph=pre(graph)
    pipeline=Pipeline(debug=True)
    # # pipeline=PipelineBackpro(debug=True)
    # graph=pipeline(graph)
    # print(pipeline.counts)
    # print(pipeline.stage1_sink,pipeline.stage2_sink,pipeline.stage3_sink)
    p=PrintNodes()
    graph=p(graph)

    # print(pipeline.stage1)
    # #Apply transformations and/or generate verilog using 'transformed_graph'
    # for node_name, node in lowered_graph.nodes.items():
    #      print(f"{node_name} - {node.op_name}")

    # rtl_param = {
    #     "algo": "svm",
    #     "input_bitwidth": 8,
    #     "internal_bitwidth": 16,
    #     "param_list": {"mu:0":"16'b1"},
    # }
    # embedded_param = {
    #     "weight": [2, 3, 4],
    #     "bias": [1]
    # }
    #
    # verilog_pass = VerilogGenerateFlexBitwidth(rtl_param)
    # # # file path of
    # new_graph = verilog_pass(lowered_graph)
    # verilog_pass.create_verilog_nonembedded('./test.v')=

if __name__ == '__main__':
    test_svm()