from axiline.compiler.compiler_flat import VerilogGenerateFlexBitwidth, VerilogCreateGraphBitwidth
from polymath.polymath.srdfg.passes import register_pass, Pass
from polymath.polymath.srdfg.passes.compiler_passes import NormalizeGraph, Lower, VerilogGenerateFlexBitwidth, TestGraph, VerilogCreateGraphBitwidth
import polymath as pm
import pprint
import numpy as np
from pathlib import Path
from polymath.polymath.tests.util import count_nodes, linear, reco
import warnings

def test_linear_reg():
    m_ = 3
    graph, input_info, out_info, keys = linear(m=m_, coarse=True)
    coarse_eval = graph(keys, input_info)
    np.testing.assert_allclose(coarse_eval, out_info["w"])

    fgraph, input_info, out_info, keys = linear(m=m_, coarse=False)
    lower_pass = Lower({})
    lowered_graph = lower_pass(fgraph, {})
    all_vals = lowered_graph(keys, input_info)
    out = np.asarray(all_vals).reshape(out_info["w"].shape)

    np.testing.assert_allclose(out, out_info["w"])
    cwd = Path(f"{__file__}").parent
    base_path = f"{cwd}/pmlang_examples"
    full_path = f"{base_path}/outputs"
    pb_path = f"{full_path}/{graph.name}.pb"

    pm.pb_store(lowered_graph, full_path)

    loaded_node = pm.pb_load(pb_path)
    _, input_info, out_info, keys = linear(m=m_, coarse=False)

    loaded_res = loaded_node(keys, input_info)
    out = np.asarray(loaded_res).reshape(out_info["w"].shape)
    np.testing.assert_allclose(out, out_info["w"])


def test_reco():
    m_ = 30
    n_ = 28
    k_ = 3
    shape_dict = {"m": n_, "k": k_, "n": n_}
    graph, input_info, out_info, keys = reco(coarse=True, **shape_dict)
    coarse_eval = graph(keys, input_info)
    np.testing.assert_allclose(coarse_eval[0], out_info["w1"])
    np.testing.assert_allclose(coarse_eval[1], out_info["w2"])


    fgraph, input_info, out_info, keys = reco(coarse=False, **shape_dict)
    lower_pass = Lower({})
    lowered_graph = lower_pass(fgraph, {})
    all_vals = lowered_graph(keys, input_info)
    w1_elems = np.prod(out_info["w1"].shape)
    w2_elems = np.prod(out_info["w2"].shape)
    out1 = np.asarray(list(all_vals[0:w1_elems])).reshape(out_info["w1"].shape)
    out2 = np.asarray(list(all_vals[w1_elems:])).reshape(out_info["w2"].shape)

    np.testing.assert_allclose(out1, out_info["w1"])
    np.testing.assert_allclose(out2, out_info["w2"])
    cwd = Path(f"{__file__}").parent
    base_path = f"{cwd}/pmlang_examples"
    full_path = f"{base_path}/outputs"
    pb_path = f"{full_path}/{graph.name}.pb"

    pm.pb_store(lowered_graph, full_path)

    loaded_node = pm.pb_load(pb_path)
    _, input_info, out_info, keys = reco(coarse=False, **shape_dict)

    loaded_res = loaded_node(keys, input_info)
    lres1 = np.asarray(list(loaded_res[0:w1_elems])).reshape(out_info["w1"].shape)
    lres2 = np.asarray(list(loaded_res[w1_elems:])).reshape(out_info["w2"].shape)
    np.testing.assert_allclose(lres1, out_info["w1"])
    np.testing.assert_allclose(lres2, out_info["w2"])



def test_svm():
    cwd = Path(f"{__file__}").parent.parent.parent
    onnx_path=f"{cwd}/benchmarks/onnx_files/svm54.onnx"
    print(onnx_path)
    graph = pm.from_onnx(onnx_path)
    # #print(graph)
    #
    # Create a dictionary of feature sizes to variable names
    shapes = {}
    # # # # Initialize the Normalize pass with dictionary
    shape_pass = pm.NormalizeGraph(shapes)
    #
    # # # Now that shapes are known, generate the scalar sub-graphs using the normalize pass
    transformed_graph = shape_pass(graph)
    # test=TestGraph()
    #new_graph=test(graph)

    lower_pass = Lower({})
    lowered_graph = lower_pass(transformed_graph, {})
    # #Apply transformations and/or generate verilog using 'transformed_graph'
    # for node_name, node in lowered_graph.nodes.items():
    #      print(f"{node_name} - {node.op_name}")

    rtl_param = {
        "algo": "svm",
        "input_bitwidth": 8,
        "internal_bitwidth": 16,
        "param_list": {"mu:0":"16'b1"},
    }
    embedded_param = {
        "weight": [2, 3, 4],
        "bias": [1]
    }

    verilog_pass = VerilogGenerateFlexBitwidth(rtl_param)
    # # file path of
    new_graph = verilog_pass(lowered_graph)
    verilog_pass.create_verilog_nonembedded('./test.v')


def test_linear():
    cwd = Path(f"{__file__}").parent
    onnx_path=f"{cwd}/onnx_examples/linear_784.onnx"
    print(onnx_path)
    graph = pm.from_onnx(onnx_path)
    print(graph)

    # # # Create a dictionary of feature sizes to variable names
    # shapes = {'m': 784}
    # # # Initialize the Normalize pass with dictionary
    # shape_pass = pm.NormalizeGraph(shapes)
    # #
    # # # Now that shapes are known, generate the scalar sub-graphs using the normalize pass
    # transformed_graph = shape_pass(graph)
    # lower_pass = Lower({})
    # lowered_graph = lower_pass(transformed_graph, {})
    # #Apply transformations and/or generate verilog using 'transformed_graph'
    # # for node_name, node in transformed_graph.nodes.items():
    # #      print(f"{node_name} - {node.op_name}")
    # rtl_param = {
    #     "algo": "linear",
    #     "weight_bitwidth": 8,
    #     "activation_bitwidth": 8,
    #     "bias_bitwidth": 8,
    #     "param_bitwidth": 8,
    #     "non_embedded_template_path": "./non_embedded_temp.sv",
    #     "embedded_template_path": "./embedded_temp.sv",
    #     "param_list": ["mu"],
    #     "variable_list": ["Activation", "Weight", "Bias", "Output"],
    #     "mu": 1,
    #     "m": 1
    # }
    # embedded_param = {
    #     "weight": [2, 3, 4],
    #     "bias": [1]
    # }
    #
    # verilog_pass = VerilogCreateGraph(rtl_param)
    # # file path of
    # new_graph = verilog_pass(lowered_graph)
    # verilog_pass.create_verilog_nonembedded()


def test_logistic():
    cwd = Path(f"{__file__}").parent
    #onnx_path = f"{cwd}/onnx_examples/linear_55.onnx"
    onnx_path = f"{cwd}/onnx_examples/logistic_54.onnx"
    print(onnx_path)
    graph = pm.from_onnx(onnx_path)
    print(graph)


    # Create a dictionary of feature sizes to variable names
    shapes = {'m': 54}
    # # Initialize the Normalize pass with dictionary
    shape_pass = pm.NormalizeGraph(shapes)
    #
    # # Now that shapes are known, generate the scalar sub-graphs using the normalize pass
    transformed_graph = shape_pass(graph)
    lower_pass = Lower({})
    lowered_graph = lower_pass(transformed_graph, {})
    # Apply transformations and/or generate verilog using 'transformed_graph'
    # for node_name, node in transformed_graph.nodes.items():
    #      print(f"{node_name} - {node.op_name}")
    rtl_param = {
        "algo": "linear",
        "weight_bitwidth": 8,
        "activation_bitwidth": 8,
        "bias_bitwidth": 8,
        "param_bitwidth": 8,
        "non_embedded_template_path": "./non_embedded_temp.sv",
        "embedded_template_path": "./embedded_temp.sv",
        "param_list": ["mu"],
        "variable_list": ["Activation", "Weight", "Bias", "Output"],
        "mu": 1,
        "m": 1
    }
    embedded_param = {
        "weight": [2, 3, 4],
        "bias": [1]
    }

    verilog_pass = VerilogCreateGraph(rtl_param)
    # file path of
    new_graph = verilog_pass(lowered_graph)
    verilog_pass.create_verilog_nonembedded()

def test_lenet():
    cwd = Path(f"{__file__}").parent
    onnx_path=f"{cwd}/onnx_examples/lenet.onnx"
    #print(onnx_path)
    graph = pm.from_onnx(onnx_path)
    #print(graph)
    test = TestGraph()
    new_graph = test(graph)
    # # Create a dictionary of feature sizes to variable names
    #shapes = {'m': 4}
    # # Initialize the Normalize pass with dictionary
    #shape_pass = pm.NormalizeGraph(6)
    #
    # # Now that shapes are known, generate the scalar sub-graphs using the normalize pass
    #transformed_graph = shape_pass(graph)


def test_backprop():
    cwd = Path(f"{__file__}").parent
    onnx_path=f"{cwd}/onnx_examples/backprop61_128_4.onnx"
    #print(onnx_path)
    graph = pm.from_onnx(onnx_path)
    #print(graph)
    # test = TestGraph()
    # new_graph = test(graph)
    shapes = {"l1": 61, "l2": 128, "l3": 4}
    shape_pass = pm.NormalizeGraph(shapes)
    transformed_graph = shape_pass(graph)
    lower_pass = Lower({})
    lowered_graph = lower_pass(transformed_graph, {})

    rtl_param = {
        "algo": "linear",
        "weight_bitwidth": 8,
        "activation_bitwidth": 8,
        "bias_bitwidth": 8,
        "param_bitwidth": 8,
        "non_embedded_template_path": "./non_embedded_temp.sv",
        "embedded_template_path": "./embedded_temp.sv",
        "param_list": ["mu"],
        "variable_list": ["Activation", "Weight", "Bias", "Output"],
        "mu": 1,
        "m": 1
    }
    embedded_param = {
        "weight": [2, 3, 4],
        "bias": [1]
    }

    verilog_pass = VerilogCreateGraphBitwidth(rtl_param)
    # file path of
    new_graph = verilog_pass(lowered_graph)
    verilog_pass.create_verilog_nonembedded()

def test_backprop_new():
    cwd = Path(f"{__file__}").parent
    onnx_path = f"{cwd}/onnx_examples/backprop8_16_4.onnx"
    # print(onnx_path)
    graph = pm.from_onnx(onnx_path)
    # print(graph)
    # test = TestGraph()
    # new_graph = test(graph)
    # shapes = {"l1": 8, "l2": 16, "l3": 4}
    # shape_pass = pm.NormalizeGraph(shapes)
    # transformed_graph = shape_pass(graph)
    #lower_pass = Lower({})
    #lowered_graph = lower_pass(transformed_graph, {})
    new_pass=TestGraph()
    new_graph=new_pass(graph)