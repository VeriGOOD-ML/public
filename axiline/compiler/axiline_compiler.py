import json
from axiline.compiler.compiler_pipeline_backpro import PipelineBackpro
from axiline.compiler.compiler_pipeline import Pipeline, AddSuccessors, PrintNodes
from axiline.compiler.compiler_flat import VerilogCreateGraphBitwidth, VerilogGenerateFlexBitwidth
from axiline.compiler.compiler_template import PrePipeline
from axiline.compiler.optimization import Optimization
from axiline.compiler.generate_verilog import generate_verilog
from axiline.compiler.axiline_dt_compiler import axiline_dt_compiler
from axiline.compiler.impl_node import ImplNode
from axiline.compiler.templates import Templates
from polymath.mgdfg.passes.compiler_passes import NormalizeGraph, Lower
from axiline.compiler.generate_vh import generate_vh
import polymath as pm
from pathlib import Path
import argparse


def axiline_compiler(mode, onnx_path, bandwidth, template_path, output_path):
    # flatten rtl
    if mode == 1:
        graph = pm.from_onnx(onnx_path)
        shape_pass = pm.NormalizeGraph({})
        transformed_graph = shape_pass(graph)
        lower_pass = Lower({})
        lowered_graph = lower_pass(transformed_graph, {})
        parameter = json.load(f"{template_path}/flatten.json")
        output_file_path = f"{output_path}/accelerator.v"
        rtl_parameter = parameter['rtl_param']
        verilog_pass = VerilogCreateGraphBitwidth(lowered_graph)
        verilog_pass.create_verilog_embedded(rtl_parameter,output_file_path)

    # 3-stage pipeline design
    elif mode == 2:
        graph = pm.from_onnx(onnx_path)
        pre = AddSuccessors()
        graph = pre(graph)
        pipeline = Pipeline()
        graph = pipeline(graph)
        # 3-stage pipeline
        if pipeline.stage1 and pipeline.stage2 and pipeline.stage3:
            vh_dir = f"{cwd.parent}/output/accelerator.vh"
            generate_vh(pipeline.counts['dim'], bandwidth, vh_dir)
        #  pipeline for backpropgations with 2 FC layer
        else:
            pipelineBackpro = PipelineBackpro()
            vh_dir = f"{cwd.parent}/output/accelerator.vh"
            generate_vh(pipelineBackpro.counts['dim'], bandwidth, vh_dir)

    # template-based design
    elif mode == 3:
        graph = pm.from_onnx(onnx_path)
        addsuc = AddSuccessors()
        graph = addsuc(graph)
        pre = PrePipeline(debug=False, template_path=template_path)
        pre(graph)
        pre.syn_pre_suc()
        impl_graph = pre.impl_graph
        opt = Optimization(impl_graph, template_path)
        impl_graph = opt.impl_graph
        generate_verilog(impl_graph=impl_graph,template_path=template_path,output_path=output_path)
        generate_vh(impl_graph=impl_graph, bandwidth=bandwidth, output_path=output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Axiline Compiler for some small ML algorithms')
    parser.add_argument('-op', '--onnx_path', required=False, type=str, help='Path of a onnx file for small algorithm')
    parser.add_argument('-tp', '--template_path', required=False, type=str, help='Path of template files directory for small algorithm')
    parser.add_argument('-o', '--output_path', required= False, type=str, help='Path of template files directory for small algorithm')
    parser.add_argument('-m', '--mode', required=True, type=int,
                        help="Mode selection. '1' for flatten RTL; '2' for 3-stage pipeline RTL; '3' for "
                             "template-based design, 4 for decision tree")
    parser.add_argument('-b', '--bandwidth', type=int, default=160, help="Bandwidth constrains (in number of "
                                                                     "bits per cycle)")
    parser.add_argument('-gi', '--generate_impl_json', default=False, action='store_true',
                        help='Generate json file for template-based design')
    cwd = Path(f"{__file__}").parent
    args = parser.parse_args()
    onnx_path = args.onnx_path
    mode = args.mode
    bandwidth = args.bandwidth
    generate_impl_json = args.generate_impl_json
    if hasattr(args, 'template_path'):
        template_path = args.template_path
    else:
        template_path = f"{cwd}/axiline/test/templates"
    if hasattr(args, 'output_path'):
        output_path = args.output_path
    else:
        output_path = f"{cwd}/axiline/test/outputs"
    if mode not in [1,2,3]:
        exit("Error, mode should be one of 1,2,3 ")
    axiline_compiler(onnx_path=onnx_path,
                     bandwidth= bandwidth,
                     mode= mode,
                     template_path=template_path,
                     output_path=output_path
                     )
