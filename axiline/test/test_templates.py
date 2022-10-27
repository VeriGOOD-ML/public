from axiline.compiler.templates import Templates
from axiline.compiler.impl_node import ImplNode
from axiline.compiler.compiler_template import PrePipeline
from axiline.compiler.compiler_pipeline import PrintNodes, AddSuccessors
from axiline.compiler.optimization import Optimization
from axiline.compiler.generate_verilog import generate_verilog
from axiline.compiler.generate_vh import generate_vh
from axiline.compiler.axiline_compiler import axiline_compiler
from pathlib import Path
import polymath as pm

def test():
    cwd = Path(f"{__file__}").parent.parent.parent
    onnx_path = f"{cwd}/axiline/test/benchmarks/svm54.onnx"
    template_path = f"{cwd}/axiline/test/templates"
    output_path = f"{cwd}/axiline/test/outputs"
    axiline_compiler(onnx_path=onnx_path,
                     bandwidth= 160,
                     function= 3,
                     template_path=template_path,
                     output_path=output_path
                     )

def test_init_json():
    osip={
        'name':'osip',
        'level':2,
        'operation':['slice_mul','sum'],
    }
    # isip = {
    #     'name': 'isip',
    #     'level': 1,
    #     'operation': ['sum','slice_mul'],
    # }
    sigmoid = {
        'name': 'sigmoid',
        'level': 0,
        'operation': ['sigmoid'],
    }
    sub = {
        'name': 'sub',
        'level': 0,
        'operation': ['sub'],
    }
    slice_sub = {
        'name': 'slice_sub',
        'level': 1,
        'operation': ['slice_sub'],
    }
    add = {
        'name': 'add',
        'level': 0,
        'operation': ['add'],
    }
    mul = {
        'name': 'mul',
        'level': 0,
        'operation': ['mul'],
    }
    slice_mul = {
        'name': 'slice_mul',
        'level': 1,
        'operation': ['slice_mul'],
    }
    gt = {
        'name': 'gt',
        'level': 0,
        'operation': ['gt'],
    }
    sgd = {
        'name': 'sgd',
        'level': 2,
        'operation': ['slice_mul','slice_mul','slice_sub'],
    }

    op_sgd = {
        'name': 'op_sgd',
        'level': 2,
        'operation': ['slice_mul','slice_sub'],
    }

    parameter = {
        'sub/x:0':0,
        'Const:0':2
    }
    # Initiate template

    basic_templates=[osip,sigmoid,sub,slice_sub,add,mul,slice_mul,gt,sgd]
    op_templates = [op_sgd]
    init_templates= {
        "templates":basic_templates,
        "op_templates":op_templates
    }
    temp=Templates(init_templates)

    # initiate DFG
    cwd = Path(f"{__file__}").parent.parent.parent
    onnx_path = f"{cwd}/benchmarks/onnx_files/svm54.onnx"
    # onnx_path = f"{cwd}/tests/onnx_examples/backprop3_4_2.onnx"
    graph = pm.from_onnx(onnx_path)
    nodes = PrintNodes()
    nodes(graph)

    addsuc = AddSuccessors()
    graph = addsuc(graph)

    # Transfer into implementation graph
    # nodes(graph)
    # pre=PrePipeline(debug=False,templates=init_templates, parameter=parameter)
    template_path = f"{cwd}/axiline/test/templates"
    output_path = f"{cwd}/axiline/test/outputs"

    pre=PrePipeline(parameter_json_path= template_path,debug=False)


    pre(graph)
    pre.syn_pre_suc()
    # debug
    impl_graph = pre.impl_graph
    opt = Optimization(impl_graph, template_path)
    impl_graph=opt.impl_graph

    for impl in impl_graph:
        if isinstance(impl, ImplNode):
            print(impl.name)
            print(f"level:{impl.level}")
            if (impl.level>2):
                print(f"value:{impl.value}")
            print(f"dim:{impl.dim}")
            print(len(impl.predecessors))
        if impl.predecessors:
            predecessors = impl.predecessors
            for pre in predecessors:
                print(f"    Predecessors:{pre}")
        if  impl.successors:
            successors = impl.successors
            for suc in successors:
                print(f"    Successors:{suc}")
    rtl = generate_verilog(impl_graph,template_path,output_path)

    feature_bandwidth = 160
    generate_vh(impl_graph, feature_bandwidth, output_path)

# def test():
#     cwd = Path(f"{__file__}").parent.parent.parent
#     onnx_path = f"{cwd}/axiline/test/benchmarks/svm54.onnx"
#     template_path = f"{cwd}/axiline/test/templates"
#     output_path = f"{cwd}/axiline/test/outputs"
#     axiline_compiler(onnx_path=onnx_path,
#                      bandwidth= 160,
#                      function= 3,
#                      template_path=template_path,
#                      output_path=output_path
#                      )
#     # Transfer into RTL


