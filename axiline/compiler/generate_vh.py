from axiline.compiler.impl_node import ImplNode
from axiline.compiler.compiler_template import PrePipeline
from math import ceil


def generate_vh(impl_graph, bandwidth,  output_path, bitw=32, in_bitw=16):
    # to be update
    for node in impl_graph:
        if not isinstance(node, ImplNode):
            exit(f"Error, need a object after PIPELINE preprocessing")
        if 'osip' in node.name:
            dim = node.dim
            size = ceil(bandwidth / in_bitw)
            pipe = ceil(dim / size)
            break
    # need to fix the error that dim of OSIP is 1 but SGD is 54 which are not matched

    with open(f"{output_path}/config.vh", 'w') as fp:
        fp.write(f"`define  `PIPE {pipe}\n"
                 f"`define  `SIZE {size}\n"
                 f"`define  `BITWIDTH {bitw}\n"
                 f"`define  `INPUT_BITWIDTH {in_bitw}\n"
                 )
