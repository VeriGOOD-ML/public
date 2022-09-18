from typing import List
from . import ReferenceOp, quantize_np

class Binary(ReferenceOp):

    def __init__(self, cdlt, hag):
        operands = [cdlt.inputs[0], cdlt.inputs[1]]
        outputs = [cdlt.outputs[0]]
        super().__init__(cdlt, operands, outputs, hag)

    def fn_impl(self, inouts):
        inpt1 = inouts['inputs'][0].data
        inpt2 = inouts['inputs'][1].data
        if len(inpt1.shape) == 4:
            inpt1 = inpt1.transpose((0, 3, 1, 2))
            inpt2 = inpt2.transpose((0, 3, 1, 2))

        quantize = False
        if "add" in self.cdlt.op_name:
            ref_fn = lambda a, b: a + b
        elif "sub" in self.cdlt.op_name:
            ref_fn = lambda a, b: a - b
        elif "div" in self.cdlt.op_name:
            quantize = True
            ref_fn = lambda a, b: a // b
        elif "equal" in self.cdlt.op_name:
            ref_fn = lambda a, b: a == b
        elif "less" in self.cdlt.op_name:
            ref_fn = lambda a, b: a > b
        elif "mul" in self.cdlt.op_name:
            quantize = True
            ref_fn = lambda a, b: a * b
        else:
            raise RuntimeError

        output = ref_fn(inpt1, inpt2)
        if quantize:
            output = quantize_np(output, "FXP32")

        if len(output.shape) == 4:
            output = output.transpose((0, 2, 3, 1))
        inouts['outputs'] = [output]
        return inouts


def load_binary_impls(cfg):

    BINARY_IMPLS = {
        "elem_add": Binary,
        "elem_add1d": Binary,
        "elem_add2d2d": Binary,
        "elem_add3d_const": Binary,
        "elem_add3d3d": Binary,
        "elem_add1d1d": Binary,
        "elem_sub": Binary,
        "elem_div": Binary,
        "elem_div_const": Binary,
        "elem_mul": Binary,
        "elem_mul3d_const": Binary,
        "elem_mul3d3d": Binary,
        "elem_less": Binary,
        "elem_equal": Binary,
    }
    return BINARY_IMPLS