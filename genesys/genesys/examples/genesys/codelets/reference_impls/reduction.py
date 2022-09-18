
from examples.genesys import FXP_CONFIGS
from fxpmath import Fxp
import numpy as np
from functools import partial
from . import ReferenceOp, quantize_np


class Reduction(ReferenceOp):

    def __init__(self, reduction_type, cdlt, program):
        self.reduction_type = reduction_type
        self.dtype = "FXP32"
        self.axis = self.cdlt.required_params['axis'].value
        operands = [cdlt.inputs[0]]
        outputs = [cdlt.outputs[0]]
        super().__init__(cdlt, operands, outputs, program)


    def fn_impl(self, inouts):
        data = inouts['inputs'][0].data
        if len(data.shape) == 4:
            data = data.transpose((0, 3, 1, 2))

        if self.reduction_type == "mean":
            out = np.sum(data, axis=(self.axis,), keepdims=True)
            denom = Fxp(1.0 / (data.shape[self.axis]), **FXP_CONFIGS[self.dtype]).val.item()
            out = out * denom
            out = quantize_np(out, self.dtype)
        elif self.reduction_type == "sum":
            out = np.sum(data, axis=(self.axis,), keepdims=True)
            out = quantize_np(out, self.dtype)
        elif self.reduction_type == "min":
            out = np.min(data, axis=(self.axis,), keepdims=True)
            out = quantize_np(out, self.dtype)
        else:
            raise RuntimeError("unknown reduction type")

        if len(out.shape) == 4:
            out = out.transpose((0, 2, 3, 1))
        inouts['outputs'] = [out]
        return inouts

def load_reduce_impls(cfg):

    REDUCTION_IMPLS = {
        "reduce_sum": partial(Reduction, "sum"),
        "reduce_mean2d": partial(Reduction, "mean"),
        # "reduce_mean2d": partial(reduce_mean, 'reduce_mean2d', 2, 0),
        "reduce_mean3d": partial(Reduction, "mean"),
        # "reduce_mean3d": partial(reduce_mean, 'reduce_mean3d', 3, 2),
        "reduce_min2d": partial(Reduction, "min"),
    }
    return REDUCTION_IMPLS