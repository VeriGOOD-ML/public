from typing import List

from collections.abc import Iterable
from examples.genesys import FXP_CONFIGS
from fxpmath import Fxp
import numpy as np
from .ref_op import OperandData
from . import ReferenceOp, quantize_np

class Unary(ReferenceOp):

    def __init__(self, cdlt, program):
        self.dtype = "FXP32"
        operands = [cdlt.inputs[0]]
        outputs = [cdlt.outputs[0]]
        super().__init__(cdlt, operands, outputs, program)

    def fn_impl(self, inouts):
        inpt1 = inouts['inputs'][0].data
        if "clip" in self.op_name:
            minval = self.cdlt.required_params['min'].value
            maxval = self.cdlt.required_params['max'].value

            params = (minval, maxval)
        elif "tensor_transpose2d" in self.op_name:
            axes = (1, 0)
            params = (axes,)
        elif "pow" in self.op_name:
            exp = self.cdlt.required_params['exp'].value
            params = (exp,)
        elif "reduce_mean" in self.op_name or "reduce_min" in self.op_name:
            axis = self.cdlt.required_params['axis'].value
            params = (axis,)
        elif "sqrt" in self.op_name:
            # inpt1 = Fxp(np.abs(inpt1), **FXP_CONFIGS[self.dtype]).val
            inpt1 = np.abs(inpt1)
            inouts['inputs'][0] = inouts['inputs'][0]._replace(data=inpt1)
            params = tuple([])
        else:
            params = tuple([])

        if len(inpt1.shape) == 4:
            inpt1 = inpt1.transpose((0, 3, 1, 2))

        output = self.unary_op(inpt1, *params)
        if len(output.shape) == 4:
            output = output.transpose((0, 2, 3, 1))
        if "sqrt" in self.op_name:
            # inpt1 = Fxp(inouts['inputs'][0].data, **FXP_CONFIGS[self.dtype]).val
            # inouts['inputs'][0] = inouts['inputs'][0]._replace(data=inpt1)
            output = Fxp(output, **FXP_CONFIGS[self.dtype]).val
        inouts['outputs'] = [output]
        return inouts

    def unary_op(self, inpt, *params):
        quantize = False
        if "leaky_relu" in self.op_name:
            quantize = True
            params = params + (0.01,)
            output = self.leaky_relu_pw(inpt, *params)
        elif "flatten" in self.op_name:
            output = np.reshape(inpt, (inpt.shape[0], -1))
        elif "relu" in self.op_name:
            output = np.maximum(inpt, 0, inpt)
        elif "tanh" in self.op_name:
            output = self.tanh_pw(inpt)
        elif "sigmoid" in self.op_name:
            output = self.sigmoid_pw(inpt)
        elif "clip" in self.op_name:
            output = self.clipfn(inpt, *params)
        elif "ceil" in self.op_name:
            output = self.ceilfn(inpt)
        elif "pow" in self.op_name:
            output = self.powfn(inpt, *params)
        elif "mean" in self.op_name:
            output = self.meanfn(inpt, *params)
        elif "exp" in self.op_name:
            output = self.exp_fn(inpt)
        elif "min" in self.op_name:
            output = self.minfn(inpt, *params)
        elif "transpose" in self.op_name:
            output = self.transposefn(inpt, *params)
        elif "flatten" in self.op_name:
            output = inpt.reshape(inpt.shape[0], -1)
        elif "sqrt" in self.op_name:
            inpt_fp = Fxp(None, **FXP_CONFIGS[self.dtype])
            inpt_fp.val = inpt
            output = np.sqrt(inpt_fp.astype(float))
        else:
            raise RuntimeError

        if quantize:
            output = quantize_np(output, self.dtype)
        return output

    def clipfn(self, data, minval, maxval):
        return np.clip(data, maxval, minval)

    def ceilfn(self, data):
        temp = Fxp(data, **FXP_CONFIGS[self.dtype])
        temp.val = data
        res = np.ceil(temp).like(temp)
        return res.val

    def powfn(self, data, exp):
        out = np.copy(data)
        for _ in range(exp - 1):
            temp = out*data
            out = quantize_np(temp, self.dtype)
        return out

    def meanfn(self, data, axis):
        out = np.sum(data, axis=(axis,), keepdims=True)
        denom = Fxp(1.0 / (data.shape[axis]), **FXP_CONFIGS[self.dtype]).val.item()
        out = out * denom
        out = quantize_np(out, self.dtype)
        return out

    def minfn(self, data, axis):
        return np.min(data, axis)

    def transposefn(self, data, axes):
        return np.transpose(data, axes)

    def exp_fn(self, xval):
        if not isinstance(xval, Iterable):
            xval = np.asarray([xval])

        def inner(x, slope, start):
            result = ((x >> slope) + start)
            return result

        pw5 = Fxp(5.0, **FXP_CONFIGS[self.dtype])
        pw2375 = Fxp(2.375, **FXP_CONFIGS[self.dtype])
        pw1 = Fxp(1.0, **FXP_CONFIGS[self.dtype])

        conds = [
            xval < -pw5.val,
            (xval < -pw2375.val) & (xval >= -pw5.val),
            (xval < -pw1.val) & (xval >= -pw2375.val),
            (xval < 0) & (xval >= -pw1.val),
            (xval >= 0) & (xval < (pw1.val)),
            (xval >= pw1.val) & (xval < (pw2375.val)),
            (xval >= pw2375.val) & (xval < (pw5.val)),
            (xval >= pw5.val)]

        p5 = Fxp(0.5, **FXP_CONFIGS[self.dtype]).val
        p625 = Fxp(0.625, **FXP_CONFIGS[self.dtype]).val
        p84375 = Fxp(0.84375, **FXP_CONFIGS[self.dtype]).val
        p375 = Fxp(0.375, **FXP_CONFIGS[self.dtype]).val
        p15625 = Fxp(0.15625, **FXP_CONFIGS[self.dtype]).val
        # one = Fxp(1.0, **FXP_CONFIGS[self.dtype])
        fns = [lambda x: 0,
               lambda x: inner(x, 5, p15625),
               lambda x: inner(x, 3, p375),
               lambda x: inner(x, 2, p5),
               lambda x: inner(x, 2, p5),
               lambda x: inner(x, 3, p625),
               lambda x: inner(x, 5, p84375),
               lambda x: pw1.val]

        res = np.piecewise(xval, conds, fns)
        # res = np.piecewise(Fxp(xval, **FXP_CONFIGS[dtype]).val, LOOP_CONDS, fns)
        return res

    def leaky_relu_pw(self, xval, alpha):
        if not isinstance(xval, Iterable):
            xval = np.asarray([xval])
        pw1 = Fxp(1.0, **FXP_CONFIGS[self.dtype])

        alpha_val = Fxp(alpha, **FXP_CONFIGS[self.dtype]).val
        one_val = Fxp(1.0, **FXP_CONFIGS[self.dtype]).val
        conds = [
            (xval <= 0),
            (xval > 0)
        ]

        fns = [
            lambda x: x * alpha_val,
            lambda x: x * one_val
        ]

        res = np.piecewise(xval, conds, fns)
        return res

    def tanh_pw(self, xval):
        if not isinstance(xval, Iterable):
            xval = np.asarray([xval])

        pw1 = Fxp(1.0, **FXP_CONFIGS[self.dtype])

        conds = [
            (xval <= (pw1.val)),
            (xval < (pw1.val)) & (xval > -pw1.val),
            (xval >= (pw1.val))
        ]

        fns = [
            lambda x: -pw1.val,
            lambda x: x,
            lambda x: pw1.val
        ]

        res = np.piecewise(xval, conds, fns)
        return res

    def sigmoid_pw(self, xval):

        if not isinstance(xval, Iterable):
            xval = np.asarray([xval])

        def inner(x, slope, start):
            result = ((x >> slope) + start)
            return result

        pw5 = Fxp(5.0, **FXP_CONFIGS[self.dtype])
        pw2375 = Fxp(2.375, **FXP_CONFIGS[self.dtype])
        pw1 = Fxp(1.0, **FXP_CONFIGS[self.dtype])

        conds = [
            xval < -pw5.val,
            (xval < -pw2375.val) & (xval >= -pw5.val),
            (xval < -pw1.val) & (xval >= -pw2375.val),
            (xval < 0) & (xval >= -pw1.val),
            (xval >= 0) & (xval < (pw1.val)),
            (xval >= pw1.val) & (xval < (pw2375.val)),
            (xval >= pw2375.val) & (xval < (pw5.val)),
            (xval >= pw5.val)]

        p5 = Fxp(0.5, **FXP_CONFIGS[self.dtype]).val
        p625 = Fxp(0.625, **FXP_CONFIGS[self.dtype]).val
        p84375 = Fxp(0.84375, **FXP_CONFIGS[self.dtype]).val
        p375 = Fxp(0.375, **FXP_CONFIGS[self.dtype]).val
        p15625 = Fxp(0.15625, **FXP_CONFIGS[self.dtype]).val
        one = Fxp(1.0, **FXP_CONFIGS[self.dtype])
        fns = [lambda x: 0,
               lambda x: inner(x, 5, p15625),
               lambda x: inner(x, 3, p375),
               lambda x: inner(x, 2, p5),
               lambda x: inner(x, 2, p5),
               lambda x: inner(x, 3, p625),
               lambda x: inner(x, 5, p84375),
               lambda x: pw1.val]

        res = np.piecewise(xval, conds, fns)
        # res = np.piecewise(Fxp(xval, **FXP_CONFIGS[dtype]).val, LOOP_CONDS, fns)

        return res

def load_unary_impls(cfg):

    UNARY_IMPLS = {
        "coarse_flatten": Unary,
        "coarse_flatten2d": Unary,
        "coarse_flatten3d": Unary,
        "elem_tanh": Unary,
        "elem_tanh2d": Unary,
        "elem_tanh3d": Unary,
        # TODO: Check if this needs to be 'sigmoid'
        "elem_sigmoid": Unary,
        "leaky_relu": Unary,
        "elem_clip": Unary,
        "elem_ceil2d": Unary,
        "elem_pow1d": Unary,
        "elem_pow2d": Unary,
        "elem_pow3d": Unary,
        "elem_exp": Unary,
        "relu": Unary,
        "relu2d": Unary,
        'tensor_transpose2d': Unary,
        'tensor_transpose3d': Unary,
        'tensor_transpose4d': Unary,
        'elem_cast': Unary,
        'elem_cast2d': Unary,
        'elem_sqrt': Unary,
        'elem_sqrt1d': Unary,
        'elem_sqrt2d': Unary,
        "inv_sqrt": Unary,
    }
    return UNARY_IMPLS