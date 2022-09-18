from examples.genesys import OP_DTYPES, QUANT_SCALE, SIGN_SHIFT, FXP_CONFIGS
from functools import partial
from fxpmath import Fxp
import numpy as np
from . import ReferenceOp, quantize_np, create_operand_data, transform_data, im2col_indices
WEIGHTS_CL_TO_CF = [3, 2, 0, 1] # (KH, KW, IC, OC) -> (OC, IC, KH, KW)
WEIGHTS_CF_TO_CL = [2, 3, 1, 0] # (OC, IC, KH, KW) -> (KH, KW, IC, OC)
ACT_CL_TO_CF = [0, 3, 1, 2] # (N, H, W, C) -> (N, C, H, W)
ACT_CF_TO_CL = [0, 2, 3, 1] # (N, C, H, W) -> (N, H, W, C)


class Pool(ReferenceOp):

    def __init__(self, pool_type, cdlt, program):
        self.pool_type = pool_type
        operands = [cdlt.inputs[0],]
        outputs = [cdlt.outputs[0]]
        self.dtype = "FXP32"
        super().__init__(cdlt, operands, outputs, program, scale=1)

    def fn_impl(self, inouts):
        data = inouts['inputs'][0].data
        k = self.cdlt.required_params['KH'].value
        stride = self.cdlt.required_params['sx'].value

        data = data.transpose(0, 3, 1, 2)
        if self.pool_type == "avg":
            output = self.avg_pool(data, k, stride, 0)
        else:
            assert self.pool_type == "max"
            output = self.max_pool(data, k, stride, 0)
        output = output.transpose(*tuple(ACT_CF_TO_CL))
        inouts['outputs'] = [output]

        return inouts


    def max_pool(self, x, k, stride, pad):
        x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
        N, C, H, W = x_padded.shape

        pool_height, pool_width = k, k

        # assert (H - pool_height) % stride == 0, 'Invalid height'
        # assert (W - pool_width) % stride == 0, 'Invalid width'

        out_height = (H - pool_height) // stride + 1
        out_width = (W - pool_width) // stride + 1

        x_split = x_padded.reshape(N * C, 1, H, W)
        x_cols = im2col_indices(x_split, pool_height, pool_width, padding=0, stride=stride)
        x_cols_argmax = np.argmax(x_cols, axis=0)
        x_cols_max = x_cols[x_cols_argmax, np.arange(x_cols.shape[1])]
        out = x_cols_max.reshape(out_height, out_width, N, C).transpose(2, 3, 0, 1)

        return out


    def avg_pool(self, x, k, stride, pad):
        x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
        N, C, H, W = x_padded.shape

        pool_height, pool_width = k, k

        # assert (H - pool_height) % stride == 0, 'Invalid height'
        # assert (W - pool_width) % stride == 0, 'Invalid width'

        out_height = (H - pool_height) // stride + 1
        out_width = (W - pool_width) // stride + 1

        x_split = x_padded.reshape(N * C, 1, H, W)
        x_cols = im2col_indices(x_split, pool_height, pool_width, padding=0, stride=stride)
        # x_cols_argmax = np.argmax(x_cols, axis=0)
        x_cols_sum = np.sum(x_cols, axis=0)

        denom = Fxp(1.0 / (x_cols.shape[0]), **FXP_CONFIGS[self.dtype]).val.item()

        x_cols_mean = quantize_np(x_cols_sum*denom, self.dtype)
        out = x_cols_mean.reshape(out_height, out_width, N, C).transpose(2, 3, 0, 1)


        return out


class Softmax(ReferenceOp):

    def __init__(self, cdlt, program):
        operands = [cdlt.inputs[0],]
        outputs = [cdlt.outputs[0]]
        self.dtype = "FXP32"
        super().__init__(cdlt, operands, outputs, program, scale=1)

class DWConv(ReferenceOp):

    def __init__(self, cdlt, program, use_bias=True, use_quantization=True):
        self.dtype = "FXP32"
        self.use_bias = use_bias
        self.use_quantization = use_quantization
        if self.use_bias:
            operands = [cdlt.inputs[0], cdlt.inputs[1], cdlt.inputs[2]]
        else:
            operands = [cdlt.inputs[0], cdlt.inputs[1]]
        outputs = [cdlt.outputs[0]]
        self.stride = cdlt.required_params['stride'].value
        super().__init__(cdlt, operands, outputs, program, scale=2)

    @property
    def data(self):
        return self.operands[0]

    @property
    def weight(self):
        return self.operands[1]

    @property
    def bias(self):
        return self.operands[2]

    def fn_impl(self, inouts):
        data = inouts['inputs'][0].data
        wgt = inouts['inputs'][1].data
        if self.use_bias:
            bias = inouts['inputs'][2].data
        else:
            bias = None


        data = data.transpose(0, 3, 1, 2)
        wgt = wgt.transpose(*tuple(WEIGHTS_CL_TO_CF))
        output = self.dw_conv2d(data, wgt, self.stride, 0, bias=bias)
        output = output.transpose(0, 2, 3, 1)
        inouts['outputs'] = [output]
        return inouts

    def dw_conv2d(self, data, w, stride, pad, bias=None):

        padded_input = np.pad(data,
                              pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad)),
                              mode='constant',
                              constant_values=0)
        padded_input = padded_input.astype(np.int64)

        kh, kw = w.shape[2], w.shape[3]
        batch, in_depth, height, width = data.shape
        assert in_depth == w.shape[0]
        oh = int(1 + (height + 2 * pad - kh) / stride)
        ow = int(1 + (width + 2 * pad - kw) / stride)
        output = np.zeros((batch, in_depth, oh, ow)).astype(np.int64)

        for n in range(batch):
            for c in range(in_depth):
                # For each input channel separately, apply its corresponsing filter
                # to the input.
                for i in range(oh):
                    for j in range(ow):
                        for fi in range(kh):
                            for fj in range(kw):
                                w_element = w[c, 0, fi, fj]
                                mul_res = padded_input[n, c, i * stride + fi, j * stride + fj] * w_element
                                mul_res_quant = quantize_np(mul_res, self.dtype)
                                # mul_res_quant = mul_res
                                output[n, c, i, j] += mul_res_quant
                if bias:
                    output[n,c] += bias[c]
        return output


class GlobalAvgPool(ReferenceOp):

    def __init__(self, cdlt, program):
        operands = [cdlt.inputs[0],]
        outputs = [cdlt.outputs[0]]
        self.dtype = "FXP32"
        super().__init__(cdlt, operands, outputs, program, scale=2)


    def fn_impl(self, inouts):
        data = inouts['inputs'][0].data.copy()
        data = data.transpose(0, 3, 1, 2)
        out = np.sum(data, axis=(2, 3), keepdims=True)
        denom = Fxp(1.0 / (data.shape[2] * data.shape[3]), **FXP_CONFIGS[self.dtype]).val.item()
        out = out * denom
        out = quantize_np(out, self.dtype)
        out = out.transpose((0, 2, 3, 1))
        inouts['outputs'] = [out]
        return inouts


class Gelu(ReferenceOp):

    def __init__(self, cdlt, program):
        operands = [cdlt.inputs[0],]
        outputs = [cdlt.outputs[0]]
        self.dtype = "FXP32"
        super().__init__(cdlt, operands, outputs, program, scale=1)

class BiasAdd(ReferenceOp):

    def __init__(self, cdlt, program):
        operands = [cdlt.inputs[0], cdlt.inputs[1]]
        outputs = [cdlt.outputs[0]]
        super().__init__(cdlt, operands, outputs, program)

    def fn_impl(self, inouts):
        data = inouts['inputs'][0].data
        bias = inouts['inputs'][1].data

        output = data + bias

        inouts['outputs'] = [output]
        return inouts

class UnImplementedOp(ReferenceOp):

    def __init__(self, cdlt, program):
        operands = []
        outputs = []
        super().__init__(cdlt, operands, outputs, program)
        raise RuntimeError(f"Op {cdlt.op_name} is not yet implemented")

def load_dnn_impls(cfg):

    DNN_IMPLS = {
        "avg_pool": partial(Pool, "avg"),
        "softmax4d": Softmax,
        "bias_add": BiasAdd,
        "batch_norm": UnImplementedOp,
        "cross_entropy_loss": UnImplementedOp,
        "depthwise_conv": partial(DWConv, use_bias=False),
        "depthwise_conv_bias": partial(DWConv, use_bias=True),
        "global_avg_pool": GlobalAvgPool,
        "max_pool": partial(Pool, "max"),
        "mean_var": UnImplementedOp,
        "gelu": Gelu,
    }
    return DNN_IMPLS