from typing import List
from functools import partial
import numpy as np
from . import ReferenceOp, quantize_np, create_operand_data, transform_data
WEIGHTS_CL_TO_CF = [3, 2, 0, 1] # (KH, KW, IC, OC) -> (OC, IC, KH, KW)
WEIGHTS_CF_TO_CL = [2, 3, 1, 0] # (OC, IC, KH, KW) -> (KH, KW, IC, OC)
ACT_CL_TO_CF = [0, 3, 1, 2] # (N, H, W, C) -> (N, C, H, W)
ACT_CF_TO_CL = [0, 2, 3, 1] # (N, C, H, W) -> (N, H, W, C)


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    # assert (H + 2 * padding - field_height) % stride == 0
    # assert (W + 2 * padding - field_height) % stride == 0
    out_height = np.int32((H + 2 * padding - field_height) / stride + 1)
    out_width = np.int32((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)

def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                 stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols

class Conv(ReferenceOp):

    def __init__(self, cdlt, program, use_bias=True, use_quantization=True):
        self.use_bias = use_bias
        self.use_quantization = use_quantization
        operands = [cdlt.inputs[0], cdlt.inputs[1], cdlt.inputs[2]]
        outputs = [cdlt.outputs[0]]
        self.stride = cdlt.required_params['stride'].value
        super().__init__(cdlt, operands, outputs, program, scale=1)

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
        bias = inouts['inputs'][2].data

        inouts["inputs"].append(
            create_operand_data(transform_data(data, "input", "shuffled", self.cdlt, self.hag), self.data, fmt='shuffled'))
        inouts["inputs"].append(
            create_operand_data(transform_data(data, "input", "raw", self.cdlt, self.hag), self.data, fmt='raw'))
        inouts["inputs"].append(
            create_operand_data(transform_data(wgt, "weights", "shuffled", self.cdlt, self.hag), self.weight, fmt='shuffled'))
        inouts["inputs"].append(
            create_operand_data(transform_data(wgt, "weights", "shuffled_raw", self.cdlt, self.hag), self.weight,
                                fmt='shuffled_raw'))
        inouts["inputs"].append(
            create_operand_data(transform_data(wgt, "weights", "raw", self.cdlt, self.hag), self.weight, fmt='raw'))

        data = data.transpose(0, 3, 1, 2)
        wgt = wgt.transpose(*tuple(WEIGHTS_CL_TO_CF))
        output = self.conv_forward(data, wgt, bias, self.stride, 0)
        output = output.transpose(0, 2, 3, 1)
        inouts['outputs'] = [output]
        assert output.shape == self.cdlt.outputs[0].shape, "Output shape is incorrect:\n" \
                                                           f"Operand shape: {self.cdlt.outputs[0].shape}\n" \
                                                           f"Data shape: {output.shape}"
        return inouts


    def conv_forward(self, x, w, b, stride, pad):

        N, C, H, W = x.shape
        num_filters, _, filter_height, filter_width = w.shape

        # Check dimensions

        # assert (W + 2 * pad - filter_width) % stride == 0, f'width does not work:\n' \
        #                                                    f'Input width: {W}\n' \
        #                                                    f'Pad: {pad}\n' \
        #                                                    f'KW: {filter_width}\n' \
        #                                                    f'Stride: {stride}'
        # assert (H + 2 * pad - filter_height) % stride == 0, 'height does not work'

        # Create output
        out_height = int(1 + (H + 2 * pad - filter_height) / stride)
        out_width = int(1 + (W + 2 * pad - filter_width) / stride)
        # out_height = (H + 2 * pad - filter_height) // stride + 1
        # out_width = (W + 2 * pad - filter_width) // stride + 1
        out = np.zeros((N, num_filters, out_height, out_width), dtype=x.dtype)

        # x_cols = im2col_indices(x, w.shape[2], w.shape[3], pad, stride)
        x_cols = im2col_indices(x, w.shape[2], w.shape[3], pad, stride)
        res = w.reshape((w.shape[0], -1)).dot(x_cols) + b.reshape(-1, 1)

        out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0])
        out = out.transpose(3, 0, 1, 2)

        return out

    def conv_forward1(self, x, w, b, stride, pad):

        N, C, H, W = x.shape
        F, C_filter, HH, WW = w.shape
        assert C == C_filter, 'Number of channels are not equal between input and filter'
        ###########################################################################
        # TODO: Implement the convolutional forward pass.                         #
        # Hint: you can use the function np.pad for padding.                      #
        ###########################################################################
        x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')

        H_new = int(1 + (H + 2 * pad - HH) / stride)
        W_new = int(1 + (W + 2 * pad - WW) / stride)

        out = np.zeros((N, F, H_new, W_new), dtype=x.dtype)

        last_row = H + 2 * pad - HH + 1
        last_col = W + 2 * pad - WW + 1

        for f in range(F):
            i_out = 0
            for i in range(0, last_row, stride):
                j_out = 0
                for j in range(0, last_col, stride):
                    x_current = x_pad[:, :, i:(i + HH), j:(j + WW)]
                    out[:, f, i_out, j_out] = np.dot(x_current.reshape((N, -1)), w[f].flatten()) + b[f]
                    j_out += 1
                i_out += 1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return out

class Gemm(ReferenceOp):

    def __init__(self, cdlt, program, use_bias=True, use_quantization=True):
        self.use_bias = use_bias
        self.use_quantization = use_quantization
        if self.use_bias:
            operands = [cdlt.inputs[0], cdlt.inputs[1], cdlt.inputs[2]]
        else:
            operands = [cdlt.inputs[0], cdlt.inputs[1]]
        outputs = [cdlt.outputs[0]]
        super().__init__(cdlt, operands, outputs, program, scale=1)

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


        inouts["inputs"].append(
            create_operand_data(transform_data(data, "input", "shuffled", self.cdlt, self.hag), self.data, fmt='shuffled'))
        inouts["inputs"].append(
            create_operand_data(transform_data(data, "input", "raw", self.cdlt, self.hag), self.data, fmt='raw'))
        inouts["inputs"].append(
            create_operand_data(transform_data(wgt, "weights", "shuffled", self.cdlt, self.hag), self.weight, fmt='shuffled'))
        inouts["inputs"].append(
            create_operand_data(transform_data(wgt, "weights", "shuffled_raw", self.cdlt, self.hag), self.weight,
                                fmt='shuffled_raw'))
        inouts["inputs"].append(
            create_operand_data(transform_data(wgt, "weights", "raw", self.cdlt, self.hag), self.weight, fmt='raw'))

        output = np.dot(np.int32(data), np.int32(wgt))
        if self.use_bias:
            output = output + inouts['inputs'][2].data
        inouts['outputs'] = [output]
        return inouts


def load_sa_impls(cfg):

    if cfg['USE_QUANTIZATION']:
        SA_IMPLS = {
            "conv_bias": partial(Conv, use_bias=True, use_quantization=True),
            "conv": partial(Conv, use_bias=False, use_quantization=True),
            "gemm": partial(Gemm, use_bias=True, use_quantization=True),
            "gemm_no_bias": partial(Gemm, use_bias=False, use_quantization=False),
            'matmul': partial(Gemm, use_bias=False, use_quantization=True),
            'matmul2d': partial(Gemm, use_bias=False, use_quantization=True),
            'matmul3d': partial(Gemm, use_bias=False, use_quantization=True),
            'matmul4d': partial(Gemm, use_bias=False, use_quantization=True),
            'matmul4d2d': partial(Gemm, use_bias=False, use_quantization=True)
        }
    else:
        SA_IMPLS = {
            "conv_bias": partial(Conv, use_bias=True, use_quantization=False),
            "conv": partial(Conv, use_bias=False, use_quantization=False),
            "gemm": partial(Gemm, use_bias=True, use_quantization=False),
            "gemm_no_bias": partial(Gemm, use_bias=False, use_quantization=False),
            'matmul2d': partial(Gemm, use_bias=False, use_quantization=True),
            'matmul': partial(Gemm, use_bias=False, use_quantization=False),
            'matmul3d': partial(Gemm, use_bias=False, use_quantization=False),
            'matmul4d': partial(Gemm, use_bias=False, use_quantization=False),
            'matmul4d2d': partial(Gemm, use_bias=False, use_quantization=False)

        }
    return SA_IMPLS