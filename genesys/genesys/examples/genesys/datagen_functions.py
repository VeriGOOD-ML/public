import numpy as np
from fxpmath import Fxp
from collections import namedtuple
from collections.abc import Iterable
from . import FXP_CONFIGS
# from . import GENESYS_CFG
import torch.nn.functional as F
import torch
from torch import nn

OperandData = namedtuple('OperandData', ['data', 'node_name', 'opname', 'idx', 'fmt'], defaults=[None])



def save_array(path, data):
    with open(path, 'w') as f:
        f.write('\n'.join([str(i) for i in data.flatten().tolist()]))

def compute_range(fxp_dtype, scale=1):
    cfg = FXP_CONFIGS[fxp_dtype]
    upper_val = (1 << (np.int32(cfg['n_word']//scale) - 1)) - 1
    lower_val = -upper_val - 1

    return lower_val, upper_val

def from_fxp(v, dtype):
    fp = Fxp(v, **FXP_CONFIGS[dtype])
    fp.val = v
    return fp


def numpy_datagen(shape, bitwidth, scale=2, cast_to=None, fxp_dtype='FXP32', constant_val=None, print_range=False):
    if constant_val is None:
        low, high = compute_range(fxp_dtype, scale)
        if print_range:
            print(f"High: {high}, Low: {low}")
        v = np.random.randint(low=low, high=high,
                              size=shape, dtype=np.int64)
    else:
        v = np.full(shape, constant_val, dtype=np.int64)

    return v

def sigmoid_pw(xval, dtype):

    if not isinstance(xval, Iterable):
        xval = np.asarray([xval])

    def inner(x, slope, start):

        result = ((x >> slope) + start)
        return result
    pw5 = Fxp(5.0, **FXP_CONFIGS[dtype])
    pw2375 = Fxp(2.375, **FXP_CONFIGS[dtype])
    pw1 = Fxp(1.0, **FXP_CONFIGS[dtype])

    conds = [
        xval < -pw5.val,
        (xval < -pw2375.val) & (xval >= -pw5.val),
        (xval < -pw1.val) & (xval >= -pw2375.val),
        (xval < 0) & (xval >= -pw1.val),
        (xval >= 0) & (xval < (pw1.val)),
        (xval >= pw1.val) & (xval < (pw2375.val)),
        (xval >= pw2375.val) & (xval < (pw5.val)),
        (xval >= pw5.val)]

    p5     = Fxp(0.5, **FXP_CONFIGS[dtype]).val
    p625   = Fxp(0.625, **FXP_CONFIGS[dtype]).val
    p84375 = Fxp(0.84375, **FXP_CONFIGS[dtype]).val
    p375   = Fxp(0.375, **FXP_CONFIGS[dtype]).val
    p15625 = Fxp(0.15625, **FXP_CONFIGS[dtype]).val
    one = Fxp(1.0, **FXP_CONFIGS[dtype])
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

# Accepts 64 bit FXP input, and quantizes to 32 bit FXP output
def quantize_values(data, high_bits, low_bits, decimal_place):
    # Remove upper `high_bits`
    out = (data << (decimal_place - high_bits))
    out = out >> (decimal_place - high_bits)
    # Remove lower `low_bits`
    out = (out >> (decimal_place - low_bits))
    return out

def quantize_np(d, dtype, inpt=None):
    if dtype == "FXP32":
        high_bits = 16
        low_bits = 16
        dec_place = 32
    else:
        raise RuntimeError

    out = (d << (dec_place - high_bits)) >> (dec_place - high_bits)
    out = (out >> (dec_place - low_bits))

    return out

def clipfn(data, minval, maxval, dtype):
    return np.clip(data, maxval, minval)

def ceilfn(data, dtype):
    temp = Fxp(data, **FXP_CONFIGS[dtype])
    temp.val = data
    res = np.ceil(temp).like(temp)
    return res.val

def powfn(data, exp, dtype):
    out = np.copy(data)
    for _ in range(exp-1):
        out = quantize_np(out*data, dtype)
    return out

def meanfn(data, axis, dtype):
    out = np.sum(data, axis=(axis,), keepdims=True)
    denom = Fxp(1.0/(data.shape[axis]), **FXP_CONFIGS[dtype]).val.item()
    out = out*denom
    out = quantize_np(out, dtype)
    return out

def minfn(data, axis, dtype):
    return np.min(data, axis)

def transposefn(data, axes, dtype):
    return np.transpose(data, axes)

def exp_fn(xval, dtype):
    if not isinstance(xval, Iterable):
        xval = np.asarray([xval])

    def inner(x, slope, start):

        result = ((x >> slope) + start)
        return result
    pw5 = Fxp(5.0, **FXP_CONFIGS[dtype])
    pw2375 = Fxp(2.375, **FXP_CONFIGS[dtype])
    pw1 = Fxp(1.0, **FXP_CONFIGS[dtype])

    conds = [
        xval < -pw5.val,
        (xval < -pw2375.val) & (xval >= -pw5.val),
        (xval < -pw1.val) & (xval >= -pw2375.val),
        (xval < 0) & (xval >= -pw1.val),
        (xval >= 0) & (xval < (pw1.val)),
        (xval >= pw1.val) & (xval < (pw2375.val)),
        (xval >= pw2375.val) & (xval < (pw5.val)),
        (xval >= pw5.val)]

    p5     = Fxp(0.5, **FXP_CONFIGS[dtype]).val
    p625   = Fxp(0.625, **FXP_CONFIGS[dtype]).val
    p84375 = Fxp(0.84375, **FXP_CONFIGS[dtype]).val
    p375   = Fxp(0.375, **FXP_CONFIGS[dtype]).val
    p15625 = Fxp(0.15625, **FXP_CONFIGS[dtype]).val
    one = Fxp(1.0, **FXP_CONFIGS[dtype])
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

def unary(op1, layer_name, dtype, *params):
    quantize = False
    if "leaky_relu" in layer_name:
        quantize = True
        ref_fn = leaky_relu_pw
        params = params + (0.01, dtype,)
    elif "flatten" in layer_name:
        ref_fn = lambda a: np.reshape(a, (a.shape[0], -1))
    elif "relu" in layer_name:
        ref_fn = lambda a: np.maximum(a, 0, a)
    elif "tanh" in layer_name:
        ref_fn = tanh_pw
        params = params + (dtype,)
    elif "sigmoid" in layer_name:
        ref_fn = sigmoid_pw
        params = params + (dtype,)
    elif "clip" in layer_name:
        ref_fn = clipfn
        params = params + (dtype,)
    elif "ceil" in layer_name:
        ref_fn = ceilfn
        params = params + (dtype,)
    elif "pow" in layer_name:
        ref_fn = powfn
        params = params + (dtype,)
    elif "mean" in layer_name:
        ref_fn = meanfn
        params = params + (dtype,)
    elif "exp" in layer_name:
        ref_fn = exp_fn
        params = params + (dtype,)
    elif "min" in layer_name:
        ref_fn = minfn
        params = params + (dtype,)
    elif "transpose" in layer_name:
        ref_fn = transposefn
        params = params + (dtype,)
    else:
        raise RuntimeError

    output = ref_fn(op1, *params)

    if quantize:
        output = quantize_np(output, dtype, op1)
    return output


def binary(op1, op2, layer_name, dtype):
    quantize = False
    if "add" in layer_name:
        ref_fn = lambda a, b: a + b
    elif "sub" in layer_name:
        ref_fn = lambda a, b: a - b
    elif "div" in layer_name:
        quantize = True
        ref_fn = lambda a, b: a // b
    elif "equal" in layer_name:
        ref_fn = lambda a, b: a == b
    elif "less" in layer_name:
        ref_fn = lambda a, b: a > b
    elif "mul" in layer_name:
        quantize = True
        ref_fn = lambda a, b: a * b
    else:
        raise RuntimeError

    output = ref_fn(op1, op2)
    if quantize:
        output = quantize_np(output, dtype, op1)
    return output

def partial_values_dw_conv(cdlt, base_path, x, w, ref_out, o_coord):

    other_test, ic_vals = manual_conv(x, w, cdlt, o_coord, layout="nhwc")
    with open(f'{base_path}/out_coords.csv', 'w') as f:
        f.write(f'IC, (oc/n/ic/kh/kw/y/x), O_idx, I_idx, W_idx, I_val, W_val, partial\n')

        for k, v in ic_vals.items():
            for l in v:
                f.write(f"IC={k}, " + "," + l + "\n")
    np.testing.assert_allclose(other_test, ref_out)

def maxpool2d(x, k, stride, padding=0):
    x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
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
    # cache = (x, x_cols, x_cols_argmax, pool_param)
    return out


def leaky_relu_pw(xval, alpha, dtype):
    if not isinstance(xval, Iterable):
        xval = np.asarray([xval])
    pw1 = Fxp(1.0, **FXP_CONFIGS[dtype])

    alpha_val = Fxp(alpha, **FXP_CONFIGS[dtype]).val
    one_val = Fxp(1.0, **FXP_CONFIGS[dtype]).val
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

def avgpool2d(x, k, stride, padding=0):
    x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    N, C, H, W = x_padded.shape

    pool_height, pool_width = k, k

    # assert (H - pool_height) % stride == 0, 'Invalid height'
    # assert (W - pool_width) % stride == 0, 'Invalid width'

    out_height = (H - pool_height) // stride + 1
    out_width = (W - pool_width) // stride + 1

    x_split = x_padded.reshape(N * C, 1, H, W)
    x_cols = im2col_indices(x_split, pool_height, pool_width, padding=0, stride=stride)
    x_cols_mean = np.mean(x_cols, axis=0)
    out = x_cols_mean.reshape(out_height, out_width, N, C).transpose(2, 3, 0, 1)
    # cache = (x, x_cols, x_cols_argmax, pool_param)
    return out


def tanh_pw(xval, dtype):
    if not isinstance(xval, Iterable):
        xval = np.asarray([xval])

    pw1 = Fxp(1.0, **FXP_CONFIGS[dtype])

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

def global_avg_pool(input_val, dtype):
    out = np.sum(input_val, axis=(2, 3), keepdims=True)
    denom = Fxp(1.0/(input_val.shape[2]*input_val.shape[3]), **FXP_CONFIGS[dtype]).val.item()
    out = out*denom
    out = quantize_np(out, dtype)

    return out

def maxpool(image, f=2, s=2):
    '''
    Downsample `image` using kernel size `f` and stride `s`
    '''
    n_c, c, h_prev, w_prev = image.shape
    h = int((h_prev - f) / s) + 1
    w = int((w_prev - f) / s) + 1

    downsampled = np.zeros((n_c,c, h, w))
    for i in range(n_c):
        for j in range(c):
            # slide maxpool window over each part of the image and assign the max value at each step to the output
            curr_y = out_y = 0
            while curr_y + f <= h_prev:
                curr_x = out_x = 0
                while curr_x + f <= w_prev:
                    downsampled[i, j, out_y, out_x] = np.max(image[i,j, curr_y:curr_y + f, curr_x:curr_x + f])
                    curr_x += s
                    out_x += 1
                curr_y += s
                out_y += 1
    return downsampled

def pad_conv(layer_data, arch_config):
    x = layer_data['input']
    out = layer_data['output']
    wgt = layer_data['params']['weight']
    b = layer_data['params']['bias']
    if x.shape[-1] % arch_config['ARRAY_M'] != 0:
        ic_init = x.shape[-1]
        ic_pad = (arch_config['ARRAY_M'] - ic_init) % arch_config['ARRAY_M']
        assert (ic_pad + ic_init) % arch_config['ARRAY_M'] == 0
        padding = (0, ic_pad)
        x_pad = ((0, 0), (0, 0), (0, 0), padding)

        x = np.pad(x, x_pad, "constant")
        assert wgt.shape[-1] == ic_init
        wgt = np.pad(wgt, x_pad, "constant")

    if out.shape[-1] % arch_config['ARRAY_N'] != 0:
        oc_init = out.shape[-1]
        oc_pad = (arch_config['ARRAY_N'] - oc_init) % arch_config['ARRAY_N']
        assert (oc_pad + oc_init) % arch_config['ARRAY_N'] == 0
        padding = (0, oc_pad)
        out_pad = ((0, 0), (0, 0), (0, 0), padding)
        out = np.pad(out, out_pad, "constant")
        assert wgt.shape[-2] == oc_init
        wgt_pad = ((0, 0), (0, 0), padding, (0, 0))
        wgt = np.pad(wgt, wgt_pad, "constant")
        assert b.shape[0] == oc_init
        b = np.pad(b, padding, "constant")
    return x, wgt, b, out


def pad_gemm(layer_data, arch_config):
    x = layer_data['input']
    out = layer_data['output']
    wgt = layer_data['params']['weight']
    b = layer_data['params']['bias']
    if x.shape[-1] % arch_config['ARRAY_M'] != 0:
        ic_init = x.shape[-1]
        ic_pad = (arch_config['ARRAY_M'] - ic_init) % arch_config['ARRAY_M']
        assert (ic_pad + ic_init) % arch_config['ARRAY_M'] == 0
        padding = (0, ic_pad)
        x_pad = ((0, 0), padding)

        x = np.pad(x, x_pad, "constant")
        assert wgt.shape[0] == ic_init
        wgt_pad = (padding, (0, 0))
        wgt = np.pad(wgt, wgt_pad, "constant")

    if out.shape[-1] % arch_config['ARRAY_N'] != 0:
        oc_init = out.shape[-1]
        oc_pad = (arch_config['ARRAY_N'] - oc_init) % arch_config['ARRAY_N']
        assert (oc_pad + oc_init) % arch_config['ARRAY_N'] == 0
        padding = (0, oc_pad)
        out_pad = ((0, 0), padding)
        out = np.pad(out, out_pad, "constant")
        assert wgt.shape[-1] == oc_init
        wgt_pad = ((0, 0), padding)
        wgt = np.pad(wgt, wgt_pad, "constant")
        assert b.shape[0] == oc_init
        b = np.pad(b, padding, "constant")
    return x, wgt, b, out

def depthwise_conv2d(input, w, stride, pad, dtype):
    import numpy as np
    """Two-dimensional depthwise convolution.

    Uses SAME padding with 0s, a stride of 1 and no dilation. A single output
    channel is used per input channel (channel_multiplier=1).

    input: input array with shape (height, width, in_depth)
    w: filter array with shape (fd, fd, in_depth)

    Returns a result with shape (height, width, in_depth).
    """
    padded_input = np.pad(input,
                          pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad)),
                          mode='constant',
                          constant_values=0)
    padded_input = padded_input.astype(np.int64)

    kh, kw = w.shape[2], w.shape[3]
    batch, in_depth, height, width = input.shape
    assert in_depth == w.shape[0]
    oh = int(1 + (height + 2*pad - kh) / stride)
    ow = int(1 + (width + 2*pad - kw) / stride)
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
                            mul_res = padded_input[n, c, i*stride + fi, j*stride + fj] * w_element
                            mul_res_quant = quantize_np(mul_res, dtype)
                            # mul_res_quant = mul_res
                            output[n, c, i, j] += mul_res_quant
    return output

def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.
    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.
    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.
    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None

    stride = conv_param['stride']
    pad = conv_param['pad']

    N, C, H, W = x.shape
    F, C_filter, HH, WW = w.shape
    assert C == C_filter, 'Number of channels are not equal between input and filter'
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')

    H_new = int(1 + (H + 2*pad - HH) / stride)
    W_new = int(1 + (W + 2*pad - WW) / stride)

    out = np.zeros((N, F, H_new, W_new), dtype=x.dtype)

    last_row = H + 2*pad - HH + 1
    last_col = W + 2*pad - WW + 1

    for f in range(F):
        i_out = 0
        for i in range(0, last_row, stride):
            j_out = 0
            for j in range(0, last_col, stride):
                x_current = x_pad[:, :, i:(i+HH), j:(j+WW)]
                out[:, f, i_out, j_out] = np.dot(x_current.reshape((N, -1)), w[f].flatten()) + b[f]
                j_out += 1
            i_out += 1
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache

def manual_gemm(inputs, weights, o_coord):
    M = inputs.shape[0]
    N = inputs.shape[1]
    P = weights.shape[1]
    outputs = np.zeros((M, P), dtype=np.int32)
    inputs = inputs.astype(np.int32)
    weights = weights.astype(np.int32)
    compilation_info = {i: [] for i in range(N)}
    for p in range(P):
        for n in range(N):
            for m in range(M):
                partial_sum = inputs[m, n] * weights[n, p]
                outputs[m, p] += partial_sum

                if (m, p) == o_coord:
                    all_coords = (m, n, p)
                    icoord = (m, n)
                    icoord_idx = np.ravel_multi_index([m, n], inputs.shape)
                    wcoord = (n, p)
                    wcoord_idx = np.ravel_multi_index([n, p], weights.shape)
                    ocoord = (m, p)
                    ocoord_idx = np.ravel_multi_index([m, p], outputs.shape)
                    compilation_info[n].append(
                        f'"{all_coords}", {ocoord_idx}, {icoord_idx}, {wcoord_idx}, {inputs[icoord]}, {weights[wcoord]}, {partial_sum}')
    return outputs, compilation_info

def manual_conv_from_existing(inputs, weights, out, stride):
    N, IH, IW, IC = inputs.shape
    KH, KW, IC_, OC = weights.shape
    N_, OH, OW, OC_ = out.shape
    assert N_ == N
    assert IC == IC_
    assert OC == OC_

    outputs = np.zeros(out.shape, dtype=np.int64)
    for oc in range(OC):
        for n in range(N):
            for ic in range(IC):
                for kh in range(KH):
                    for kw in range(KW):
                        for y in range(OH):
                            for x in range(OW):
                                partial_sum = inputs[n, kh + y * stride, kw + x * stride, ic] * weights[kh, kw, ic, oc]
                                outputs[n, y, x, oc] += partial_sum
    return outputs

def manual_dw_conv(inputs, weights, cdlt, o_coord, dtype):

    N, IC, IH, IW = inputs.shape
    OC, IC_, KH, KW,  = weights.shape
    N_, OH, OW, OC_ = cdlt.outputs[0].shape
    out_shape = (N_, OC_, OH, OW)
    assert isinstance(o_coord, tuple) and len(o_coord) == 4
    assert N_ == N
    assert IC == IC_
    assert OC == OC_
    outputs = np.zeros(out_shape, dtype=np.int64)
    inputs = inputs.astype(np.int64)
    weights = weights.astype(np.int64)
    stride = cdlt.required_params['stride'].value

    compilation_info = {i: [] for i in range(IC)}
    for oc in range(OC):
        for n in range(N):
            for ic in range(IC):
                for kh in range(KH):
                    for kw in range(KW):
                        for y in range(OH):
                            for x in range(OW):

                                partial_sum = inputs[n, kh + y*stride, kw + x*stride, ic] * weights[kh, kw, ic, oc]
                                partial_sum = quantize_np(partial_sum, dtype)
                                outputs[n, y, x, oc] += partial_sum
                                if (n, y, x, oc) == o_coord:
                                    all_coords = (oc, n, ic, kh, kw, y, x)
                                    icoord = (n, kh + y*stride, kw + x*stride, ic)
                                    icoord_idx = np.ravel_multi_index([n, kh + y*stride, kw + x*stride, ic], inputs.shape)
                                    wcoord = (kh, kw, ic, oc)
                                    wcoord_idx = np.ravel_multi_index([kh, kw, ic, oc], weights.shape)
                                    ocoord = (n, y, x, oc)
                                    ocoord_idx = np.ravel_multi_index([n, y, x, oc], outputs.shape)
                                    compilation_info[ic].append(f'"{all_coords}", {ocoord_idx}, {icoord_idx}, {wcoord_idx}, {inputs[icoord]}, {weights[wcoord]}, {partial_sum}')
    return compilation_info


def manual_conv(inputs, weights, cdlt, o_coord, layout="nhwc"):
    if layout == "nhwc":
        N, IH, IW, IC = inputs.shape
        KH, KW, IC_, OC = weights.shape
        # KH, KW, OC, IC_ = weights.shape
        N_, OH, OW, OC_ = cdlt.outputs[0].shape
        out_shape = cdlt.outputs[0].shape
    else:
        N, IC, IH, IW = inputs.shape
        OC, IC_, KH, KW,  = weights.shape
        N_, OH, OW, OC_ = cdlt.outputs[0].shape
        out_shape = (N_, OC_, OH, OW)
    assert isinstance(o_coord, tuple) and len(o_coord) == 4
    assert N_ == N
    assert IC == IC_
    assert OC == OC_
    outputs = np.zeros(out_shape, dtype=np.int64)
    inputs = inputs.astype(np.int64)
    weights = weights.astype(np.int64)
    stride = cdlt.required_params['stride'].value
    compilation_info = {i: [] for i in range(IC)}
    if layout == "nhwc":
        for oc in range(OC):
            for n in range(N):
                for ic in range(IC):
                    for kh in range(KH):
                        for kw in range(KW):
                            for y in range(OH):
                                for x in range(OW):


                                    partial_sum = inputs[n, kh + y*stride, kw + x*stride, ic] * weights[kh, kw, ic, oc]
                                    outputs[n, y, x, oc] += partial_sum
                                    if (n, y, x, oc) == o_coord:
                                        all_coords = (oc, n, ic, kh, kw, y, x)
                                        icoord = (n, kh + y*stride, kw + x*stride, ic)
                                        icoord_idx = np.ravel_multi_index([n, kh + y*stride, kw + x*stride, ic], inputs.shape)
                                        wcoord = (kh, kw, ic, oc)
                                        wcoord_idx = np.ravel_multi_index([kh, kw, ic, oc], weights.shape)
                                        ocoord = (n, y, x, oc)
                                        ocoord_idx = np.ravel_multi_index([n, y, x, oc], outputs.shape)
                                        compilation_info[ic].append(f'"{all_coords}", {ocoord_idx}, {icoord_idx}, {wcoord_idx}, {inputs[icoord]}, {weights[wcoord]}, {partial_sum}')

                                    # outputs[n, y, x, oc] += inputs[n, kh + y*stride, kw + x*stride, ic] * weights[kh, kw, oc, ic]

    else:
        compilation_info = {}
        for oc in range(OC):
            for n in range(N):
                for ic in range(IC):
                    for kh in range(KH):
                        for kw in range(KW):
                            for y in range(OH):
                                for x in range(OW):
                                    outputs[n, oc, y, x] += inputs[n, ic, kh + y * stride, kw + x * stride] * weights[
                                        oc, ic, kh, kw]
        outputs = outputs.transpose(0, 2, 3, 1)

    return outputs, compilation_info

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

def conv_forward_im2col(x, w, b, conv_param):
    """
    A fast implementation of the forward pass for a convolutional layer
    based on im2col and col2im.
    """
    N, C, H, W = x.shape
    num_filters, _, filter_height, filter_width = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']

    # Check dimensions

    assert (W + 2 * pad - filter_width) % stride == 0, 'width does not work'
    assert (H + 2 * pad - filter_height) % stride == 0, 'height does not work'

    # Create output
    out_height = (H + 2 * pad - filter_height) // stride + 1
    out_width = (W + 2 * pad - filter_width) // stride + 1
    out = np.zeros((N, num_filters, out_height, out_width), dtype=x.dtype)

    # x_cols = im2col_indices(x, w.shape[2], w.shape[3], pad, stride)
    x_cols = im2col_indices(x, w.shape[2], w.shape[3], pad, stride)
    res = w.reshape((w.shape[0], -1)).dot(x_cols) + b.reshape(-1, 1)

    out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0])
    out = out.transpose(3, 0, 1, 2)

    cache = (x, w, b, conv_param, x_cols)
    return out, cache

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

def get_indices(X_shape, HF, WF, stride, pad):
    """
        Returns index matrices in order to transform our input image into a matrix.

        Parameters:
        -X_shape: Input image shape.
        -HF: filter height.
        -WF: filter width.
        -stride: stride value.
        -pad: padding value.

        Returns:
        -i: matrix of index i.
        -j: matrix of index j.
        -d: matrix of index d.
            (Use to mark delimitation for each channel
            during multi-dimensional arrays indexing).
    """
    # get input size
    m, n_C, n_H, n_W = X_shape

    # get output size
    out_h = int((n_H + 2 * pad - HF) / stride) + 1
    out_w = int((n_W + 2 * pad - WF) / stride) + 1

    # ----Compute matrix of index i----

    # Level 1 vector.
    level1 = np.repeat(np.arange(HF), WF)
    # Duplicate for the other channels.
    level1 = np.tile(level1, n_C)
    # Create a vector with an increase by 1 at each level.
    everyLevels = stride * np.repeat(np.arange(out_h), out_w)
    # Create matrix of index i at every levels for each channel.
    i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)

    # ----Compute matrix of index j----

    # Slide 1 vector.
    slide1 = np.tile(np.arange(WF), HF)
    # Duplicate for the other channels.
    slide1 = np.tile(slide1, n_C)
    # Create a vector with an increase by 1 at each slide.
    everySlides = stride * np.tile(np.arange(out_w), out_h)
    # Create matrix of index j at every slides for each channel.
    j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)

    # ----Compute matrix of index d----

    # This is to mark delimitation for each channel
    # during multi-dimensional arrays indexing.
    d = np.repeat(np.arange(n_C), HF * WF).reshape(-1, 1)

    return i, j, d


def im2col(X, HF, WF, stride, pad):
    """
        Transforms our input image into a matrix.

        Parameters:
        - X: input image.
        - HF: filter height.
        - WF: filter width.
        - stride: stride value.
        - pad: padding value.

        Returns:
        -cols: output matrix.
    """
    # Padding
    X_padded = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    i, j, d = get_indices(X.shape, HF, WF, stride, pad)
    # Multi-dimensional arrays indexing.
    cols = X_padded[:, d, i, j]
    cols = np.concatenate(cols, axis=-1)
    return cols


def compute_im2col_dims(params, oh, ow):
    if 'stride' in params:
        stride = params['stride']
    elif 'strides' in params:
        stride = params['strides']
    else:
        raise KeyError(f"Could not find stride {list(params.keys())}")
    if 'pads' in params:
        pad = params['pads']
    elif 'pad' in params:
        pad = params['pad']
    else:
        raise KeyError(f"Could not find stride {list(params.keys())}")
    input = np.random.randint(low=0, high=127, size=(params['N'], params['IC'], params['IH'], params['IW']), dtype=np.int32)
    weights = np.random.randint(low=0, high=127, size=(params['OC'], params['IC'], params['KH'], params['KW']), dtype=np.int32)
    bias = np.zeros(shape=params['OC'], dtype=np.int32)
    tout = F.conv2d(torch.from_numpy(input.astype(np.float64)), torch.from_numpy(weights.astype(np.float64)),
                    torch.from_numpy(bias.astype(np.float64)), stride=stride, padding=pad)
    M = oh*ow
    N = params['KH']*params['KW']*params['IC']
    P = params['OC']
    # x_cols = im2col_indices(input, weights.shape[2], weights.shape[3], 0, params['stride'])
    # im2col(X, HF, WF, stride, pad)
    x_cols = im2col(input, weights.shape[2], weights.shape[3], stride, pad)

    assert M == x_cols.shape[1] and N == x_cols.shape[0]
    torch_res = F.linear(torch.from_numpy(x_cols.transpose(1, 0).astype(np.float64)),
                         torch.from_numpy(weights.reshape((weights.shape[0], -1)).astype(np.float64)),
                    torch.from_numpy(bias.astype(np.float64)))\
        .resize(oh, ow, params['N'], params['OC']).permute(2, 3, 0, 1)
    res = weights.reshape((weights.shape[0], -1)).dot(x_cols) + bias.reshape(-1, 1)
    out = res.reshape(params['OC'], oh, ow, params['N'])
    out = out.transpose(3, 0, 1, 2)
    torch.testing.assert_allclose(torch.from_numpy(out.astype(np.float64)), tout)
    torch.testing.assert_allclose(torch_res, tout)
    return M, N, P

def check_conv_params(n, ic, oc, ih, iw, k, stride, pad):
    layer = nn.Conv2d(ic, oc, k, stride, pad)