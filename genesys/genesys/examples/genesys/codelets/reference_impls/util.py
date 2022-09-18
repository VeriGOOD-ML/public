import numpy as np
from fxpmath import Fxp
from examples.genesys import FXP_CONFIGS

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