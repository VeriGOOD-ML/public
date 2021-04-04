import numpy as np
from .systolic_array import SystolicArray
from collections import namedtuple

# CompilerParams = namedtuple('CompilerParams', [''])

def compute_tiling(t_method, dims, sys_dims):
    # dims = {"N": N, "M": M, "F": F, "E": E, "R": R, "S": S, "C": C}
    # dims = {"B": N, "K": M, "C": F, "Y": E, "X": R, "Fy": S, "Fx": C}

    if t_method == 'even':
        pass


def sys_naive_convolution(I, W, B, U, P, sys_array, comp_params):
    N = I.shape_symbols[0] # fmap batch size
    C = I.shape_symbols[1] # num input fmap/filter channels
    M = W.shape_symbols[0] # num of filters / # output fmap channels
    H_ = I.shape_symbols[-2] # input fmap height
    W_ = I.shape_symbols[-1] # input fmap bitwidth
    R = W.shape_symbols[-2] # filter height
    S = W.shape_symbols[-1] # filter bitwidth
    E = int((H_ - R + 2*P)/U) + 1 # output fmap height
    F = int((W_ - S + 2*P)/U) + 1# output fmap bitwidth
    O = np.empty((N, M, E, F))
    I = np.pad(I, ((0, 0), (0, 0), (P, P), (P, P)), mode='constant')

    # I = np.pad(I, ((0,0), (P, P), (P, P), (0,0)), 'constant', constant_values=(0,0))
    Nt  = N // 4
    Mt = M // 4
    Ft = F // 4
    Et = E // 4

    for nt in range(Nt):
        for mt in range(Mt):
            for xt in range(Ft):
                for yt in range(Et):
                    # Starting conv window
                    O[nt*Nt:(nt+1)*Nt, mt*Mt:(mt+1)*Mt, xt*Ft:(xt+1)*Ft, yt*Et:(yt+1)*Et] = B[mt*Mt:(mt+1)*Mt]
                    for nnt in range(nt*Nt, (nt+1)*Nt):
                        for mmt in range(mt*Mt, (mt+1)*Mt):
                            for xxt in range(xt*Ft, (xt+1)*Ft):
                                for yyt in range(yt*Et, (yt+1)*Et):
                                    #
                                    for i in range(R):
                                        for j in range(S):
                                            for k in range(C):
                                                O[nnt, mmt, xxt, yyt] += I[nnt, k, U*xxt + i, U*yyt + j] * W[mmt, k, i, j]
    return O



def generate_data(input_size, weight_size, zero_bias=False):
    data = np.random.randint(-3,3, input_size)
    weights = np.random.randint(-3,3, weight_size)
    if zero_bias:
        bias = np.zeros((weight_size[0]))
    else:
        bias = np.random.randint(-3, 3, (weight_size[0]))
    return data, weights, bias


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
  # First figure out what the size of the output should be
  N, C, H, W = x_shape
  assert (H + 2 * padding - field_height) % stride == 0
  assert (W + 2 * padding - field_height) % stride == 0
  out_height = int((H + 2 * padding - field_height) / stride + 1)
  out_width = int((W + 2 * padding - field_width) / stride + 1)

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

  k, i, j = get_im2col_indices(x.shape_symbols, field_height, field_width, padding,
                               stride)

  cols = x_padded[:, k, i, j]
  C = x.shape_symbols[1]
  cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
  return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
  """ An implementation of col2im based on fancy indexing and np.add.at """
  N, C, H, W = x_shape
  H_padded, W_padded = H + 2 * padding, W + 2 * padding
  x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
  k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                               stride)
  cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
  cols_reshaped = cols_reshaped.transpose(2, 0, 1)
  np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
  if padding == 0:
    return x_padded
  return x_padded[:, :, padding:-padding, padding:-padding]

def stanford_conv(x, w, b, stride, pad):
  """
  A fast implementation of the forward pass for a convolutional layer
  based on im2col and col2im.
  """
  N, C, H, W = x.shape_symbols
  num_filters, _, filter_height, filter_width = w.shape_symbols

  # Check dimensions
  assert (W + 2 * pad - filter_width) % stride == 0, 'bitwidth does not work'
  assert (H + 2 * pad - filter_height) % stride == 0, 'height does not work'

  # Create output
  out_height = int((H + 2 * pad - filter_height) / stride + 1)
  out_width = int((W + 2 * pad - filter_width) / stride + 1)
  out = np.zeros((N, num_filters, out_height, out_width), dtype=x.dtype)

  x_cols = im2col_indices(x, w.shape_symbols[2], w.shape_symbols[3], pad, stride)
  # x_cols = im2col_cython(x, w.shape_symbols[2], w.shape_symbols[3], pad, stride)
  res = w.reshape((w.shape_symbols[0], -1)).dot(x_cols) + b.reshape(-1, 1)

  out = res.reshape(w.shape_symbols[0], out.shape[2], out.shape[3], x.shape_symbols[0])
  out = out.transpose(3, 0, 1, 2)

  return out

def istell_naive_convolution(I, W, Bs, U, P):

    B = I.shape_symbols[0] # fmap batch size
    C = I.shape_symbols[1] # num input fmap/filter channels
    K = W.shape_symbols[0] # num of filters / # output fmap channels
    W_ = I.shape_symbols[-2] # input fmap height
    H_ = I.shape_symbols[-1] # input fmap bitwidth
    Fx = W.shape_symbols[-2] # filter height
    Fy = W.shape_symbols[-1] # filter bitwidth
    Y = int((H_ - Fy + 2*P)/U) + 1# output fmap height
    X = int((W_ - Fx + 2*P)/U) + 1 # output fmap bitwidth
    # O = np.empty((B, K, X, Y))
    O = np.zeros((B, K, X, Y))
    I = np.pad(I, ((0, 0), (0, 0), (P, P), (P, P)), mode='constant')
    # print(f"Stride: {U}\tShape: {O.shape_symbols}\t{I.shape_symbols}\t{W.shape_symbols}")

    # tile size dependant on storage size
    # I = np.pad(I, ((0,0), (P, P), (P, P), (0,0)), 'constant', constant_values=(0,0))
    for b in range(B):
        for y in range(Y):
            for x in range(X):
                for k in range(K):
                    for c in range(C):

                    # O[b, k, x, y] = Bs[k]

                    # Starting conv window
                    # Populate buffers
                        for fy in range(Fy): # Replace loops with IB
                            for fx in range(Fx):
                                O[b, k, x, y] += I[b, c, fx + U*x, fy + U*y] * W[k, c, fx, fy]
                        # Write back
    return O

def naive_convolution(I, W, B, U, P):

    N = I.shape_symbols[0] # fmap batch size
    C = I.shape_symbols[1] # num input fmap/filter channels
    M = W.shape_symbols[0] # num of filters / # output fmap channels
    H_ = I.shape_symbols[-2] # input fmap height
    W_ = I.shape_symbols[-1] # input fmap bitwidth
    R = W.shape_symbols[-2] # filter height
    S = W.shape_symbols[-1] # filter bitwidth
    E = int((H_ - R + 2*P)/U) + 1 # output fmap height
    F = int((W_ - S + 2*P)/U) + 1# output fmap bitwidth
    O = np.empty((N, M, E, F))
    I = np.pad(I, ((0, 0), (0, 0), (P, P), (P, P)), mode='constant')
    # print(f"Stride: {U}\tShape: {O.shape_symbols}\t{I.shape_symbols}\t{W.shape_symbols}")

    # tile size dependant on storage size
    # I = np.pad(I, ((0,0), (P, P), (P, P), (0,0)), 'constant', constant_values=(0,0))
    for n in range(N):
        for m in range(M):
            for x in range(F):
                for y in range(E):
                    # Starting conv window
                    # Populate buffers
                    O[n, m, x, y] = B[m]

                    for i in range(R): # Replace loops with IB
                        for j in range(S):
                            for k in range(C):
                                O[n, m, x, y] += I[n, k, U*x + i, U*y + j] * W[m, k, i, j]
                    # Write back
    return O


def naive_convolution_ws(I, W, B, U, P):
    lenet = ((1,1,28,28), (6,1,5,5), 1, 0)

    N = I.shape_symbols[0] # fmap batch size
    C = I.shape_symbols[1] # num input fmap/filter channels
    M = W.shape_symbols[0] # num of filters / # output fmap channels
    H_ = I.shape_symbols[-2] # input fmap height
    W_ = I.shape_symbols[-1] # input fmap bitwidth
    R = W.shape_symbols[-2] # filter height
    S = W.shape_symbols[-1] # filter bitwidth
    E = int((H_ - R + 2*P)/U) + 1 # output fmap height
    F = int((W_ - S + 2*P)/U) + 1# output fmap bitwidth
    O = np.empty((N, M, E, F))
    I = np.pad(I, ((0, 0), (0, 0), (P, P), (P, P)), mode='constant')
    # print(f"Stride: {U}\tShape: {O.shape_symbols}\t{I.shape_symbols}\t{W.shape_symbols}")
    # O[n, m, x, y] = B[m]

    # tile size dependant on storage size
    # I = np.pad(I, ((0,0), (P, P), (P, P), (0,0)), 'constant', constant_values=(0,0))
    for m in range(M):
        for i in range(R): # Replace loops with IB
            for j in range(S):
                for k in range(C):
                    for n in range(N):
                        for y in range(E):
                            for x in range(F):
                                O[n, m, x, y] += I[n, k, U*x + i, U*y + j] * W[m, k, i, j]
                    # Write back
    return O

def naive_mmul_conv(I, W, Bs, U, P):
    B = I.shape_symbols[0] # fmap batch size
    C = I.shape_symbols[1] # num input fmap/filter channels
    K = W.shape_symbols[0] # num of filters / # output fmap channels
    W_ = I.shape_symbols[-2] # input fmap height
    H_ = I.shape_symbols[-1] # input fmap bitwidth
    Fx = W.shape_symbols[-2] # filter height
    Fy = W.shape_symbols[-1] # filter bitwidth
    Y = int((H_ - Fy + 2*P)/U) + 1# output fmap height
    X = int((W_ - Fx + 2*P)/U) + 1 # output fmap bitwidth
    # O = np.empty((B, K, X, Y))
    O = np.zeros((B, K, X, Y))
    I = np.pad(I, ((0, 0), (0, 0), (P, P), (P, P)), mode='constant')


    for b in range(B):
        for k in range(K):
            for c in range(C):
                #load source over channels
                for y in range(Y):
                    for x in range(X):
                        I_fx_ind = np.repeat(np.arange(Fx), Fy) + U * np.repeat(x, Fx*Fy)
                        I_fy_ind = np.repeat(np.arange(Fy).reshape(1,-1), Fx, axis=0).reshape(1,-1) + U * np.repeat(y, Fx*Fy)

                        W_fx_ind = np.repeat(np.arange(Fx), Fy)
                        W_fy_ind = np.repeat(np.arange(Fy).reshape(1,-1), Fx, axis=0).reshape(1,-1)
                        M = I[b,c, I_fx_ind, I_fy_ind].reshape(-1)
                        N = W[k, c, W_fx_ind, W_fy_ind].reshape(-1)

                        O[b,k,x,y] += M.dot(N)

    return O






def conv_forward_im2col(x, w, b,  stride, pad):
  """
  A fast implementation of the forward pass for a convolutional layer
  based on im2col and col2im.
  """
  N, C, H, W = x.shape_symbols
  num_filters, _, filter_height, filter_width = w.shape_symbols

  # Check dimensions
  assert (W + 2 * pad - filter_width) % stride == 0, 'bitwidth does not work'
  assert (H + 2 * pad - filter_height) % stride == 0, 'height does not work'

  # Create output
  out_height = int((H + 2 * pad - filter_height) / stride + 1)
  out_width = int((W + 2 * pad - filter_width) / stride + 1)
  out = np.zeros((N, num_filters, out_height, out_width), dtype=x.dtype)

  x_cols = im2col_indices(x, w.shape_symbols[2], w.shape_symbols[3], pad, stride)
  # x_cols = im2col_cython(x, w.shape_symbols[2], w.shape_symbols[3], pad, stride)
  res = w.reshape((w.shape_symbols[0], -1)).dot(x_cols) + b.reshape(-1, 1)

  out = res.reshape(w.shape_symbols[0], out.shape[2], out.shape[3], x.shape_symbols[0])
  out = out.transpose(3, 0, 1, 2)

  return out


def conv_forward_strides(x, w, b,  stride, pad):
  N, C, H, W = x.shape_symbols
  F, _, HH, WW = w.shape_symbols
  # Check dimensions
  #assert (W + 2 * pad - WW) % stride == 0, 'bitwidth does not work'
  #assert (H + 2 * pad - HH) % stride == 0, 'height does not work'

  # Pad the input
  p = pad
  x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

  # Figure out output dimensions
  H += 2 * pad
  W += 2 * pad
  out_h = int((H - HH) / stride + 1)
  out_w = int((W - WW) / stride + 1)

  # Perform an im2col operand by picking clever strides
  shape = (C, HH, WW, N, out_h, out_w)
  strides = (H * W, W, 1, C * H * W, stride * W, stride)
  strides = x.itemsize * np.array(strides)
  x_stride = np.lib.stride_tricks.as_strided(x_padded,
                shape=shape, strides=strides)
  x_cols = np.ascontiguousarray(x_stride)
  x_cols.shape = (C * HH * WW, N * out_h * out_w)

  # Now all our convolutions are a big matrix multiply
  res = w.reshape(F, -1).dot(x_cols) + b.reshape(-1, 1)

  # Reshape the output
  res.shape_symbols = (F, N, out_h, out_w)
  out = res.transpose(1, 0, 2, 3)

  # Be nice and return a contiguous array
  # The old version of conv_forward_fast doesn't do this, so for a fair
  # comparison we won't either
  out = np.ascontiguousarray(out)

  return out

def conv_einsum(I, W, B, U, P):
    B = I.shape_symbols[0] # fmap batch size
    C = I.shape_symbols[1] # num input fmap/filter channels
    K = W.shape_symbols[0] # num of filters / # output fmap channels
    W_ = I.shape_symbols[-2] # input fmap height
    H_ = I.shape_symbols[-1] # input fmap bitwidth
    Fx = W.shape_symbols[-2] # filter height
    Fy = W.shape_symbols[-1] # filter bitwidth
    Y = int((H_ - Fy + 2*P)/U) + 1# output fmap height
    X = int((W_ - Fx + 2*P)/U) + 1 # output fmap bitwidth
    # O = np.empty((B, K, X, Y))
    O = np.zeros((B, K, X, Y))
    # strided_x = np.lib.stride_tricks.as_strided(I,
    #                                             (B, C, H_ - K + 1, W_ - K + 1, K, K),
    #                                             (H_ * W_ * C, H_ * W_, W_, 1, W_, 1))
    # strided_x = np.lib.stride_tricks.as_strided(I,
    #                                             (B, C, Y, X, K, K),
    #                                             (H_ * W_ * C, H_ * W_, W_, 1, W_, 1))
    # strided_x = np.lib.stride_tricks.as_strided(I, (B, C, H_ - Fy + 1, W_ - Fx + 1, Fy, Fx), (H_ * W_ * C, H_ * W_, W_, 1, W_, 1))
    # strided_x = np.lib.stride_tricks.as_strided(I, (B, C, Y - Fy + 1, X - Fx + 1, Fy, Fx), (Y * X * C, Y * X, X, 1, X, 1))
    # strided_x = np.lib.stride_tricks.as_strided(I, (B, C, Y, X, Fy, Fx), (Y * X * C, Y * X, X, 1, X, 1))

    # I = np.pad(I, ((0, 0), (0, 0), (P, P), (P, P)), mode='constant')
    s0, s1 = I.strides[-2:]
    matrix_dims = np.ndim(I)
    view_shape = I.shape_symbols[:2 - matrix_dims] + (X, Y, Fx, Fy)
    strides = I.strides[:2-matrix_dims] + (U * s0, U * s1, s0, s1)
    sub_shape = tuple(np.subtract(I.shape_symbols, W.shape_symbols) + 1)
    I = np.lib.stride_tricks.as_strided(I, view_shape, strides=strides)
    # out = np.einsum('hij,hijklm->klm', W, strd)

    out = np.einsum('bihwkl,oikl->bohw', I, W)


    # for b in range(B):
    #     for k in range(K):
    #         for c in range(C):
    #             #load source over channels
    #             for y in range(Y):
    #                 for x in range(X):
    #                     I_fx_ind = np.repeat(np.arange(Fx), Fy) + U * np.repeat(x, Fx*Fy)
    #                     I_fy_ind = np.repeat(np.arange(Fy).reshape(1,-1), Fx, axis=0).reshape(1,-1) + U * np.repeat(y, Fx*Fy)
    #
    #                     W_fx_ind = np.repeat(np.arange(Fx), Fy)
    #                     W_fy_ind = np.repeat(np.arange(Fy).reshape(1,-1), Fx, axis=0).reshape(1,-1)
    #                     M = I[b,c, I_fx_ind, I_fy_ind].reshape(-1)
    #                     N = W[k, c, W_fx_ind, W_fy_ind].reshape(-1)
    #
    #                     O[b,k,x,y] += M.dot(N)
    # print(out.shape_symbols)
    return I


def main():
    sys_array = SystolicArray(32)
    I = np.random.randint(-3,3, (32, 32))
    W = np.random.randint(-3,3, (32, 32))
    B = np.random.randint(-3,3, 32)


    params = []
    # # Lenet-5
    # lenet = ((1,1,28,28), (6,1,5,5), 1, 0)
    # params.append(lenet)
    # #
    #
    # # Simple
    # naive_simple1 = ((2,3,4,4), (3,3,4,4), 2, 1)
    naive_simple1 = ((100,3,31,31), (25,3,3,3), 2, 1)
    # naive_simple2 = ((4,3,5,5), (2,3,3,3), 1, 1)
    params.append(naive_simple1)
    # params.append(naive_simple2)

    # fast_simple1 = ((100,3,31,31), (25,3,3,3), 2, 1)
    # fast_simple2 = ((2,3,16,16), (3,3,3,3), 1, 1)
    # params.append(fast_simple1)
    # params.append(fast_simple2)
    #
    #
    #
    for p in params:
        I, W, B = generate_data(p[0], p[1], zero_bias=True)
        einsum_O = conv_einsum(I, W, B, p[2], p[3])
        print(einsum_O.shape)

        # stanford_O = stanford_conv(I, W, B, p[2], p[3])

        stanford_O = conv_forward_strides(I, W, B, p[2], p[3])

        naive_O = naive_convolution(I, W, B, p[2], p[3])
        print(stanford_O.shape)
        print(naive_O.shape)
        # print(einsum_O.shape_symbols)
        # naive_O = naive_mmul_conv(I, W, B, p[2], p[3])

        # np.testing.assert_allclose(naive_O, stanford_O)
        # np.testing.assert_allclose(einsum_O, stanford_O)


if __name__ == "__main__":
    main()
