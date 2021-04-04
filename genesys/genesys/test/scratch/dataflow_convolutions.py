import numpy as np
from systolic_array import SystolicArray
from collections import namedtuple
from convolution import generate_data, istell_naive_convolution, conv_forward_strides

def os_load_inputs(W, SA_SIZE, offsets, h_fold):
    start = np.prod(offsets)

    for i in range(SA_SIZE):
        for j in range(SA_SIZE):
            if offsets[2] + i >= SA_SIZE:
                offsets[2] = 0
                offsets[1] += 1
            A[i,j] = W[offsets[0], offsets[1], offsets[2] + i, offsets[3] + j]
    return A

def os_load_weights(I, SA_SIZE, offsets, h_fold):
    start = np.prod(offsets)
    # if
    A = np.zeros((SA_SIZE, SA_SIZe))
    for i in range(SA_SIZE):
        for j in range(SA_SIZE):
            I[i,j] = W[offsets[0], offsets[1], offsets[2] + i, offsets[3] + j]
    return A

def ow_load_weights(W, SA_SIZE, offsets):
    pass

def weight_stationary(I, W, Bs, U, P, sys_array):
    SA_SIZE, BUF_SIZE = sys_array.array_size, sys_array.ibuf_size

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

def output_stationary(I, W, Bs, U, P, sys_array):
    SA_SIZE, BUF_SIZE = sys_array.array_size, sys_array.ibuf_size

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
    column_size = (Fx*Fy*C)
    row_size = ()
    h_unroll = np.ceil( column_size / sys_array.array_size)

    # channels_unrolled =np.ceil( / (sys_array.array_size**2))

    for b in range(B):
        for k in range(K):
                #load source over channels
            for y in range(Y):
                for x in range(X):

                    for c in range(h_unroll):
                        sys_array.load_obuf()

                        # I_fx_ind = np.repeat(np.arange(Fx), Fy) + U * np.repeat(x, Fx*Fy)
                        # I_fy_ind = np.repeat(np.arange(Fy).reshape(1,-1), Fx, axis=0).reshape(1,-1) + U * np.repeat(y, Fx*Fy)
                        #
                        # W_fx_ind = np.repeat(np.arange(Fx), Fy)
                        # W_fy_ind = np.repeat(np.arange(Fy).reshape(1,-1), Fx, axis=0).reshape(1,-1)
                        # M = I[b,c, I_fx_ind, I_fy_ind].reshape(-1)
                        # N = W[k, c, W_fx_ind, W_fy_ind].reshape(-1)
                        # # print(M.shape_symbols)
                        # # print(N.shape_symbols)
                        O[b,k,x,y] += M.dot(N)
    return O






def main():
    sarray_size = 32

    sys_array = SystolicArray(sarray_size)
    a1 = np.random.randint(-3,3, sarray_size*sarray_size).reshape((sarray_size,sarray_size))
    a2 = np.random.randint(-3,3, sarray_size*sarray_size).reshape((sarray_size,sarray_size))
    a3 = np.random.randint(-3,3, sarray_size*sarray_size).reshape((sarray_size,sarray_size))
    a4 = np.random.randint(-3,3, sarray_size*sarray_size).reshape((sarray_size,sarray_size))

    b1 = np.random.randint(-3,3, sarray_size*sarray_size).reshape((sarray_size,sarray_size))
    b2 = np.random.randint(-3,3, sarray_size*sarray_size).reshape((sarray_size,sarray_size))
    b3 = np.random.randint(-3,3, sarray_size*sarray_size).reshape((sarray_size,sarray_size))
    b4 = np.random.randint(-3,3, sarray_size*sarray_size).reshape((sarray_size,sarray_size))

    c1 = np.zeros((sarray_size, sarray_size)) + 1
    c2 = np.zeros((sarray_size, sarray_size)) + 1
    c3 = np.zeros((sarray_size, sarray_size)) + 1
    c4 = np.zeros((sarray_size, sarray_size)) + 1
    sys_array.load_ibuf(np.concatenate((a1,a2,a3,a4)))
    sys_array.load_wbuf(np.concatenate((b1,b2,b3,b4)))
    sys_array.load_obuf(np.concatenate((c1,c2,c3,c4)))


    # a1 b1
    sys_array.read_ibuf(0)
    sys_array.read_wbuf(0)
    sys_array.read_obuf(0)
    res = sys_array.run()

    np.testing.assert_allclose(res, a1.dot(b1) + 1)

    # a2 b2
    sys_array.read_ibuf(sarray_size*1)
    sys_array.read_wbuf(sarray_size*1)
    sys_array.read_obuf(sarray_size*1)
    res = sys_array.run()

    np.testing.assert_allclose(res, a2.dot(b2) + 1)


    # params = []
    # # # # Lenet-5
    # lenet = ((1,1,28,28), (6,1,5,5), 1, 0)
    # params.append(lenet)
    # # #
    # # #
    # # # # Simple
    # # # naive_simple1 = ((2,3,4,4), (3,3,4,4), 2, 1)
    # # # naive_simple2 = ((4,3,5,5), (2,3,3,3), 1, 1)
    # # # params.append(naive_simple1)
    # # # params.append(naive_simple2)
    # # #
    # # # fast_simple1 = ((100,3,31,31), (25,3,3,3), 2, 1)
    # # # fast_simple2 = ((2,3,16,16), (3,3,3,3), 1, 1)
    # # # params.append(fast_simple1)
    # # # params.append(fast_simple2)
    # # #
    # # #
    # # #
    # for p in params:
    #     I, W, B = generate_data(p[0], p[1], zero_bias=True)
    #     # stanford_O = stanford_conv(I, W, B, p[2], p[3])
    #     stanford_O = conv_forward_strides(I, W, B, p[2], p[3])
    #     # naive_O = naive_convolution(I, W, B, p[2], p[3])
    #     naive_O = output_stationary(I, W, B, p[2], p[3], sys_array)
    #     np.testing.assert_allclose(naive_O, stanford_O)

if __name__ == "__main__":
    main()
