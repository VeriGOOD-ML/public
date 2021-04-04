import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import numpy as np
from collections import defaultdict
from scratch.convolution import stanford_conv
# from test.scra

NUM_WT = 4
NUM_IT = 2
NUM_OT = 2
NUM_BT = 1

WT_SIZE = 32
IT_SIZE = 32
OT_SIZE = 32
BT_SIZE = 16

def _preprocess(I_shape, W_shape, U, P):
    I = np.random.randint(128, size=I_shape, dtype=np.int)
    W = np.random.randint(128, size=W_shape, dtype=np.int)
    # B = np.random.randint(128, size=W_shape[0], dtype=np.int)
    B = np.zeros(W_shape[0], dtype=np.int)
    N = I.shape[0] # fmap batch size
    C = I.shape[1] # num input fmap/filter channels
    M = W.shape[0] # num of filters / # output fmap channels
    H_ = I.shape[-2] # input fmap height
    W_ = I.shape[-1] # input fmap bitwidth
    R = W.shape[-2] # filter height
    S = W.shape[-1] # filter bitwidth
    E = int((H_ - R + 2*P)/U) + 1 # output fmap height
    F = int((W_ - S + 2*P)/U) + 1# output fmap bitwidth
    O = np.empty((N, M, E, F), dtype=I.dtype)
    I = np.pad(I, ((0, 0), (0, 0), (P, P), (P, P)), mode='constant')
    return I, W, O, B

def verification_convolution(I, W, B, U, P):
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

    for n in range(N):
        for m in range(M):
            for x in range(F):
                for y in range(E):
                    # Starting conv window
                    # Populate buffers
                    # O[n, m, x, y] = B[m]
                    O[n, m, x, y] = 0
                    for i in range(R): # Replace loops with IB
                        for j in range(S):
                            for k in range(C):
                                O[n, m, x, y] += I[n, k, U*x + i, U*y + j] * W[m, k, i, j]
                    # Write back
    return O

def _convolution(I, W, B, U, P):
    N = I.shape_symbols[0] # fmap batch size
    IC = I.shape_symbols[1] # num input fmap/filter channels
    OC = W.shape_symbols[0] # num of filters / # output fmap channels
    H_ = I.shape_symbols[-2] # input fmap height
    W_ = I.shape_symbols[-1] # input fmap bitwidth
    KH = W.shape_symbols[-2] # filter height
    KW = W.shape_symbols[-1] # filter bitwidth
    OH = int((H_ - KH + 2*P)/U) + 1 # output fmap height
    OW = int((W_ - KW + 2*P)/U) + 1# output fmap bitwidth
    O = np.empty((N, OC, OH, OW))
    I = np.pad(I, ((0, 0), (0, 0), (P, P), (P, P)), mode='constant')
    summation_map = defaultdict(list)
    for n in range(N):
        for x in range(OH):
            for y in range(OW):
                for oc in range(OC):
                    O[n, oc, x, y] = B[oc]
    from pprint import pprint
    for oc in range(OC):
        for n in range(N):
            for ic in range(IC):
                for kh in range(KH):
                    for kw in range(KW):
                        for x in range(OW):
                            for y in range(OH):
                                # inp_idx = (n, ic, U*x + kh, U*y + kw)
                                # wt_idx = (oc, ic, kh, kw)
                                # out_idx = (n, oc, x, y)
                                # summation_map[out_idx].append({'I': inp_idx, 'W': wt_idx})
                                O[n, oc, x, y] += I[n, ic, U*x + kh, U*y + kw] * W[oc, ic, kh, kw]
    pprint(summation_map)
                    # Write back
    return O

def execute_tile(all_tiling, tiles, stride, pe_dims):
    IBUF_TILE, WBUF_TILE, OBUF_TILE = tiles
    CONV_STRIDE = stride
    w_mat = np.zeros(pe_dims).reshape(-1)
    i_mat = np.zeros(pe_dims[0])
    o_mat = np.zeros(pe_dims[1])
    i_mat_idx = 0
    w_mat_idx = 0
    o_mat_idx = 0

    for oc in range(all_tiling["OC"]):
        for n in range(all_tiling["N"]):
            for ic in range(all_tiling["IC"]):
                for kh in range(all_tiling["KH"]):
                    for kw in range(all_tiling["KW"]):
                        WBUF_BASE_ADDR = oc * all_tiling["IC"] * all_tiling["KW"] * all_tiling["KW"]
                        WBUF_BASE_ADDR += ic * all_tiling["KW"] * all_tiling["KW"]
                        WBUF_BASE_ADDR += kh * all_tiling["KW"]
                        WBUF_BASE_ADDR += kw
                        w_mat[w_mat_idx % np.prod(pe_dims)] = WBUF_TILE[WBUF_BASE_ADDR]
                        w_mat_idx += 1
                        for y in range(all_tiling["OH"]):
                            for x in range(all_tiling["OW"]):
                                IBUF_BASE_ADDR = n * all_tiling["IH"] * all_tiling["IW"] * all_tiling["IC"]
                                IBUF_BASE_ADDR += kh * all_tiling["IW"] * all_tiling["IC"]
                                IBUF_BASE_ADDR += y * all_tiling["IW"] * all_tiling["IC"] * CONV_STRIDE
                                IBUF_BASE_ADDR += kw * all_tiling["IC"]
                                IBUF_BASE_ADDR += x * all_tiling["IC"] * CONV_STRIDE
                                IBUF_BASE_ADDR += ic
                                i_mat[i_mat_idx % pe_dims[0]] = IBUF_TILE[IBUF_BASE_ADDR]
                                i_mat_idx += 1
                                # i_mat[][]

                                WBUF_BASE_ADDR = oc * all_tiling["IC"] * all_tiling["KW"] * all_tiling["KW"]
                                WBUF_BASE_ADDR += ic * all_tiling["KW"] * all_tiling["KW"]
                                WBUF_BASE_ADDR += kh * all_tiling["KW"]
                                WBUF_BASE_ADDR += kw

                                OBUF_BASE_ADDR = n * all_tiling["OH"] * all_tiling["OW"] * all_tiling["OC"]
                                OBUF_BASE_ADDR += y * all_tiling["OW"] * all_tiling["OC"]
                                OBUF_BASE_ADDR += x * all_tiling["OC"]
                                OBUF_BASE_ADDR += oc
                                # if ic ==0 and oc == 0:
                                # print(WBUF_BASE_ADDR)
                                # w_mat[oc][ic] = WBUF_TILE[WBUF_BASE_ADDR]
                                # i_mat[ic] = IBUF_TILE[IBUF_BASE_ADDR]
                                # o_mat[oc] = OBUF_TILE[OBUF_BASE_ADDR]
                                # if
                                OBUF_TILE[OBUF_BASE_ADDR] += IBUF_TILE[IBUF_BASE_ADDR] * WBUF_TILE[WBUF_BASE_ADDR]
            # o_mat[oc] += np.dot(w_mat, i_mat)
    print()
    return OBUF_TILE



def store_output_tile(all_tiling, output_shape, O_BASE_ADDR, OBUF_TILE, O):
    N, OH, OW, OC = output_shape
    for n in range(all_tiling["N"]):
        for y in range(all_tiling["OH"]):
            for x in range(all_tiling["OW"]):
                for oc in range(all_tiling["OC"]):
                    OBUF_BASE_ADDR = n * all_tiling["OH"] * all_tiling["OW"] * all_tiling["OC"]
                    OBUF_BASE_ADDR += y * all_tiling["OW"] * all_tiling["OC"]
                    OBUF_BASE_ADDR += x * all_tiling["OC"]
                    OBUF_BASE_ADDR += oc
                    OUTPUT_BASE_OFFSET = n * OH * OW * OC
                    OUTPUT_BASE_OFFSET += y * OW * OC
                    OUTPUT_BASE_OFFSET += x * OC
                    OUTPUT_BASE_OFFSET += oc
                    O[O_BASE_ADDR + OUTPUT_BASE_OFFSET] = OBUF_TILE[OBUF_BASE_ADDR]
    return O

def load_output_tile(all_tiling, output_shape, O_BASE_ADDR, OBUF_TILE, O):
    N, OH, OW, OC = output_shape
    for n in range(all_tiling["N"]):
        for y in range(all_tiling["OH"]):
            for x in range(all_tiling["OW"]):
                for oc in range(all_tiling["OC"]):
                    OBUF_BASE_ADDR = n * all_tiling["OH"] * all_tiling["OW"] * all_tiling["OC"]
                    OBUF_BASE_ADDR += y * all_tiling["OW"] * all_tiling["OC"]
                    OBUF_BASE_ADDR += x * all_tiling["OC"]
                    OBUF_BASE_ADDR += oc
                    OUTPUT_BASE_OFFSET = n * OH * OW * OC
                    OUTPUT_BASE_OFFSET += y * OW * OC
                    OUTPUT_BASE_OFFSET += x * OC
                    OUTPUT_BASE_OFFSET += oc
                    OBUF_TILE[OBUF_BASE_ADDR] = O[O_BASE_ADDR + OUTPUT_BASE_OFFSET]
    return OBUF_TILE

def load_weight_tile(all_tiling, W_BASE_ADDR, WBUF_TILE, W):
    weight_tile_size = int(np.prod((all_tiling["OC"], all_tiling["IC"], all_tiling["KH"], all_tiling["KW"])))
    for w in range(weight_tile_size):
        WBUF_TILE[w] = W[W_BASE_ADDR + w].copy()
    return WBUF_TILE

def load_input_tile(all_tiling, stride, input_shape, I_BASE_ADDR, IBUF_TILE, I):
    CONV_STRIDE = stride
    N, IH, IW, IC = input_shape
    DRAM_ADDR = []
    IBUF_ADDR = []

    for n in range(all_tiling["N"]):
        for ih in range(all_tiling["IH"]):
            for iw in range(all_tiling["IW"]):
                for ic in range(all_tiling["IC"]):
                    IBUF_BASE_ADDR = n * all_tiling["IH"] * all_tiling["IW"] * all_tiling["IC"]
                    IBUF_BASE_ADDR += ih * all_tiling["IW"] * all_tiling["IC"]
                    IBUF_BASE_ADDR += iw * all_tiling["IC"]
                    IBUF_BASE_ADDR += ic

                    INPUT_BASE_OFFSET = n * IH * IW * IC
                    INPUT_BASE_OFFSET += ih * IW * IC
                    INPUT_BASE_OFFSET += iw * IC
                    INPUT_BASE_OFFSET += ic
                    IBUF_TILE[IBUF_BASE_ADDR] = I[I_BASE_ADDR + INPUT_BASE_OFFSET]

    return IBUF_TILE


def sys_array_conv(I, W, B, O, stride, pad):
    N_ROWS = 32
    M_COLS = 32
    CONV_STRIDE = stride

    I = I.transpose((0, 2, 3, 1))
    O = O.transpose((0, 2, 3, 1))

    N = I.shape_symbols[0] # fmap batch size
    IC = I.shape_symbols[3] # num input fmap/filter channels
    IH = I.shape_symbols[1] # input height
    IW = I.shape_symbols[2] # input width
    OC = W.shape_symbols[0] # num of filters / # output fmap channels
    KH = W.shape_symbols[-2] # filter height
    KW = W.shape_symbols[-1] # filter bitwidth
    OH = O.shape_symbols[-3]
    OW = O.shape_symbols[-2]
    shapes = {"N": N, "IC": IC, "IW": IW, "IH": IH, "OH": OH, "OW": OW, "OC": OC, "KH": KH, "KW": KW}
    tiling_factors = {"N": 1, "IC": 1, "OH": 1, "OW": 1, "KH": 1, "KW": 1, "OC": 1}
    all_tiling = {}
    for k, v in tiling_factors.items():
        assert shapes[k] % tiling_factors[k] == 0
        all_tiling[k] = int(shapes[k]/tiling_factors[k])
    all_tiling["IH"] = (all_tiling["OH"] - 1) * stride - 2*pad + all_tiling["KH"]
    all_tiling["IW"] = (all_tiling["OW"] - 1) * stride - 2*pad + all_tiling["KW"]
    output_tiling = (all_tiling["N"], all_tiling["OH"], all_tiling["OW"], all_tiling["OC"])
    weight_tiling = (all_tiling["OC"], all_tiling["IC"], all_tiling["KH"], all_tiling["KW"])

    input_tiling = (all_tiling["N"], all_tiling["IH"], all_tiling["IW"], all_tiling["IC"])
    input_tile_size = int(np.prod(input_tiling).astype(np.int))
    output_tile_size = int(np.prod(output_tiling).astype(np.int))
    weight_tile_size = int(np.prod(weight_tiling).astype(np.int))

    for n in range(N):
        for x in range(OH):
            for y in range(OW):
                for oc in range(OC):
                    O[n, x, y, oc] = B[oc]
    I_orig = I.copy()
    O_orig = O.copy()
    W_orig = W.copy()
    O = O.reshape(-1)
    I = I.reshape(-1)
    W = W.reshape(-1)

    W_BASE_ADDR = 0
    I_BASE_ADDR = 0
    O_BASE_ADDR = 0

    IBUF_TILE = np.zeros(input_tile_size, dtype=I.dtype)
    WBUF_TILE = np.zeros(weight_tile_size, dtype=W.dtype)
    OBUF_TILE = np.zeros(output_tile_size, dtype=O.dtype)
    W_TILE_OC_STRIDE = all_tiling["OC"]*IC*KW*KH
    W_TILE_IC_STRIDE = all_tiling["IC"]*KH*KW
    W_TILE_KH_STRIDE = all_tiling["KH"]*KW
    W_TILE_KW_STRIDE = all_tiling["KW"]

    print(W_TILE_OC_STRIDE)
    print(W_TILE_IC_STRIDE)
    print(W_TILE_KH_STRIDE)
    print(W_TILE_KW_STRIDE)
    print()

    # I_TILE_N_STRIDE = all_tiling["N"]*IH*IW*IC
    # I_TILE_OH_STRIDE = all_tiling["OH"]*IW*IC*CONV_STRIDE
    # I_TILE_KH_STRIDE = all_tiling["KH"]*IW*IC
    # I_TILE_OW_STRIDE = all_tiling["OW"]*IC*CONV_STRIDE
    # I_TILE_KW_STRIDE = all_tiling["KW"]*IC
    # I_TILE_IC_STRIDE = all_tiling["IC"]
    # I = [N, IC, IH, IW]
    # IC*IH*IW
    # I[IC*OH*stride + KH]
    # I_IBUF[
    I_TILE_N_STRIDE = all_tiling["N"]*IH*IW*IC
    I_TILE_IC_STRIDE = all_tiling["IC"]*IH*IW
    I_TILE_OH_STRIDE = all_tiling["OH"]*IW*CONV_STRIDE
    I_TILE_KH_STRIDE = all_tiling["KH"]*IW
    I_TILE_OW_STRIDE = all_tiling["OW"]*CONV_STRIDE
    I_TILE_KW_STRIDE = all_tiling["KW"]


    O_TILE_N_STRIDE = all_tiling["N"]*OH*OW*OC
    O_TILE_OH_STRIDE = all_tiling["OH"]*OW*OC
    O_TILE_OW_STRIDE = all_tiling["OW"]*OC
    O_TILE_OC_STRIDE = all_tiling["OC"]
    print(I_TILE_N_STRIDE)
    print(I_TILE_IC_STRIDE)
    print(I_TILE_OH_STRIDE)
    print(I_TILE_KH_STRIDE)
    print(I_TILE_OW_STRIDE)
    print(I_TILE_KW_STRIDE)

    for oc_tile in range(tiling_factors["OC"]): # SA_LOOP 1, 0, num_oc_tiles
        for n_tile in range(tiling_factors["N"]): # SA_LOOP 2, 1, num_n_tiles
            for ic_tile in range(tiling_factors["IC"]): # SA_LOOP 3, 2, num_ic_tiles
                for kh_tile in range(tiling_factors["KH"]): # SA_LOOP 4, 3, num_kh_tiles
                    for kw_tile in range(tiling_factors["KW"]): # SA_LOOP 5, 4, num_kw_tiles
                        for y_tile in range(tiling_factors["OH"]): # SA_LOOP 6, 5, num_oh_tiles
                            for x_tile in range(tiling_factors["OW"]): # SA_LOOP 7, 6, num_oh_tiles
                                # W_BASE_ADDR = oc_tile*W_TILE_OC_STRIDE
                                # W_BASE_ADDR += ic_tile*W_TILE_IC_STRIDE
                                # W_BASE_ADDR += kh_tile*W_TILE_KH_STRIDE
                                # W_BASE_ADDR += kw_tile*W_TILE_KW_STRIDE

                                # O_BASE_ADDR = n_tile*O_TILE_N_STRIDE
                                # O_BASE_ADDR += y_tile*O_TILE_OH_STRIDE
                                # O_BASE_ADDR += x_tile*O_TILE_OW_STRIDE
                                # O_BASE_ADDR += oc_tile * O_TILE_OC_STRIDE
                                #

                                # I_BASE_ADDR = n_tile*I_TILE_N_STRIDE
                                # I_BASE_ADDR += ic_tile*I_TILE_IC_STRIDE
                                # I_BASE_ADDR += kh_tile*I_TILE_KH_STRIDE
                                # I_BASE_ADDR += y_tile*I_TILE_OH_STRIDE
                                # I_BASE_ADDR += kw_tile*I_TILE_KW_STRIDE
                                # I_BASE_ADDR += x_tile*I_TILE_OW_STRIDE

                                IBUF_TILE = load_input_tile(all_tiling, CONV_STRIDE, (N, IH, IW, IC), I_BASE_ADDR, IBUF_TILE, I)
                                WBUF_TILE = load_weight_tile(all_tiling, W_BASE_ADDR, WBUF_TILE, W)
                                OBUF_TILE = load_output_tile(all_tiling, (N, OH, OW, OC), O_BASE_ADDR, OBUF_TILE, O)
                                OBUF_TILE = execute_tile(all_tiling, (IBUF_TILE, WBUF_TILE, OBUF_TILE), CONV_STRIDE, (N_ROWS, M_COLS))
                                O = store_output_tile(all_tiling, (N, OH, OW, OC), O_BASE_ADDR, OBUF_TILE, O)

                                I_BASE_ADDR += I_TILE_OW_STRIDE
                                O_BASE_ADDR += O_TILE_OW_STRIDE

                            I_BASE_ADDR -= tiling_factors["OW"] * I_TILE_OW_STRIDE
                            I_BASE_ADDR += I_TILE_OH_STRIDE

                            O_BASE_ADDR -= tiling_factors["OW"] * O_TILE_OW_STRIDE
                            O_BASE_ADDR += O_TILE_OH_STRIDE

                        I_BASE_ADDR -= tiling_factors["OH"]*I_TILE_OH_STRIDE
                        I_BASE_ADDR += I_TILE_KW_STRIDE

                        O_BASE_ADDR -= tiling_factors["OH"] * O_TILE_OH_STRIDE

                        W_BASE_ADDR += W_TILE_KW_STRIDE

                    I_BASE_ADDR -= tiling_factors["KW"]*I_TILE_KW_STRIDE
                    I_BASE_ADDR += I_TILE_KH_STRIDE

                    W_BASE_ADDR -= tiling_factors["KW"] * W_TILE_KW_STRIDE
                    W_BASE_ADDR += W_TILE_KH_STRIDE

                I_BASE_ADDR -= tiling_factors["KH"] * I_TILE_KH_STRIDE
                I_BASE_ADDR += I_TILE_IC_STRIDE

                W_BASE_ADDR -= tiling_factors["KH"] * W_TILE_KH_STRIDE
                W_BASE_ADDR += W_TILE_IC_STRIDE

            I_BASE_ADDR -= tiling_factors["IC"]*I_TILE_IC_STRIDE
            I_BASE_ADDR += I_TILE_N_STRIDE

            O_BASE_ADDR += O_TILE_N_STRIDE

            W_BASE_ADDR -= tiling_factors["IC"] * W_TILE_IC_STRIDE


        I_BASE_ADDR -= tiling_factors["N"] * I_TILE_N_STRIDE

        O_BASE_ADDR -= tiling_factors["N"] * O_TILE_N_STRIDE
        O_BASE_ADDR += O_TILE_OC_STRIDE

        W_BASE_ADDR += W_TILE_OC_STRIDE

    O = O.reshape(N, OH, OW, OC)
    O = O.transpose((0, 3, 1, 2))
    O_orig = O_orig.transpose((0, 3, 1, 2))

    return O


def _sa_conv(I, W, B, U, P):
    N = I.shape_symbols[0] # fmap batch size
    IC = I.shape_symbols[1] # num input fmap/filter channels
    OC = W.shape_symbols[0] # num of filters / # output fmap channels
    H_ = I.shape_symbols[-2] # input fmap height
    W_ = I.shape_symbols[-1] # input fmap bitwidth
    KH = W.shape_symbols[-2] # filter height
    KW = W.shape_symbols[-1] # filter bitwidth
    OH = int((H_ - KH + 2*P)/U) + 1 # output fmap height
    OW = int((W_ - KW + 2*P)/U) + 1# output fmap bitwidth
    O = np.empty((N, OC, OH, OW))
    I = np.pad(I, ((0, 0), (0, 0), (P, P), (P, P)), mode='constant')
    summation_map = defaultdict(list)
    for n in range(N):
        for x in range(OH):
            for y in range(OW):
                for oc in range(OC):
                    O[n, oc, x, y] = B[oc]
    from pprint import pprint
    for oc in range(OC):
        for n in range(N):
            for x in range(OW):
                for y in range(OH):
                    for ic in range(IC):
                        for kh in range(KH):
                            for kw in range(KW):
                                inp_idx = (n, ic, U*x + kh, U*y + kw)
                                wt_idx = (oc, ic, kh, kw)
                                out_idx = (n, oc, x, y)
                                summation_map[out_idx].append({'I': inp_idx, 'W': wt_idx})
                                O[n, oc, x, y] += I[n, ic, U*x + kh, U*y + kw] * W[oc, ic, kh, kw]

                    # Write back
    return O

def run_stuff():
    lenet_conv = ((1, 32, 16, 16), (32, 32, 6, 6), 2, 0)
    resnet_conv = ((1, 3, 224, 224), (64, 3, 7, 7), 3, 2)

    stride = lenet_conv[-2]
    pad = lenet_conv[-1]
    I, W, O, B = _preprocess(*resnet_conv)
    I_np = I.copy()
    W_np = W.copy()
    B_np = B.copy()
    # print(I.shape_symbols)
    #
    our_answer = sys_array_conv(I, W, B, O, stride, pad)
    real_answer = stanford_conv(I_np, W_np, B_np, stride, pad)
    np.testing.assert_allclose(our_answer, real_answer)


if __name__ == "__main__":
    # lenet_conv = ((1, 32, 16, 16), (32, 32, 6, 6), 2, 0)
    max_val = -1
    loop3 = (0, 8, 7)
    stride = 2
    loop5 = (0, 113, 56)
    print(np.arange(*loop3).max())
    num_iter = lambda l: (l[1] - l[0]) // l[2]
    max_lam = lambda l: (num_iter(l) - 1)*l[2] + 1
    # tval = max_lam(loop3) + max_lam((loop5[0], loop5[1], stride*loop5[2]))
    # print(max_lam(loop3))
    # print(max_lam((loop5[0], loop5[1], stride*loop5[2])))
    # for i in range(*loop5):
    #     for j in range(*loop3):
    #         idx = i*stride + j
    #         if idx > max_val:
    #             max_val = idx
    # print(f"Test max: {tval}\n"
    #       f"Actual: {max_val}")
    # lenet_conv = ((1, 1, 32, 32), (6, 1, 2, 2), 2, 0)
    # # resnet_conv = ((1, 1, 32, 32), (6, 1, 5, 5), 2, 0)
    #
    # stride = lenet_conv[-2]
    # # pad = lenet_conv[-1]
    # I, W, O, B = _preprocess(*lenet_conv)
    # I_np = I.copy()
    # W_np = W.copy()
    # B_np = B.copy()
    # pad = 0
    # # print(I.shape_symbols)
    # #
    # # our_answer = sys_array_conv(I, W, B, O, stride, pad)
    # # real_answer = test_convolution(I_np, W_np, B_np, stride, pad)
    # real_answer = stanford_conv(I_np, W_np, B_np, stride, pad)
    # print(real_answer.shape_symbols)
    # np.testing.assert_allclose(our_answer, real_answer)
    # print(np.allclose(real_answer, our_answer))

