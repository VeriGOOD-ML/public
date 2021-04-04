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
    I = np.random.randint(128, size=I_shape)
    W = np.random.randint(128, size=W_shape)
    B = np.random.randint(128, size=W_shape[0])
    N = I.shape[0] # fmap batch size
    C = I.shape[1] # num input fmap/filter channels
    M = W.shape[0] # num of filters / # output fmap channels
    H_ = I.shape[-2] # input fmap height
    W_ = I.shape[-1] # input fmap bitwidth
    R = W.shape[-2] # filter height
    S = W.shape[-1] # filter bitwidth
    E = int((H_ - R + 2*P)/U) + 1 # output fmap height
    F = int((W_ - S + 2*P)/U) + 1# output fmap bitwidth
    O = np.empty((N, M, E, F))
    I = np.pad(I, ((0, 0), (0, 0), (P, P), (P, P)), mode='constant')
    return I, W, O, B

def verification_convolution(I, W, B, U, P):
    N = I.shape[0] # fmap batch size
    C = I.shape[1] # num input fmap/filter channels
    M = W.shape[0] # num of filters / # output fmap channels
    H_ = I.shape[-2] # input fmap height
    W_ = I.shape[-1] # input fmap bitwidth
    R = W.shape[-2] # filter height
    S = W.shape[-1] # filter bitwidth
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
                    O[n, m, x, y] += B[m]
                    # Write back
    return O


def systolic_conv(I, W, B, O, U, P):
    # ALL 2: [kw, kh, ox, oy, oc, ic, on]
    #ALL = (iw: 28, ih: 28, ic: 48, oc:256, kw: 5, kh: 5, b: 16)
    # I = (b: 16, ic: 48, ih: 28, iw: 28)
    # I = (b: (1,2,8) , ic: (3,1,1), ih: 28, iw: 28)
    # W = (oc: 256, ic: 48, kh: 5, kw: 28)
    # O = (b: 16, oc: 256, oh: 28, ow: 28)
    CONV_STRIDE = U
    # Loop blocking: [kw: (5, 1, 1), kh: (5, 1, 1), ow: (1, 7, 4), oh: (1, 14, 2), oc: (2, 1, 8), ic: (3, 1, 1), b: (1, 2, 8)]
    # Loop partitioning: [(1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (16, 1, 1), (16, 1, 1), (1, 1, 1)]
    # Loop orders [(0, 6, 6), (1, 6, 6), (6, 0, 0), (6, 1, 1), (2, 6, 3), (3, 6, 6), (6, 2, 2)]
    N = I.shape[0] # fmap batch size
    IC = I.shape[1] # num input fmap/filter channels
    IH = I.shape[2] # input height
    IW = I.shape[3] # input width
    OC = W.shape[0] # num of filters / # output fmap channels
    KH = W.shape[-2] # filter height
    KW = W.shape[-1] # filter bitwidth
    OH = O.shape[-2]
    OW = O.shape[-1]

    OC_TILE_SIZE = 1
    N_TILE_SIZE = 1
    IC_TILE_SIZE = 1
    KH_TILE_SIZE = 1
    KW_TILE_SIZE = 1
    OH_TILE_SIZE = 1
    OW_TILE_SIZE = 1

    IW_TILE_SIZE = (OW_TILE_SIZE - 1)*CONV_STRIDE - 2 * P + KW_TILE_SIZE
    IH_TILE_SIZE = (OH_TILE_SIZE - 1)*CONV_STRIDE - 2 * P + KH_TILE_SIZE

    BASE_ADDR_GEN_LOOP_ITER_LOOKUP_TABLE = {}
    BASE_ADDR_GEN_LOOP_STRIDE_LOOKUP_TABLE = {}

    LD_WBUF_ADDR_GEN_LOOP_ITER_LOOKUP_TABLE = {}


    LD_WBUF_ADDR_GEN_LOOP_STRIDE_LOOKUP_TABLE = {}

    LD_IBUF_ADDR_GEN_LOOP_ITER_LOOKUP_TABLE = {}
    LD_IBUF_ADDR_GEN_LOOP_STRIDE_LOOKUP_TABLE = {}

    ST_OBUF_ADDR_GEN_LOOP_ITER_LOOKUP_TABLE = {}
    ST_OBUF_ADDR_GEN_LOOP_STRIDE_LOOKUP_TABLE = {}

    LD_BBUF_ADDR_GEN_LOOP_ITER_LOOKUP_TABLE = {}
    LD_BBUF_ADDR_GEN_LOOP_STRIDE_LOOKUP_TABLE = {}

    WEIGHT_DTYPE_SIZE = 2
    INPUT_DTYPE_SIZE = 2
    OUTPUT_DTYPE_SIZE = 2
    BIAS_DTYPE_SIZE = 2
    all_tiling = {"N": N, "IC": IC, "OH": OH//2, "OW": OW//2, "KH": KH, "KW": KW}
    output_tiling = (all_tiling["N"], all_tiling["OC"], all_tiling["OH"], all_tiling["OW"])
    weight_tiling = (all_tiling["OC"], all_tiling["IC"], all_tiling["KH"], all_tiling["KW"])
    all_tiling["IH"] = (all_tiling["OH"] - 1) * U - 2*P + all_tiling["KH"]
    all_tiling["IW"] = (all_tiling["OW"] - 1) * U - 2*P + all_tiling["KW"]
    input_tiling = (all_tiling["N"], all_tiling["IC"], all_tiling["IH"], all_tiling["IW"])


    INPUT_TILE_SIZE = int(np.prod(input_tiling))
    WEIGHT_TILE_SIZE = int(np.prod(weight_tiling))
    BIAS_TILE_SIZE = OC
    NUM_INPUT_TILES = int(np.ceil(np.prod(I.shape)/INPUT_TILE_SIZE))
    NUM_WEIGHT_TILES = int(np.ceil(np.prod(W.shape) / WEIGHT_TILE_SIZE))

    NUM_OUTPUT_TILES = NUM_INPUT_TILES * NUM_WEIGHT_TILES
    OUTPUT_TILE_SIZE = int(np.ceil(int(np.prod(O.shape))/NUM_OUTPUT_TILES))

    # SET_BASE_ADDR LOW/HIGH, NS, WBUF, 0
    BASE_ADDRESS_WEIGHT = 0

    # SET_BASE_ADDR LOW/HIGH, NS, IBUF, 0
    BASE_ADDRESS_INPUT = np.prod(W.shape)*WEIGHT_DTYPE_SIZE

    # SET_BASE_ADDR LOW/HIGH, NS, OBUF, 0
    BASE_ADDRESS_OUTPUT = BASE_ADDRESS_INPUT + np.prod(I.shape)*INPUT_DTYPE_SIZE

    # SET_BASE_ADDR LOW/HIGH, NS, OBUF, 0
    BASE_ADDRESS_BIAS = BASE_ADDRESS_OUTPUT + np.prod(O.shape)*OUTPUT_DTYPE_SIZE

    # SA_LOOP 1, 0, NUM_WEIGHT_TILES
    # SET_STRIDE LOW/HIGH, LD, WBUF, 0, WEIGHT_TILE_SIZE
    BASE_ADDR_GEN_LOOP_ITER_LOOKUP_TABLE[0] = {"level": 1, "iterations": NUM_WEIGHT_TILES, "current": 0}
    BASE_ADDR_GEN_LOOP_STRIDE_LOOKUP_TABLE[0] = {"current_offset": BASE_ADDRESS_WEIGHT, "stride": WEIGHT_TILE_SIZE}

    # SA_LOOP 2, 1, NUM_INPUT_TILES
    # SET_STRIDE LOW/HIGH, LD, IBUF, 0, INPUT_TILE_SIZE
    BASE_ADDR_GEN_LOOP_ITER_LOOKUP_TABLE[1] = {"level": 2, "iterations": NUM_INPUT_TILES, "current": 0}
    BASE_ADDR_GEN_LOOP_STRIDE_LOOKUP_TABLE[1] = {"current_offset": BASE_ADDRESS_INPUT, "stride": INPUT_TILE_SIZE}

    # SA_LOOP 3, 2, NUM_OUTPUT_TILES
    # SET_STRIDE LOW/HIGH, ST, OBUF, 0, OUTPUT_TILE_SIZE
    BASE_ADDR_GEN_LOOP_ITER_LOOKUP_TABLE[2] = {"level": 3, "iterations": NUM_OUTPUT_TILES, "current": 0}
    BASE_ADDR_GEN_LOOP_STRIDE_LOOKUP_TABLE[2] = {"current_offset": BASE_ADDRESS_OUTPUT, "stride": OUTPUT_TILE_SIZE}

    M, N = 32, 32
    OFFCHIP_WORD_SIZE = 32
    LD_REQUEST_SIZE = 1

    ## LOAD WBUF TILE

    LD_WBUF_ADDR_GEN_LOOP_STRIDE_LOOKUP_TABLE[4] = {}
    LD_WBUF_ADDR_GEN_LOOP_STRIDE_LOOKUP_TABLE[4]["current_offset"] = 0
    weight_tile_fetch_size = int(np.ceil((LD_REQUEST_SIZE*N*M*WEIGHT_DTYPE_SIZE)/OFFCHIP_WORD_SIZE))
    LD_WBUF_ADDR_GEN_LOOP_STRIDE_LOOKUP_TABLE[4]["stride"] = weight_tile_fetch_size

    LD_WBUF_ADDR_GEN_LOOP_ITER_LOOKUP_TABLE[4] = {}
    LD_WBUF_ADDR_GEN_LOOP_ITER_LOOKUP_TABLE[4]["level"] = 4
    ld_wbuf_iters = int(np.ceil(WEIGHT_TILE_SIZE/(weight_tile_fetch_size*OFFCHIP_WORD_SIZE)))
    LD_WBUF_ADDR_GEN_LOOP_ITER_LOOKUP_TABLE[4]["iterations"] = ld_wbuf_iters
    LD_WBUF_ADDR_GEN_LOOP_ITER_LOOKUP_TABLE[4]["current"] = 0


    ## LOAD IBUF TILE

    LD_IBUF_ADDR_GEN_LOOP_STRIDE_LOOKUP_TABLE[5] = {}
    LD_IBUF_ADDR_GEN_LOOP_STRIDE_LOOKUP_TABLE[5]["current_offset"] = 0
    input_tile_fetch_size = int(np.ceil((LD_REQUEST_SIZE*N*M*INPUT_DTYPE_SIZE)/OFFCHIP_WORD_SIZE))
    LD_IBUF_ADDR_GEN_LOOP_STRIDE_LOOKUP_TABLE[5]["stride"] = input_tile_fetch_size

    LD_IBUF_ADDR_GEN_LOOP_ITER_LOOKUP_TABLE[5] = {}
    LD_IBUF_ADDR_GEN_LOOP_ITER_LOOKUP_TABLE[5]["level"] = 4
    ld_ibuf_iters = int(np.ceil(INPUT_TILE_SIZE/(input_tile_fetch_size*OFFCHIP_WORD_SIZE)))
    LD_IBUF_ADDR_GEN_LOOP_ITER_LOOKUP_TABLE[5]["iterations"] = ld_ibuf_iters
    LD_IBUF_ADDR_GEN_LOOP_ITER_LOOKUP_TABLE[5]["current"] = 0


    ## STORE OBUF TILE
    ST_OBUF_ADDR_GEN_LOOP_STRIDE_LOOKUP_TABLE[6] = {}
    ST_OBUF_ADDR_GEN_LOOP_STRIDE_LOOKUP_TABLE[6]["current_offset"] = BASE_ADDRESS_INPUT
    output_tile_fetch_size = int(np.ceil((LD_REQUEST_SIZE*N*M*OUTPUT_DTYPE_SIZE)/OFFCHIP_WORD_SIZE))
    ST_OBUF_ADDR_GEN_LOOP_STRIDE_LOOKUP_TABLE[6]["stride"] = output_tile_fetch_size

    ST_OBUF_ADDR_GEN_LOOP_ITER_LOOKUP_TABLE[6] = {}
    ST_OBUF_ADDR_GEN_LOOP_ITER_LOOKUP_TABLE[6]["level"] = 4
    st_obuf_iters = int(np.ceil(OUTPUT_TILE_SIZE/(output_tile_fetch_size*OFFCHIP_WORD_SIZE)))
    ST_OBUF_ADDR_GEN_LOOP_ITER_LOOKUP_TABLE[6]["iterations"] = st_obuf_iters
    ST_OBUF_ADDR_GEN_LOOP_ITER_LOOKUP_TABLE[6]["current"] = 0

    ## LOAD BBUF TILE
    LD_BBUF_ADDR_GEN_LOOP_STRIDE_LOOKUP_TABLE[7] = {}
    LD_BBUF_ADDR_GEN_LOOP_STRIDE_LOOKUP_TABLE[7]["current_offset"] = BASE_ADDRESS_INPUT
    bias_tile_fetch_size = int(np.ceil((LD_REQUEST_SIZE*N*M*BIAS_DTYPE_SIZE)/OFFCHIP_WORD_SIZE))
    LD_BBUF_ADDR_GEN_LOOP_STRIDE_LOOKUP_TABLE[7]["stride"] = bias_tile_fetch_size

    LD_BBUF_ADDR_GEN_LOOP_ITER_LOOKUP_TABLE[7] = {}
    LD_BBUF_ADDR_GEN_LOOP_ITER_LOOKUP_TABLE[7]["level"] = 4
    ld_bbuf_iters = int(np.ceil(BIAS_TILE_SIZE/(bias_tile_fetch_size*OFFCHIP_WORD_SIZE)))
    LD_BBUF_ADDR_GEN_LOOP_ITER_LOOKUP_TABLE[7]["iterations"] = ld_bbuf_iters
    LD_BBUF_ADDR_GEN_LOOP_ITER_LOOKUP_TABLE[7]["current"] = 0

    ## RD/WT LOOKUP tables




    for weight_tile in range(NUM_WEIGHT_TILES): # SA-LOOP: LOOP-ID=0, LOOP-LEVEL=1

        tile_weight_stride = BASE_ADDR_GEN_LOOP_STRIDE_LOOKUP_TABLE[0]['stride']
        tile_weight_current_offset = BASE_ADDR_GEN_LOOP_STRIDE_LOOKUP_TABLE[0]['current_offset']
        tile_weight_base_addr = tile_weight_stride + tile_weight_current_offset
        BASE_ADDR_GEN_LOOP_STRIDE_LOOKUP_TABLE[0]['current_offset'] = tile_weight_base_addr
        for input_tile in range(NUM_INPUT_TILES): # SA-LOOP: LOOP-ID=1, LOOP-LEVEL=2

            tile_input_stride = BASE_ADDR_GEN_LOOP_STRIDE_LOOKUP_TABLE[1]['stride']
            tile_input_current_offset = BASE_ADDR_GEN_LOOP_STRIDE_LOOKUP_TABLE[1]['current_offset']
            tile_input_base_addr = tile_input_stride + tile_input_current_offset
            BASE_ADDR_GEN_LOOP_STRIDE_LOOKUP_TABLE[1]['current_offset'] = tile_input_base_addr

            for output_tile in range(NUM_OUTPUT_TILES): # SA-LOOP: LOOP-ID=2, LOOP-LEVEL=3
                tile_output_stride = BASE_ADDR_GEN_LOOP_STRIDE_LOOKUP_TABLE[2]['stride']
                tile_output_current_offset = BASE_ADDR_GEN_LOOP_STRIDE_LOOKUP_TABLE[2]['current_offset']
                tile_output_base_addr = tile_output_stride + tile_output_current_offset
                BASE_ADDR_GEN_LOOP_STRIDE_LOOKUP_TABLE[2]['current_offset'] = tile_output_base_addr

                weight_chunk_iters = LD_WBUF_ADDR_GEN_LOOP_ITER_LOOKUP_TABLE[4]["iterations"]
                input_chunk_iters = LD_IBUF_ADDR_GEN_LOOP_ITER_LOOKUP_TABLE[4]["iterations"]
                output_chunk_iters = ST_OBUF_ADDR_GEN_LOOP_ITER_LOOKUP_TABLE[4]["iterations"]
                LD_WBUF_ADDR_GEN_LOOP_STRIDE_LOOKUP_TABLE[4]["current_offset"] = tile_weight_base_addr
                LD_IBUF_ADDR_GEN_LOOP_STRIDE_LOOKUP_TABLE[5]["current_offset"] = tile_input_base_addr
                ST_OBUF_ADDR_GEN_LOOP_STRIDE_LOOKUP_TABLE[6]["current_offset"] = tile_output_base_addr

                for weight_tile_chunk in range(weight_chunk_iters): #SA_LOOP LOOP-LEVEL=4, LOOP-ID=4, ITERS=NUM_WEIGHT_CHUNKS
                    wt_prev_offset = LD_WBUF_ADDR_GEN_LOOP_STRIDE_LOOKUP_TABLE[4]["current_offset"]
                    wt_stride = LD_WBUF_ADDR_GEN_LOOP_STRIDE_LOOKUP_TABLE[4]["stride"]
                    wt_chunk_new_offset = wt_prev_offset + weight_tile_chunk*wt_stride
                    ## LD NS, WBUF, LOOP-ID=4, 1 --> generated address for fetching
                    LD_WBUF_ADDR_GEN_LOOP_STRIDE_LOOKUP_TABLE[4]["current_offset"] = wt_chunk_new_offset

                for input_tile_chunk in range(input_chunk_iters):#SA_LOOP LOOP-LEVEL=4, LOOP-ID=5, ITERS=NUM_INPUT_CHUNKS
                    input_prev_offset = LD_IBUF_ADDR_GEN_LOOP_STRIDE_LOOKUP_TABLE[5]["current_offset"]
                    input_stride = LD_IBUF_ADDR_GEN_LOOP_STRIDE_LOOKUP_TABLE[5]["stride"]
                    input_chunk_new_offset = input_prev_offset + input_tile_chunk * input_stride
                    ## LD NS, IBUF, LOOP-ID=5, 1 --> generated address for fetching
                    LD_IBUF_ADDR_GEN_LOOP_STRIDE_LOOKUP_TABLE[5]["current_offset"] = input_chunk_new_offset




                oc_stride_wbuf = KH_TILE_SIZE*KW_TILE_SIZE*IC_TILE_SIZE/N
                ic_stride_wbuf = KH_TILE_SIZE*KW_TILE_SIZE
                kh_stride_wbuf = KW_TILE_SIZE
                kw_stride_wbuf = 1

                n_stride_obuf = OW_TILE_SIZE*OH_TILE_SIZE*OC_TILE_SIZE/M
                oc_stride_obuf = OW_TILE_SIZE*OH_TILE_SIZE
                oh_stride_obuf = OW_TILE_SIZE
                ow_stride_obuf = 1

                n_stride_ibuf = IW_TILE_SIZE*IH_TILE_SIZE*IC_TILE_SIZE/N
                ic_stride_ibuf = IW_TILE_SIZE*IH_TILE_SIZE
                kh_stride_ibuf = IW_TILE_SIZE
                kw_stride_ibuf = 1
                oh_stride_ibuf = CONV_STRIDE*IW_TILE_SIZE
                ow_stride_ibuf = CONV_STRIDE

                ibuf_addr_offset = 0
                obuf_addr_offset = 0
                wbuf_addr_offset = 0

                for oc in range(OC_TILE_SIZE // M):
                    wbuf_addr_offset += oc*ic_stride_wbuf
                    obuf_addr_offset += oc*oc_stride_obuf
                    for n in range(N_TILE_SIZE):
                        ibuf_addr_offset += n * n_stride_ibuf
                        obuf_addr_offset += n * n_stride_obuf
                        for ic in range(IC_TILE_SIZE // N):
                            ibuf_addr_offset += ic * ic_stride_ibuf
                            wbuf_addr_offset += ic * ic_stride_wbuf
                            for kh in range(KH_TILE_SIZE): # Replace loops with IB
                                wbuf_addr_offset += kh * kh_stride_wbuf
                                ibuf_addr_offset += kh * kh_stride_ibuf
                                for kw in range(KW_TILE_SIZE):
                                    wbuf_addr_offset += kw * kw_stride_wbuf
                                    ibuf_addr_offset += kw * kw_stride_ibuf
                                    for y in range(OH_TILE_SIZE):
                                        obuf_addr_offset += y * oh_stride_obuf
                                        ibuf_addr_offset += y * oh_stride_ibuf
                                        for x in range(OW_TILE_SIZE):
                                            obuf_addr_offset += x * ow_stride_obuf
                                            ibuf_addr_offset += x * ow_stride_ibuf
                                            O[obuf_addr_offset] += I[ibuf_addr_offset]*W[wbuf_addr_offset]



    return O

def sys_array_conv(I, W, B, O, stride, pad):
    N_ROWS = 32
    M_COLS = 32
    I = I.tranpose((0, 2, 3, 1))
    O = O.tranpose((0, 2, 3, 1))
    N = I.shape[0] # fmap batch size
    IC = I.shape[3] # num input fmap/filter channels
    IH = I.shape[1] # input height
    IW = I.shape[2] # input width
    OC = W.shape[0] # num of filters / # output fmap channels
    KH = W.shape[-2] # filter height
    KW = W.shape[-1] # filter bitwidth
    OH = O.shape[-3]
    OW = O.shape[-2]

    all_tiling = {"N": N, "IC": IC, "OH": int(np.ceil(OH/2)), "OW": int(np.ceil(OW/2)), "KH": KH, "KW": KW, "OC": OC}
    tiling_factors = {"N": 1, "IC": 1, "OH": 2, "OW": 2, "KH": 1, "KW": 1, "OC": 1}
    output_tiling = (all_tiling["N"], all_tiling["OC"], all_tiling["OH"], all_tiling["OW"])
    weight_tiling = (all_tiling["OC"], all_tiling["IC"], all_tiling["KH"], all_tiling["KW"])
    all_tiling["IH"] = (all_tiling["OH"] - 1) * stride - 2*pad + all_tiling["KH"]
    all_tiling["IW"] = (all_tiling["OW"] - 1) * stride - 2*pad + all_tiling["KW"]
    input_tiling = (all_tiling["N"], all_tiling["IC"], all_tiling["IH"], all_tiling["IW"])

    NUM_WEIGHT_TILES = int(np.ceil(np.prod(W.shape) / np.prod(weight_tiling)))
    NUM_OUTPUT_TILES = int(np.ceil(np.prod(O.shape) / np.prod(output_tiling)))
    NUM_INPUT_TILES = int(np.ceil(np.prod(I.shape) / np.prod(weight_tiling)))
    otpt_tile_size = all_tiling["N"] * all_tiling["OH"] * all_tiling["OW"] * all_tiling["OC"]

    wt_tile_size = all_tiling["IC"] * all_tiling["OC"] * all_tiling["KH"] * all_tiling["KW"]
    for n in range(N):
        for x in range(OH):
            for y in range(OW):
                for oc in range(OC):
                    O[n, oc, x, y] = B[oc]
    O = O.reshape(-1)
    I = I.reshape(-1)
    W = W.reshape(-1)

    for oc_tile in range(tiling_factors["OC"]): # SA_LOOP 1, 0, num_oc_tiles # OC
        # GENADDR LOW, X, LD, WBUF, 0, wt_oc_stride
        # GENADDR LOW, X, LD, OBUF, 0, output_oc_stride
        # GENADDR LOW, X, ST, OBUF, 0, output_oc_stride
        oc_range_high = (oc_tile + 1) * all_tiling["OC"]
        oc_range_low = (oc_tile) * all_tiling["OC"]
        for n_tile in range(tiling_factors["N"]): # SA_LOOP 2, 1, num_n_tiles # N
            # GENADDR LOW, X, LD, IBUF, 1, input_n_stride
            # GENADDR LOW, X, LD, OBUF, 1, output_n_stride
            # GENADDR LOW, X, ST, OBUF, 1, output_n_stride
            n_range_high = (n_tile + 1) * all_tiling["N"]
            n_range_low = (n_tile) * all_tiling["N"]
            for ic_tile in range(tiling_factors["IC"]): # SA_LOOP 3, 2, num_ic_tiles # IC

                # GENADDR LOW, X, LD, WBUF, 2, wt_ic_stride
                # GENADDR LOW, X, LD, IBUF, 2, input_ic_stride [N, H, W, C]
                ic_range_high = (ic_tile + 1) * all_tiling["IC"]
                ic_range_low = (ic_tile) * all_tiling["IC"]
                for kh_tile in range(tiling_factors["KH"]): # SA_LOOP 4, 3, num_kh_tiles # KH

                    # GENADDR LOW, X, LD, WBUF, 3, wt_kh_stride
                    # GENADDR LOW, X, LD, IBUF, 3, wt_kh_stride

                    kh_range_high = (kh_tile + 1) * all_tiling["KH"]
                    kh_range_low = (kh_tile) * all_tiling["KH"]
                    for kw_tile in range(tiling_factors["KW"]): # SA_LOOP 5, 4, num_kw_tiles # KW

                        # GENADDR LOW, X, LD, WBUF, 4, wt_kw_stride
                        # GENADDR LOW, X, LD, IBUF, 4, wt_kw_stride
                        kw_range_high = (kw_tile + 1) * all_tiling["KW"]
                        kw_range_low = (kw_tile) * all_tiling["KW"]

                        # OPTION 2
                        # SA_LOOP 4, 21, WEIGHT_TILE_SIZE/REQUEST_SIZE
                        # GENADDR LOW, X, LD, WBUF, 21, REQUEST_SIZE
                        # LD NS, X, WBUF, 21, REQUEST_SIZE

                        for y_tile in range(tiling_factors["OH"]): # SA_LOOP 6, 5, num_oh_tiles # OH

                            # GENADDR LOW, X, LD, IBUF, 5, output_oh_stride + (stride*dtype)
                            # GENADDR LOW, X, LD, OBUF, 5, output_oh_stride
                            # GENADDR LOW, X, ST, OBUF, 5, output_oh_stride
                            y_range_high = (y_tile + 1) * all_tiling["OH"]
                            y_range_low = (y_tile) * all_tiling["OH"]
                            for x_tile in range(tiling_factors["OW"]): # SA_LOOP 7, 6, num_ow_tiles # OW

                                # GENADDR LOW, X, LD, IBUF, 6, output_ow_stride + (stride*dtype)
                                # GENADDR LOW, X, LD, OBUF, 6, output_ow_stride
                                # GENADDR LOW, X, ST, OBUF, 6, output_ow_stride
                                x_range_high = (x_tile+1)*all_tiling["OW"]
                                x_range_low = (x_tile)*all_tiling["OW"]
                                ih_range_low = stride*y_range_low + kh_range_low
                                ih_range_high = stride*y_range_high + kh_range_high - 1
                                iw_range_low = stride*x_range_low + kw_range_low
                                iw_range_high = stride*x_range_high + kw_range_high - 1


                                for n in range(all_tiling["N"]): # SA_LOOP 8, 7, N_TILE_SIZE # N
                                    # GENADDR LOW, X, LD, IBUF, 7, ih*iw*ic
                                    for kh in range(all_tiling["KH"]): # SA_LOOP 9, 8, KH_TILE_SIZE # KH
                                        # GENADDR LOW, X, LD, IBUF, 8, ic*iw
                                        for kw in range(all_tiling["KW"]): # SA_LOOP 10, 9, KW_TILE_SIZE # KW
                                            # GENADDR LOW, X, LD, IBUF, 9, ic
                                            for y in range(all_tiling["OH"]): # SA_LOOP 11, 10, OH_TILE_SIZE # OH
                                                # GENADDR LOW, X, LD, IBUF, 10, conv_stride_y*ic*iw
                                                for x in range(all_tiling["OW"]): # SA_LOOP 12, 11, OH_TILE_SIZE # OH
                                                    # GENADDR LOW, X, LD, IBUF, 11, conv_stride_x*ic
                                                    for ic in range(int(np.ceil(all_tiling["IC"]/N_ROWS))): # SA_LOOP 13, 12, IC_TILE_SIZE/REQ_SIZE #
                                                        # GENADDR LOW, X, LD, IBUF, 12, N
                                                        # LD NS, X, IBUF, 12, 1
                                                        pass

                                for w in range(int(np.ceil(wt_tile_size/(N_ROWS*M_COLS)))): # SA_LOOP 8, 13, int(np.ceil(wt_tile_size/(N_ROWS*M_COLS)))
                                    pass
                                    # GENADDR LOW, X, LD, WBUF, 13, (N_ROWS*M_COLS)
                                    # LD NS, X, WBUF, 13, 1

                                for o in range(int(np.ceil(otpt_tile_size/(M_COLS)))): # SA_LOOP 8, 14, int(np.ceil(otpt_tile_size/(M_COLS)))
                                    pass
                                    # GENADDR LOW, X, LD, OBUF, 14, (M_COLS)
                                    # LD NS, X, OBUF, 14, 1

                                W_tile = W[oc_range_low:oc_range_high, ic_range_low:ic_range_high,
                                         kh_range_low:kh_range_high,
                                         kw_range_low:kw_range_high]
                                I_tile = I[n_range_low:n_range_high, ic_range_low:ic_range_high, ih_range_low:ih_range_high, iw_range_low:iw_range_high]

                                w_index_low = (oc_range_low, ic_range_low, kh_range_low, kw_range_low)
                                w_index_high = (oc_range_high - 1, ic_range_high - 1, kh_range_high - 1, kw_range_high - 1)


                                i_index_low = (n_range_low, ic_range_low, ih_range_low, iw_range_low)
                                i_index_high = (n_range_high - 1, ic_range_high - 1, ih_range_high - 1, iw_range_high - 1)
                                w_idx_high_flattened = np.ravel_multi_index(w_index_high, W.shape)
                                w_idx_low_flattened = np.ravel_multi_index(w_index_low, W.shape)
                                i_idx_high_flattened = np.ravel_multi_index(i_index_high, I.shape)
                                i_idx_low_flattened = np.ravel_multi_index(i_index_low, I.shape)

                                O_tile = O[n_range_low:n_range_high, oc_range_low:oc_range_high, y_range_low:y_range_high, x_range_low:x_range_high]
                                # W: [OC, IC, KH, KW]
                                # I: [N, IH, IW, IC]
                                # O: [N, OH, OW, OC]
                                # W_BASE_ADDR = 0
                                for oc in range(all_tiling["OC"]): # SA_LOOP 8, 15, OC_TILE_SIZE # OC/M_COLS
                                    # W_BASE_ADDR = oc*IC_TILE_SIZE*KH_TILE_SIZE*KW_TILE_SIZE/N_ROWS
                                    # GENADDR LOW, X, RD, WBUF, 15, IC_TILE_SIZE*KH_TILE_SIZE*KW_TILE_SIZE/N_ROWS
                                    # GENADDR LOW, X, RD, OBUF, 15, 1

                                    for n in range(all_tiling["N"]): # SA_LOOP 9, 16, N_TILE_SIZE # N
                                        # GENADDR LOW, X, RD, IBUF, 16, IH_TILE_SIZE*IW_TILE_SIZE*IC_TILE_SIZE/N_ROWS
                                        # GENADDR LOW, X, RD, OBUF, 16, OH_TILE_SIZE*OW_TILE_SIZE*OC_TILE_SIZE/M_COLS
                                        for ic in range(all_tiling["IC"]): # SA_LOOP 10, 17, IC_TILE_SIZE/N_ROWS # IC
                                            # W_BASE_ADDR += ic*KH_TILE_SIZE*KW_TILE_SIZE
                                            # GENADDR LOW, X, RD, WBUF, 17, KH_TILE_SIZE*KW_TILE_SIZE
                                            # GENADDR LOW, X, RD, IBUF, 17, 1
                                            for kh in range(all_tiling["KH"]): # SA_LOOP 11, 18, KH_TILE_SIZE # KH
                                                # W_BASE_ADDR += kh*KW_TILE_SIZE
                                                # GENADDR LOW, X, RD, WBUF, 18, KW_TILE_SIZE
                                                # GENADDR LOW, X, RD, IBUF, 18, IC_TILE_SIZE*IW_TILE_SIZE/N_ROWS
                                                for kw in range(all_tiling["KW"]): # SA_LOOP 12, 19, KW_TILE_SIZE # KW
                                                    # W_BASE_ADDR += kw
                                                    # GENADDR LOW, X, RD, WBUF, 19, 1
                                                    # GENADDR LOW, X, RD, IBUF, 19, IC_TILE_SIZE/N_ROWS
                                                    for y in range(all_tiling["OH"]): # SA_LOOP 13, 20, OH_TILE_SIZE # OH
                                                        # GENADDR LOW, X, RD, IBUF, 13, CONV_STRIDE*IC_TILE_SIZE*IW_TILE_SIZE/N_ROWS
                                                        # GENADDR LOW, X, RD, OBUF, 13, OW_TILE_SIZE*OC_TILE_SIZE/M_COLS
                                                        for x in range(all_tiling["OW"]): # SA_LOOP 14, 21, OW_TILE_SIZE # OW
                                                            # GENADDR LOW, X, RD, IBUF, 14, CONV_STRIDE*IC_TILE_SIZE/N_ROWS
                                                            # GENADDR LOW, X, RD, OBUF, 14, OC_TILE_SIZE/M_COLS
                                                            O_tile[n, oc, x, y] += I_tile[n, ic, stride*x + kh, stride*y + kw] * W_tile[oc, ic, kh, kw]
                                                            # GENADDR LOW, X, WR, OBUF, 14, OC_TILE_SIZE/M_COLS
                                                        # GENADDR LOW, X, WR, OBUF, 13, OW_TILE_SIZE*OC_TILE_SIZE/M_COLS
                                    # GENADDR LOW, X, WR, OBUF, 9, OH_TILE_SIZE*OW_TILE_SIZE*OC_TILE_SIZE/M_COLS
                                # GENADDR LOW, X, WR, OBUF, 8, IC_TILE_SIZE*KH_TILE_SIZE*KW_TILE_SIZE/N_ROWS
    O = O.reshape(N, OH, OW, OC)
    O = O.tranpose((0, 3, 1, 2))
    return O


def test_convolution(I, W, B, U, P):
    N = I.shape[0] # fmap batch size
    IC = I.shape[1] # num input fmap/filter channels
    OC = W.shape[0] # num of filters / # output fmap channels
    H_ = I.shape[-2] # input fmap height
    W_ = I.shape[-1] # input fmap bitwidth
    KH = W.shape[-2] # filter height
    KW = W.shape[-1] # filter bitwidth
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

if __name__ == "__main__":
    lenet_conv = ((1, 1, 32, 32), (6, 1, 5, 5), 1, 0)

    stride = lenet_conv[-2]
    pad = lenet_conv[-1]
    I, W, O, B = _preprocess(*lenet_conv)
    I_np = I.copy()
    W_np = W.copy()
    B_np = B.copy()
    print(I.shape)
    #
    our_answer = sys_array_conv(I, W, B, O, stride, pad)
    # real_answer = test_convolution(I_np, W_np, B_np, stride, pad)
    real_answer = stanford_conv(I_np, W_np, B_np, stride, pad)
    np.testing.assert_allclose(our_answer, real_answer)
    print(np.allclose(real_answer, our_answer))

