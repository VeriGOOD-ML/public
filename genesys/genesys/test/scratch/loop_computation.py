import numpy as np

def verification_convolution(I, W, O, U, P):
    N = I.shape_symbols[0] # fmap batch size
    C = I.shape_symbols[1] # num input fmap/filter channels
    M = W.shape_symbols[0] # num of filters / # output fmap channels
    H_ = I.shape_symbols[-2] # input fmap height
    W_ = I.shape_symbols[-1] # input fmap bitwidth
    R = W.shape_symbols[-2] # filter height
    S = W.shape_symbols[-1] # filter bitwidth
    # E = int((H_ - R + 2*P)/U) + 1 # output fmap height
    # F = int((W_ - S + 2*P)/U) + 1# output fmap bitwidth
    E = O.shape_symbols[2]
    F = O.shape_symbols[3]
    # O = np.empty((N, M, E, F))
    # I = np.pad(I, ((0, 0), (0, 0), (P, P), (P, P)), mode='constant')

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

def untouched_conv_indices(I, W, O, P, S):
    all_accessed_indices = []
    N, OC, OH, OW = O
    _, IC, KH, KW = W
    computed_size = (OH - 1) * S - 2*P + KH
    test_size = (OH - 1)*S + (KH ) + 1
    print(f"Actual size: {computed_size}")
    print(f"Test size: {test_size}")
    for oc in range(OC):
        for n in range(N):
            for ic in range(IC):
                accessed_indices = []
                for kh in range(KH):
                    for kw in range(KW):
                        for oh in range(OH):
                            for ow in range(OW):
                                accessed_indices.append(oh*S + kh)
                all_accessed_indices.append(list(set(accessed_indices)))
    printed_full = False
    for ac_idx_list in all_accessed_indices:
        if len(ac_idx_list) != computed_size:
            print(f"Access index list is less than computed size: {len(ac_idx_list)}")
        if len(ac_idx_list) != test_size:
            print(f"Access index list is not same as test: {len(ac_idx_list)}\n"
                  f"")
            if not printed_full:
                print(f"{list(sorted(ac_idx_list))}")
                printed_full = True

        print()



def tiled_offsets_indices(I, W, O, P, S):
    all_accessed_indices = []
    N, OC, OH, OW = O
    _, IC, KH, KW = W
    I = np.random.randint(0, 5, I)
    W = np.random.randint(0, 5, W)
    O = np.zeros(O)
    tile_splits = {"N": 1, "OC": 2, "OH": 2, "OW": 1, "IC": 1, "KH": 1, "KW": 1}
    I_TILE_SHAPE = (N//tile_splits['N'], IC//tile_splits['IC'], OH//tile_splits['OH']*S + KH//tile_splits["KH"], OW//tile_splits['OW']*S + KW//tile_splits["KW"])
    W_TILE_SHAPE = (OC//tile_splits["OC"], IC//tile_splits["IC"], KH//tile_splits["KH"], KW//tile_splits["KW"])
    O_TILE_SHAPE = (N//tile_splits['N'], OC//tile_splits['OC'], OH//tile_splits['OH'], OW//tile_splits['OW'])

    I_TILE = np.zeros(I_TILE_SHAPE)
    W_TILE = np.zeros(W_TILE_SHAPE)
    O_TILE = np.zeros(O_TILE_SHAPE)


    computed_size = (OH - 1) * S - 2*P + KH
    test_size = (OH - 1)*S + (KH) + 1

    O_REF = verification_convolution(I, W, np.copy(O), S, P)


    # STEPS
    oc_t_step = OC//tile_splits["OC"]
    n_t_step = N//tile_splits["N"]
    ic_t_step = IC//tile_splits["IC"]
    kh_t_step = KH//tile_splits["KH"]
    kw_t_step = KW//tile_splits["KW"]
    oh_t_step = OH//tile_splits["OH"]
    ow_t_step = OW//tile_splits["OW"]
    max_outer = -1
    print((oh_t_step - 1)*S + kh_t_step)
    for oc_t in range(0, OC, OC//tile_splits["OC"]):
        for n_t in range(0, N, N//tile_splits["N"]):
            for ic_t in range(0, IC, IC//tile_splits["IC"]):
                for kh_t in range(0, KH, KH//tile_splits["KH"]):
                    for kw_t in range(0, KW, KW//tile_splits["KW"]):
                        for oh_t in range(0, OH, OH//tile_splits["OH"]):
                            for ow_t in range(0, OW, OW//tile_splits["OW"]):
                                #END FIRST LEVEL
                                if (oh_t*S + kh_t) > max_outer:
                                    max_outer = oh_t*S + kh_t
                                i_tile_idx = (slice(n_t, (n_t+1)*n_t_step), slice(ic_t, (ic_t+1)*ic_t_step),
                                              slice(oh_t*S + kh_t, (oh_t + 1) * oh_t_step * S + (kh_t + 1) * kh_t_step),
                                              slice(ow_t*S + kw_t, (ow_t + 1) * ow_t_step * S + (kw_t + 1) * kw_t_step),
                                              )
                                o_tile_idx = (slice(n_t, (n_t+1)*n_t_step), slice(oc_t, (oc_t+1)*oc_t_step),
                                              slice(oh_t, (oh_t+1)*oh_t_step), slice(ow_t, (ow_t+1)*ow_t_step))
                                w_tile_idx = (slice(oc_t, (oc_t+1)*oc_t_step), slice(ic_t, (ic_t+1)*ic_t_step),
                                              slice(kh_t, (kh_t+1)*kh_t_step), slice(kw_t, (kw_t+1)*kw_t_step))
                                I_TILE = I[i_tile_idx[0], i_tile_idx[1], i_tile_idx[2], i_tile_idx[3]]
                                W_TILE = W[w_tile_idx[0], w_tile_idx[1], w_tile_idx[2], w_tile_idx[3]]
                                O_TILE = O[o_tile_idx[0], o_tile_idx[1], o_tile_idx[2], o_tile_idx[3]]

                                # print(I_TILE.shape_symbols)
                                # print(f"OH: {oh_t}, OW: {ow_t}, KH: {kh_t}, KW: {kw_t}")
                                # print(np.arange(0, OH//tile_splits["OH"]).max() * S + np.arange(0, KH//tile_splits["KH"]).max())

                                max_idx_h = 0
                                max_idx_w = 0
                                for oc in range(OC//tile_splits["OC"]):
                                    for n in range(N//tile_splits["N"]):
                                        for ic in range(IC//tile_splits["IC"]):
                                            for kh in range(KH//tile_splits["KH"]):
                                                for kw in range(KW//tile_splits["KW"]):
                                                    for oh in range(OH//tile_splits["OH"]):
                                                        for ow in range(OW//tile_splits["OW"]):
                                                            if (oh*S + kh) > max_idx_h:
                                                                max_idx_h = (oh*S + kh)
                                                            if (ow * S + kw) > max_idx_w:
                                                                max_idx_w = (ow * S + kw)
                                                            O_TILE[n, oc, oh, ow] += I_TILE[n, ic, oh*S + kh, ow*S + kw] * W_TILE[oc, ic, kh, kw]
                                print(f"Max: {max_idx_h}\t{max_idx_w}\n")
    # print(f"Max outer: {max_outer}")

    np.testing.assert_allclose(O_REF, O)



if __name__ == "__main__":
    I = (1, 1, 224, 224)
    W = (64, 1, 7, 7)
    O = (1, 64, 112, 112)
    P = 3
    S = 2

    # I = (1, 1, 32, 32)
    # W = (6, 1, 2, 2)
    # O = (1, 6, 16, 16)
    # P = 0
    # S = 2


    I_P = (I[0], I[1], I[2] + 2*P, I[3] + 2*P)
    tiled_offsets_indices(I_P, W, O, P, S)
    # untouched_conv_indices(I_P, W, O, P, S)
