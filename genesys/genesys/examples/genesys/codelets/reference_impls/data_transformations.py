import numpy as np
# from examples.genesys import GENESYS_CFG

# Shuffles weights within a tile for correct mapping to systolic array
def shuffle_weights(w_orig, arch_config, layer_type="conv"):

    # Layout of weights is in (KH, KW, IC, OC) format
    weights = w_orig.copy()

    w_dim = weights.shape
    result = np.zeros(w_dim, dtype=weights.dtype)
    tile_m = arch_config['ARRAY_M']
    tile_n = arch_config['ARRAY_N']
    coord_map = {}

    if "conv" in layer_type:
        ic_coords = []
        oc_coords = []
        all_ics = []
        for kh in range(w_dim[0]):
            for kw in range(w_dim[1]):
                for ic in range(0, w_dim[2], tile_n):  # IC
                    for oc in range(0, w_dim[3], tile_m): # OC
                        for n in range(tile_n):  # Rows
                            for m in range(tile_m): # Columns
                                # Reverse order within a tile because systolic array is filled from last column first.

                                src_coord = kh, kw, ic + n, oc + m
                                dst_coord = kh, kw, ic + n, oc + tile_m - m - 1
                                # dst_coord = kh, kw, ic + tile_n - n - 1, oc + tile_m - m - 1

                                assert src_coord not in coord_map
                                coord_map[src_coord] = dst_coord
                                assert src_coord[-1] < result.shape[-1], f"Invalid coordinate for source: {src_coord[-1]}\n" \
                                                                         f"OC: {oc}, Column: {m}"
                                assert dst_coord[-1] < result.shape[-1], f"Invalid coordinate for source: {dst_coord[-1]}\n" \
                                                                         f"OC: {oc}, Column: {n}"

                                result[dst_coord[0]][dst_coord[1]][dst_coord[2]][dst_coord[3]] = weights[src_coord[0]][src_coord[1]][src_coord[2]][src_coord[3]]

    else:
        assert  "linear" in layer_type or  "gemm" in layer_type or "matmul" in layer_type, f"Invalid layer type: {layer_type}"
       # result = np.zeros((w_dim[1], w_dim[0]), dtype=weights.dtype)
        for ic in range(0, w_dim[0], tile_n):
            for oc in range(0, w_dim[1], tile_m):
                for n in range(tile_n):
                    for m in range(tile_m):
                        # Reverse order within a tile because systolic array is filled from last column first.
                        # Adjacent values in memory are filled in systolic array column.
                        # So, if systolic array size is 32x32, weight at (0, 0) should be in (31,0) in memory
                        # weight at (1, 0) should be in (31,1) in memory and so on.
                        # dst_coord = (nn + n, mm + tile_m - m - 1)
                        # src_coord = (nn+n, mm+m)
                        # coord_map[src_coord] = dst_coord
                        #
                        # result[nn + n][mm + tile_m - m - 1] = weights[nn + n][mm + m]
                        # result[kh][kw][nn + n][mm + tile_m - m - 1] = weights[kh][kw][nn + n][mm + m]
                        src_coord = ic + n, oc + m
                        dst_coord = ic + n, oc + tile_m - m - 1
                        coord_map[src_coord] = dst_coord

                        result[dst_coord[0]][dst_coord[1]] = \
                        weights[src_coord[0]][src_coord[1]]

    return result, coord_map

# Sequentially write out tiles of weights which will be written in DRAM
# A tile is written out in column-major order.
# Column major order to enable a tile-size sequential read from DRAM to go to column of systolic array
def tiled_flatten(weights, dram_tiling, cdlt, arch_config, layer_type = 'gemm'):

    if isinstance(weights, tuple):
        weights, coord_map = weights
        rev_coords = {v: k for k,v in coord_map.items()}
    else:
        rev_coords = {}
    final_coords = {}
    result = list()
    tile_m = arch_config['ARRAY_M']
    tile_n = arch_config['ARRAY_N']
    weight_symbols = list(cdlt.inputs[1].shape_symbols.keys())
    w_dim = weights.shape
    loop_order = [i for i in cdlt.get_loop_order() if i in weight_symbols]
    bw = arch_config['PARAM_BUF_CHANNEL_BW'] // 8
    systolic_array_row_size = weights.dtype.itemsize * tile_m
    systolic_array_column_size = weights.dtype.itemsize * tile_n
    # interleave_factor = bw // tile_n
    interleave_factor = bw // tile_n
    assert interleave_factor >= 1, f"Invalid interleave factor:\n" \
                                   f"Bandwidth: {bw}\n" \
                                   f"Sys array col size: {systolic_array_column_size}"
    assert tile_n == tile_m
    if layer_type == 'gemm' or 'matmul' in layer_type:

        big_tile_size_oc = dram_tiling[loop_order[0]]
        w_dim_outer = weight_symbols.index(loop_order[0])

        w_dim_inner = weight_symbols.index(loop_order[1])
        big_tile_size_ic = dram_tiling[loop_order[1]]

        assert tile_n * interleave_factor <= big_tile_size_oc
        for big_tile_oc in range(0, w_dim[w_dim_outer], big_tile_size_oc):  # Tile over OC
            for big_tile_ic in range(0, w_dim[w_dim_inner], big_tile_size_ic):  # Tile over IC
                for ic in range(0, big_tile_size_ic, tile_m):  # IC
                    for oc in range(0, big_tile_size_oc, tile_n * interleave_factor):  # OC
                        for n in range(tile_n):  # Rows
                            for m in range(tile_m):  # Columns
                                # src_coord = (big_tile_ic + ic + m, big_tile_oc + oc + n)
                                for k in range(interleave_factor):
                                    src_coord = [None, None]
                                    src_coord[w_dim_outer] = big_tile_oc + oc + n + (k*tile_n)
                                    src_coord[w_dim_inner] = big_tile_ic + ic + m
                                    src_coord = tuple(src_coord)
                                    dst_coord = np.unravel_index([len(result)], weights.shape)
                                    final_coords[rev_coords[src_coord]] = dst_coord
                                    result.append(weights[src_coord[0]][src_coord[1]])
    else:
        assert 'conv'  in layer_type
        big_tile_size_oc = dram_tiling['OC']
        big_tile_size_ic = dram_tiling['IC']
        assert tile_n * interleave_factor <= big_tile_size_oc, f"Invalid size with interleave factor:\n" \
                                                               f"Tile: {tile_n}\n" \
                                                               f"Interleave: {interleave_factor}\n" \
                                                               f"Big tile oc: {big_tile_size_oc}\n" \
                                                               f"Big tile ic: {big_tile_size_ic}"
        for big_tile_oc in range(0, w_dim[3], big_tile_size_oc):  # Tile over OC
            for big_tile_ic in range(0, w_dim[2], big_tile_size_ic):  # Tile over IC
                for kh in range(w_dim[0]):
                    for kw in range(w_dim[1]):
                        for ic in range(0, big_tile_size_ic, tile_m):  # IC
                            for oc in range(0, big_tile_size_oc, tile_n*interleave_factor):  # OC
                                for n in range(tile_n):  # Rows
                                    for m in range(tile_m):  # Columns
                                        for k in range(interleave_factor):
                                            # max_coord = max(max_coord, big_tile_oc + oc + n + (k*tile_n))
                                            src_coord = (kh, kw, big_tile_ic + ic + m, big_tile_oc + oc + n + (k*tile_n))

                                            dst_coord = np.unravel_index([len(result)], weights.shape)

                                            final_coords[rev_coords[src_coord]] = dst_coord
                                            result.append(weights[kh][kw][big_tile_ic + ic + m][big_tile_oc + oc + n + (k*tile_n)])
        # print(f"Max coord is: {max_coord}")
        # exit()
    return np.array(result, weights.dtype)

def dram_layout(weights, print_debug=False):
    dram_weights = []
    flat_weights = weights.flatten()
    flat_weights = flat_weights.astype(np.uint8)
    n = flat_weights.shape[0]
    assert n >= 4
    i = 0
    while i < (n-4):
        concat_weights = (flat_weights[i]) + \
                         (flat_weights[i + 1] << 8) + \
                         (flat_weights[i + 2] << 16) + \
                         (flat_weights[i + 3] << 24)
        dram_weights.append(concat_weights)

        i += 4
    concat_weights = flat_weights[i]
    if i + 1 < n:
        concat_weights += flat_weights[i + 1] << 8
    if i + 2 < n:
        concat_weights += flat_weights[i + 2] << 16
    if i + 3 < n:
        concat_weights += flat_weights[i + 3] << 24
    dram_weights.append(concat_weights)
    # dram_weights = [str(x) for x in dram_weights]
    return np.asarray(dram_weights)

def transform_data(data, operand_type, transformation, cdlt, hag):
    if operand_type == "input":
        if transformation == "shuffled":
            return dram_layout(data)
        elif transformation == "raw":
            return data
            # return [str(i) for i in data.flatten().tolist()]
        else:
            raise RuntimeError
    elif operand_type == "weights":
        tiling_parameters = cdlt.param_tiling
        # DRAM tiling is in level 1.
        dram_tiling = tiling_parameters[1]
        if transformation == "shuffled":
            shuffled_data = shuffle_weights(data, hag.meta_cfg, layer_type=cdlt.op_name)
            tiled_data = tiled_flatten(shuffled_data, dram_tiling, cdlt, hag.meta_cfg, layer_type=cdlt.op_name)
            dram_data = dram_layout(tiled_data)
            return dram_data
        elif transformation == "shuffled_raw":
            shuffled_data = shuffle_weights(data, hag.meta_cfg, layer_type=cdlt.op_name)
            tiled_data = tiled_flatten(shuffled_data, dram_tiling, cdlt, hag.meta_cfg, layer_type=cdlt.op_name)
            # raw_data = [str(i) for i in tiled_data]
            return tiled_data
        elif transformation == "raw":
            return data
            # return [str(i) for i in data.flatten().tolist()]
        else:
            raise RuntimeError



