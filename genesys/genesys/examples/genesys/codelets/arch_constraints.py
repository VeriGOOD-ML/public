from examples.genesys import DTYPE_MAP


def add_simd_constraint(hag, cdlt, fixed_dim):
    assert isinstance(fixed_dim, str)
    inpt_dtype = f"FXP{hag.meta_cfg['DATA_WIDTH']}"
    acc_dtype = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    simd_dims = hag.get_subgraph_node("SIMD").dimensions
    simd_edge = hag.get_subgraph_edge('DRAM', 'VMEM1')
    bandwidth = simd_edge.bandwidth
    cdlt.update_compilation_param(f"{fixed_dim}_hint2", f"size == {simd_dims[0]}")
    cdlt.update_compilation_param(f"{fixed_dim}_hint1", f"size*{DTYPE_MAP[acc_dtype].bits()} % {bandwidth} == 0")
    return cdlt

def add_multi_simd_constraint(hag, cdlt, fixed_dims):
    assert isinstance(fixed_dims, list)
    inpt_dtype = f"FXP{hag.meta_cfg['DATA_WIDTH']}"
    acc_dtype = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    simd_dims = hag.get_subgraph_node("SIMD").dimensions
    simd_edge = hag.get_subgraph_edge('DRAM', 'VMEM1')
    bandwidth = simd_edge.bandwidth
    cdlt.update_compilation_param(f"{fixed_dims}_hint2", f"size == {simd_dims[0]}")
    cdlt.update_compilation_param(f"{fixed_dims}_hint1", f"size*{DTYPE_MAP[acc_dtype].bits()} % {bandwidth} == 0")
    return cdlt

def add_flex_simd_constraints(hag, cdlt, dim_options):
    assert isinstance(dim_options, list)
    inpt_dtype = f"FXP{hag.meta_cfg['DATA_WIDTH']}"
    acc_dtype = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    simd_dims = hag.get_subgraph_node("SIMD").dimensions
    simd_edge = hag.get_subgraph_edge('DRAM', 'VMEM1')
    bandwidth = simd_edge.bandwidth
    l1_constraints = [f"sizes['{i}']*{DTYPE_MAP[acc_dtype].bits()} % {bandwidth} == 0" for i in dim_options]
    l1_constraint = f" or ".join(l1_constraints)
    cdlt.update_compilation_param("LEVEL1_hint", l1_constraint)
    #
    l2_constraints = [f"sizes['{i}'] == {simd_dims[0]}" for i in dim_options]
    l2_constraint = f" or ".join(l2_constraints)
    cdlt.update_compilation_param("LEVEL2_hint", l2_constraint)

    return cdlt

def add_simd_tile_constraint(hag, cdlt, fixed_dims):
    if isinstance(fixed_dims, str):
        fixed_dims = [fixed_dims]
    assert isinstance(fixed_dims, list)
    for fd in fixed_dims:
        cdlt.update_compilation_param(f"{fd}_hint1", f"split == 1")
    return cdlt

def add_conv_constraints(hag, cdlt, is_fusion=False):
    inpt_dtype = f"FXP{hag.meta_cfg['DATA_WIDTH']}"
    acc_dtype = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    sys_array_dims = hag.get_subgraph_node("pe_array").dimensions
    # if not is_fusion:
    #     cdlt.update_compilation_param("LOOP_TILE_ORDER", ["KH", "KW", "OC", "IC", "N", "OH", "OW"])

    wbuf_elements = hag.get_subgraph_node("WBUF").addressable_elements
    obuf_elements = hag.get_subgraph_node("OBUF").addressable_elements
    wbuf_index_size = f"sizes['KH']*sizes['KW']*sizes['IC']*sizes['OC']"
    obuf_index_size = f"sizes['N']*sizes['OH']*sizes['OW']*sizes['OC']"


    gt_one_tiles = f"np.prod(list(splits.values())) > 1"
    if is_fusion:
        ic_tiling = f"(splits['IC'] == 1)"
    else:
        ic_tiling = f"(splits['IC'] == 1 or any([splits['KH'] > 1, splits['KW'] > 1, splits['OH'] > 1, splits['OW'] > 1]))"


    constraint = f"{gt_one_tiles} and {ic_tiling}"
    ic_bandwidth = hag.get_subgraph_edge('DRAM', 'IBUF').bandwidth
    oc_bandwidth = hag.get_subgraph_edge('DRAM', 'WBUF').bandwidth

    if hag.meta_cfg['SA_TILE_CONSTR'] and not hag.meta_cfg['ASIC_CONFIG']:
        ic_hint0 = f"sizes['IC']*{DTYPE_MAP[acc_dtype].bits()} % {ic_bandwidth} == 0"
        oc_hint0 = f"sizes['OC']*{DTYPE_MAP[acc_dtype].bits()} % {oc_bandwidth} == 0"
        ic_hint1 = f"sizes['IC']*{DTYPE_MAP[inpt_dtype].bits()} % {ic_bandwidth} == 0"
        oc_hint1 = f"sizes['OC']*{DTYPE_MAP[inpt_dtype].bits()} % {oc_bandwidth} == 0"

        ic_hint = f"{ic_hint0} and {ic_hint1}"
        oc_hint = f"{oc_hint0} and {oc_hint1}"
        constraint = f"{constraint} and {ic_hint} and {oc_hint} and {wbuf_index_size} <= {wbuf_elements} and {obuf_index_size} <= {obuf_elements}"
    elif not hag.meta_cfg['SA_TILE_CONSTR']:
        ic_hint = f"sizes['IC']*{DTYPE_MAP[inpt_dtype].bits()} >= {ic_bandwidth}"
        oc_hint = f"sizes['OC']*{DTYPE_MAP[acc_dtype].bits()} >= {oc_bandwidth}"
        constraint = f"{constraint} and {ic_hint} and {oc_hint} and {wbuf_index_size} <= {wbuf_elements} and {obuf_index_size} <= {obuf_elements}"

    cdlt.update_compilation_param("LEVEL1_hint", constraint)

    ## DRAM to buffers
    cdlt.add_compilation_param("IC_hint1", f"size % {sys_array_dims[0]} == 0")
    cdlt.add_compilation_param("OC_hint1", f"size % {sys_array_dims[1]} == 0")
    cdlt.add_compilation_param("KH_hint1", f"split == 1")
    cdlt.add_compilation_param("KW_hint1", f"split == 1")

    ## Buffer to systolic array
    cdlt.add_compilation_param("IC_hint0", f"size % {sys_array_dims[0]} == 0")
    cdlt.add_compilation_param("OC_hint0", f"size % {sys_array_dims[1]} == 0")
    cdlt.add_compilation_param("KH_hint0", f"size == 1")
    cdlt.add_compilation_param("KW_hint0", f"size == 1")
    ####
    return cdlt

def add_gemm_constraints(hag, cdlt):

    sys_array_dims = hag.get_subgraph_node("pe_array").dimensions

    wbuf_elements = hag.get_subgraph_node("WBUF").addressable_elements
    obuf_elements = hag.get_subgraph_node("OBUF").addressable_elements
    wbuf_index_size = f"sizes['N']*sizes['P']"
    obuf_index_size = f"sizes['M']*sizes['P']"
    constraint = f"np.prod(list(splits.values())) > 1"

    if not hag.meta_cfg['ASIC_CONFIG']:
        inpt_dtype = f"FXP{hag.meta_cfg['DATA_WIDTH']}"
        acc_dtype = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
        sg_edge = hag.get_subgraph_edge('DRAM', 'IBUF')
        bandwidth = sg_edge.bandwidth
        fpga_constr = f"{wbuf_index_size} <= {wbuf_elements} and " \
                      f"{obuf_index_size} <= {obuf_elements} and " \
                      f"sizes['N']*{DTYPE_MAP[inpt_dtype].bits()} % {bandwidth} == 0"
        constraint = f"{constraint} and {fpga_constr}"
    cdlt.update_compilation_param("LEVEL1_hint", constraint)

    ## DRAM to buffers
    cdlt.update_compilation_param("N_hint1", f"size % {sys_array_dims[0]} == 0")
    cdlt.update_compilation_param("P_hint1", f"size % {sys_array_dims[1]} == 0")

    ## Buffer to systolic array
    cdlt.update_compilation_param("N_hint0", f"size % {sys_array_dims[0]} == 0")
    cdlt.update_compilation_param("P_hint0", f"size % {sys_array_dims[1]} == 0")
    return cdlt