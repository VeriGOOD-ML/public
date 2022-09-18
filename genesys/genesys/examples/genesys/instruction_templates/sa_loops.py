from codelets.adl.graph import ArchitectureNode

LOOPS_PER_LEVEL = 7

IS_SA_COMPUTE_LOOP = ""

def inner_sa_loops(hag: ArchitectureNode):
    instructions = []
    inner_loop_id_str = f"(op.loop_level % {LOOPS_PER_LEVEL}) + {LOOPS_PER_LEVEL}"
    conv_red_loop = '("conv" in cdlt.op_name and cdlt.loop_param_map[op.op_str] in ["IC", "KH", "KW"])'
    gemm_red_loop = '("gemm" in cdlt.op_name and cdlt.loop_param_map[op.op_str] == "N")'
    matmul_red_loop = '("matmul" in cdlt.op_name and cdlt.loop_param_map[op.op_str] == "N")'
    # reduction_loop_cond = '("conv" in cdlt.op_name and cdlt.loop_param_map[op.op_str] in ["IC", "KH", "KW"]) ' \
    #                       'or ("gemm" in cdlt.op_name and cdlt.loop_param_map[op.op_str] == "N")'
    reduction_loop_cond = f"{conv_red_loop} or {gemm_red_loop} or {matmul_red_loop}"


    conv_sa_loop = '("conv" in cdlt.op_name and cdlt.loop_param_map[op.op_str] in ["IC", "OC"])'
    gemm_sa_loop = '("gemm" in cdlt.op_name and cdlt.loop_param_map[op.op_str] in ["N","P"])'
    matmul_sa_loop = '("matmul" in cdlt.op_name and cdlt.loop_param_map[op.op_str] in ["N","P"])'
    # sa_loop_cond = '("conv" in cdlt.op_name and cdlt.loop_param_map[op.op_str] in ["IC", "OC"]) ' \
    #                       'or ("gemm" in cdlt.op_name and cdlt.loop_param_map[op.op_str] in ["N", "P"])'
    sa_loop_cond = f"{conv_sa_loop} or {gemm_sa_loop} or {matmul_sa_loop}"
    rd_loc = "{operand}.get_transfer_source('pe_array')"
    wr_loc = "{operand}.get_transfer_dest('pe_array')"
    # rd_check = "({operand}.has_transfer([{operand}.get_ld_storage_location(cdlt, 1), 'pe_array']))"
    rd_check = "({operand}.has_transfer([{operand}.get_ld_storage_location(cdlt, 1), 'pe_array']))"
    wr_check = "({operand}.has_transfer(['pe_array', {operand}.get_ld_storage_location(cdlt, 1)]))"
    systolic_compute = f"(cdlt.loop_compute_op(op))"

    # systolic_operands = f"({systolic_compute}.operands)"
    systolic_operands = f"({systolic_compute}.unique_operands)"
    # operand_storage_cond = f"(hag.is_adjacent(operand.get_ld_storage_location(cdlt, 1), 'pe_array'))"
    # output_storage_cond = f"(hag.is_adjacent( cdlt.outputs[0].get_ld_storage_location(cdlt, 1), 'pe_array'))"

    instr = hag.get_primitive_template("SA_LOOP_CFG")
    instr.add_condition(f'cdlt.is_direct_loop_dep(op, "pe_array")')

    instr.set_field_flex_param("LOOP_ID", inner_loop_id_str)
    sa_constraints = hag.get_subgraph_node("pe_array").dimensions[0]
    n_iter_str = f"int(np.ceil(op.iter_count / {sa_constraints})) - 1" \
                 f"if {sa_loop_cond} else op.iter_count - 1"
    instr.set_field_flex_param("NUM_ITERATIONS", f"{n_iter_str}")
    instructions.append(instr)

    instr = hag.get_primitive_template("SA_REDUCTION_LOOP")
    instr.add_condition(f'cdlt.is_direct_loop_dep(op, "pe_array") and ({reduction_loop_cond})')
    instr.set_field_flex_param("LOOP_DIM", "cdlt.loop_param_map[op.op_str]")
    instr.set_field_by_name("LOOP_TYPE", "INNER")
    instr.set_field_flex_param("LOOP_ID", f"op.loop_id % (cdlt.op_id_counters['loop']//2)")
    instructions.append(instr)

    # offset_str = f"operand.get_offset(cdlt, 2, op.loop_id, hag, outer_loop=False) * op.stride"
    offset_str = f"operand.get_offset(cdlt, op.loop_id, hag, {systolic_compute}.op_str, 'pe_array', outer_loop=False) * op.stride"
    macro_instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    macro_instr.add_iterable('operand', systolic_operands)
    macro_instr.add_condition(f'cdlt.is_direct_loop_dep(op, "pe_array") and {rd_check.format(operand="operand")}')
    macro_instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    macro_instr.set_field_by_name("ACCESS_TYPE", "RD")
    macro_instr.set_field_flex_param("LOOP_ID", inner_loop_id_str)
    macro_instr.set_field_flex_param("BUFFER", f"operand.get_ld_storage_location(cdlt, 1)")
    macro_instr.set_field_flex_param("STRIDE",
                                     f"program.extract_bits({offset_str}, 16, 0)"
                                     )

    micro_instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    micro_instr.add_iterable('operand', systolic_operands)
    micro_instr.add_condition(f'cdlt.is_direct_loop_dep(op, "pe_array") and {rd_check.format(operand="operand")}')
    micro_instr.set_field_by_name("LOW_HIGH_BITS", "HIGH")
    micro_instr.set_field_by_name("ACCESS_TYPE", "RD")
    micro_instr.set_field_flex_param("LOOP_ID", inner_loop_id_str)
    micro_instr.set_field_flex_param("BUFFER", f"operand.get_ld_storage_location(cdlt, 1)")
    micro_instr.set_field_flex_param("STRIDE",
                                     f"program.extract_bits({offset_str}, 16, 16)"
                                     )
    macro_instr.add_base_instruction(micro_instr)
    instructions.append(macro_instr)
    # wrt_offset =  f"program.extract_bits({systolic_compute}.dests[0].get_offset(cdlt, 2, op.loop_id, hag, outer_loop=False, movement_type='down'), 16, 0)"
    wrt_offset = f"program.extract_bits({systolic_compute}.dests[0].get_offset(cdlt, op.loop_id, hag, {systolic_compute}.op_str, 'pe_array', outer_loop=False, write=True), 16"
    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.add_condition(f'cdlt.is_direct_loop_dep(op, "pe_array") and {wr_check.format(operand=f"{systolic_compute}.dests[0]")}')
    instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    instr.set_field_by_name("ACCESS_TYPE", "WR")
    instr.set_field_flex_param("LOOP_ID", inner_loop_id_str)
    instr.set_field_flex_param("BUFFER", wr_loc.format(operand=f'{systolic_compute}.dests[0]'))
    instr.set_field_flex_param("STRIDE", f"{wrt_offset}, 0)")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.add_condition(f'cdlt.is_direct_loop_dep(op, "pe_array") and {wr_check.format(operand=f"{systolic_compute}.dests[0]")}')
    instr.set_field_by_name("LOW_HIGH_BITS", "HIGH")
    instr.set_field_by_name("ACCESS_TYPE", "WR")
    instr.set_field_flex_param("LOOP_ID", inner_loop_id_str)
    instr.set_field_flex_param("BUFFER", wr_loc.format(operand=f'{systolic_compute}.dests[0]'))

    instr.set_field_flex_param("STRIDE", f"{wrt_offset}, 16)")
    instructions.append(instr)
    return instructions


def outer_sa_loops(hag: ArchitectureNode):
    instructions = []
    pe_target_cond = 'cdlt.is_loop_node_target(op, "pe_array") and not cdlt.is_direct_loop_dep(op, "pe_array")'
    # operand_storage_cond = f"(hag.is_adjacent(operand.get_ld_storage_location(cdlt, 1), 'pe_array'))"
    ld_loc = "{operand}.get_transfer_dest('DRAM')"
    st_loc = "{operand}.get_transfer_source('DRAM')"
    # st_check = "({operand}.has_transfer([{operand}.get_ld_storage_location(cdlt, 1), 'DRAM']))"
    st_check = "({operand}.has_transfer([{operand}.get_transfer_dest('pe_array'), 'DRAM']))"
    ld_check = "({operand}.has_transfer(['DRAM', {operand}.get_ld_storage_location(cdlt, 1)]))"
    # temp_operand_storage_cond = f"(hag.is_adjacent(operand.get_ld_storage_location(cdlt, 1, return_null=True), 'pe_array'))"

    # loop_cond_str = f"{pe_target_cond} and {operand_storage_cond}"
    conv_red_loop = '("conv" in cdlt.op_name and cdlt.loop_param_map[op.op_str] in ["IC", "KH", "KW"])'
    gemm_red_loop = '("gemm" in cdlt.op_name and cdlt.loop_param_map[op.op_str] == "N")'
    matmul_red_loop = '("matmul" in cdlt.op_name and cdlt.loop_param_map[op.op_str] == "N")'
    # reduction_loop_cond = '("conv" in cdlt.op_name and cdlt.loop_param_map[op.op_str] in ["IC", "KH", "KW"]) ' \
    #                       'or ("gemm" in cdlt.op_name and cdlt.loop_param_map[op.op_str] == "N")'
    reduction_loop_cond = f"{conv_red_loop} or {gemm_red_loop} or {matmul_red_loop}"

    systolic_compute = f"(cdlt.loop_compute_op(op))"
    # systolic_operands = f"({systolic_compute}.operands)"

    # systolic_operands = f"({systolic_compute}.unique_operands)"
    systolic_ld = f"({systolic_compute}).sources"
    systolic_st = f"({systolic_compute}).dests"
    # operand_iter = f'cdlt.operands'
    # operand_iter = systolic_operands

    instr = hag.get_primitive_template("SA_LOOP_CFG")
    instr.add_condition(f"{pe_target_cond}")
    instr.set_field_flex_param("LOOP_ID", "op.loop_level")
    instr.set_field_flex_param("NUM_ITERATIONS", "op.iter_count - 1")
    instructions.append(instr)

    instr = hag.get_primitive_template("SA_REDUCTION_LOOP")
    instr.add_condition(f'{pe_target_cond} and ({reduction_loop_cond})')
    instr.set_field_flex_param("LOOP_DIM", "cdlt.loop_param_map[op.op_str]")
    instr.set_field_by_name("LOOP_TYPE", "OUTER")
    instr.set_field_flex_param("LOOP_ID", "op.loop_level")
    instructions.append(instr)


    denom_str = f"hag.get_subgraph_edge('DRAM', operand.get_ld_storage_location(cdlt, 1)).bandwidth"
    if hag.meta_cfg['ASIC_CONFIG']:
        # Product of stride of inner loops * stride * operand.dtype_size / bandwidth DRAM-BUF 256 (word/line size)
        # stride_str = f"operand.get_offset(cdlt, 1, op.loop_id, hag)*op.stride"
        # stride_str = f"operand.get_offset(cdlt, 1, op.loop_id, hag, outer_loop=True)*operand.dtype.bits()//{denom_str}"
        stride_str = f"operand.get_offset(cdlt, op.loop_id, hag, {systolic_compute}.op_str, 'DRAM', outer_loop=True)*operand.dtype.bits()//{denom_str}"
    else:
        # Product of iteration of inner loops * stride * operand.dtype_size / 8
        # stride_str = f"(operand.get_offset(cdlt, 1, op.loop_id, hag,outer_loop=True)*operand.dtype.bits()//8)"
        # stride_str = f"(operand.get_offset(cdlt, op.loop_id, hag, {systolic_compute}.op_str, 'DRAM', outer_loop=True)*operand.dtype.bits()//8)"
        stride_str = f"(operand.get_offset(cdlt, op.loop_id, hag, {systolic_compute}.op_str, 'DRAM', outer_loop=True)*operand.dtype.bits()//8)"

    # stride_str = f"operand.get_offset_(cdlt, 'DRAM', 1, op.loop_id, hag)*op.stride"
    macro_instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    macro_instr.add_iterable('operand', systolic_ld)
    macro_instr.add_condition(f"{pe_target_cond} and {ld_check.format(operand='operand')}")
    macro_instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    macro_instr.set_field_by_name("ACCESS_TYPE", "LD")
    macro_instr.set_field_flex_param("LOOP_ID", "op.loop_level")
    # macro_instr.set_field_flex_param("BUFFER", f"operand.get_ld_storage_location(cdlt, 1)")
    macro_instr.set_field_flex_param("BUFFER", f"{ld_loc.format(operand='operand')}")
    macro_instr.set_field_flex_param("STRIDE",
                                     f"program.extract_bits({stride_str}, 16, 0)")

    micro_instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    micro_instr.add_iterable('operand', systolic_ld)
    micro_instr.add_condition(f"{pe_target_cond} and {ld_check.format(operand='operand')}")
    micro_instr.set_field_by_name("LOW_HIGH_BITS", "HIGH")
    micro_instr.set_field_by_name("ACCESS_TYPE", "LD")
    micro_instr.set_field_flex_param("LOOP_ID", "op.loop_level")
    # micro_instr.set_field_flex_param("BUFFER", f"operand.get_ld_storage_location(cdlt, 1)")
    micro_instr.set_field_flex_param("BUFFER", f"{ld_loc.format(operand='operand')}")
    micro_instr.set_field_flex_param("STRIDE",
                                     f"program.extract_bits({stride_str}, 16, 16)")
    macro_instr.add_base_instruction(micro_instr)
    instructions.append(macro_instr)


    if hag.meta_cfg['ASIC_CONFIG']:
        denom_str = f"hag.get_subgraph_edge('DRAM', cdlt.outputs[0].get_ld_storage_location(cdlt, 1)).bandwidth"

        # out_stride_str = f"cdlt.outputs[0].get_offset(cdlt, 1, op.loop_id, hag, movement_type='down', outer_loop=True)*cdlt.outputs[0].dtype.bits() // {denom_str}"
        # out_stride_str = f"{systolic_compute}.dests[0].get_offset(cdlt, 1, op.loop_id, hag, movement_type='down', outer_loop=True)*{systolic_compute}.dests[0].dtype.bits() // {denom_str}"
        out_stride_str = f"{systolic_compute}.dests[0].get_offset(cdlt, op.loop_id, hag, {systolic_compute}.op_str, 'DRAM', write=True, outer_loop=True)*{systolic_compute}.dests[0].dtype.bits() // {denom_str}"
        # out_stride_str = f"cdlt.outputs[0].get_offset(cdlt, 1, op.loop_id, hag)*op.stride"
    else:
        # out_stride_str = f"(cdlt.outputs[0].get_offset(cdlt, 1, op.loop_id, hag, movement_type='down', outer_loop=True)*cdlt.outputs[0].dtype.bits()//8) "
        # out_stride_str = f"({systolic_compute}.dests[0].get_offset(cdlt, 1, op.loop_id, hag, movement_type='down', outer_loop=True)*{systolic_compute}.dests[0].dtype.bits()//8) "
        out_stride_str = f"({systolic_compute}.dests[0].get_offset(cdlt, op.loop_id, hag,{systolic_compute}.op_str, 'DRAM', write=True, outer_loop=True)*{systolic_compute}.dests[0].dtype.bits()//8) "

    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.add_condition(f"{pe_target_cond} and {st_check.format(operand=f'{systolic_compute}.dests[0]')}")
    instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    instr.set_field_by_name("ACCESS_TYPE", "ST")
    instr.set_field_flex_param("LOOP_ID", "op.loop_level")
    instr.set_field_flex_param("BUFFER", st_loc.format(operand=f'{systolic_compute}.dests[0]'))
    instr.set_field_flex_param("STRIDE", "program.extract_bits("
                                         f"{out_stride_str},"
                                         "16, 0)")

    instructions.append(instr)

    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.add_condition(f"{pe_target_cond} and {st_check.format(operand=f'{systolic_compute}.dests[0]')}")
    instr.set_field_by_name("LOW_HIGH_BITS", "HIGH")
    instr.set_field_by_name("ACCESS_TYPE", "ST")
    instr.set_field_flex_param("LOOP_ID", "op.loop_level")
    instr.set_field_flex_param("BUFFER", st_loc.format(operand=f'{systolic_compute}.dests[0]'))
    instr.set_field_flex_param("STRIDE", "program.extract_bits("
                                         f"{out_stride_str},"
                                         "16, 16)")
    instructions.append(instr)
    return instructions