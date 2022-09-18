from codelets.adl.graph import ArchitectureNode

OP_COMPUTE_CYCLES = {
    "SIGMOID": 4,
    "RELU": 0,
    "ADD": 0,
    "SUB": 0,
    "DIV": 0,
    "LEAKY_RELU": 0,
    "MAX": 0,
    "MIN": 0,
    "MUL": 0,
    "MACC": 2,
    "RSHIFT": 0,
    "LSHIFT": 0,
    "MOVE": 0,
    "COND_MOVE_TRUE": 0,
    "COND_MOVE_FALSE": 0,
    "NOT": 0,
    "AND": 0,
    "OR": 0,
    "NOP": 0,
    "ABS": 0,
    "SIGN": 0,
    "TANH": 4,
    "EXP": 0,
    "POW": 0,
    "LN": 0,
    "SQRT": 8,
    "INV_SQRT": 0,
    "LOG2": 0,
    "EQUAL": 0,
    "NEQ": 0,
    "GT": 0,
    "GTE": 0,
    "LTE": 0,
    "LT": 0,
    "32FXP_16FXP": 0,
    "32FXP_8FXP": 0,
    "32FXP_4FXP": 0,
    "16FXP_32FXP": 0,
    "8FXP_32FXP": 0,
    "4FXP_32FXP": 0,
    "32FP_16FP": 0,
    "16FP_32FP": 0,
    "32FP_162BFP": 0,
    "16BFP_32BFP": 0,
    "FLOOR": 0,
    "CEIL": 0,
    "TRANSPOSE": 0
}
DTYPE_CAST_NAMES = ["32FXP_16FXP", "32FXP_8FXP", "32FXP_4FXP", "16FXP_32FXP", "8FXP_32FXP", "4FXP_32FXP",
                        "32FP_16FP", "16FP_32FP"]
# Loops
# ALL_LOOP_ID = f"(len(cdlt.get_ops_by_type('loop'))//2)"
ALL_LOOP_ID = f"(len(cdlt.compute_node_loops('SIMD')))"
OPERAND_ITER = ("operand", "op.operands_by_unique_location")
LOOP_ITER = ('loop_op', f'cdlt.get_ops_by_type("loop")')

# Stride calculation
MVMT_TYPE = f"'up' if operand.data_path[0] == 'DRAM' else 'down'"
RD_WRITE_OPERAND = f"(True if operand in op.dests else False)"

LOOP_STRIDE = f"operand.get_offset(cdlt," \
              f"loop_op.loop_id," \
              f"hag, op.op_str, 'SIMD', write={RD_WRITE_OPERAND}, " \
              f"outer_loop=False)"



# Instruction generation conditions
IS_DEP_COND = f'loop_op.op_str in cdlt.all_dependencies(op.dependencies)'
COMPUTE_DEP = f'loop_op.op_str in op.operand_indices'
OUTER_LOOP_COND = f'loop_op.loop_level < op.loop_level'
IS_DIRECT_DEP_COND = f"cdlt.is_direct_loop_dep(loop_op, 'SIMD')"
# LOOP_CONDS = [IS_DEP_COND, COMPUTE_DEP, OUTER_LOOP_COND, IS_DIRECT_DEP_COND]
LOOP_CONDS = [COMPUTE_DEP, OUTER_LOOP_COND, IS_DIRECT_DEP_COND]

# Operand location string
OPERAND_LOC = f"op.get_operand_location(operand.name)"

BASE_SIGN_EXT_TEMPLATE = "({OPERAND}.get_mem_offset({OP_LOC})//(hag.get_subgraph_node({OP_LOC}).data_size)) if op.get_operand_location({OPERAND}.name) " \
                "!= 'IMM' else cdlt.temps.index({OPERAND})"

BASE_SIGN_EXT = BASE_SIGN_EXT_TEMPLATE.format(OPERAND="operand", OP_LOC=OPERAND_LOC)
# BASE_SIGN_EXT = f"(operand.get_mem_offset({OPERAND_LOC})//(hag.get_subgraph_node({OPERAND_LOC}).data_size) + 1) if op.get_operand_location(operand.name) " \
#                 f"!= 'IMM' else cdlt.temps.index(operand)"



ENUM_IS_DEP_COND = f'loop_op[1].op_str in cdlt.all_dependencies(op.dependencies)'
ENUM_OUTER_LOOP_COND = f'loop_op[1].loop_level < op.loop_level'
ENUM_IS_DIRECT_DEP_COND = f"cdlt.is_direct_loop_dep(loop_op[1], 'SIMD')"
ENUM_LOOP_CONDS = [ENUM_IS_DEP_COND, ENUM_OUTER_LOOP_COND, ENUM_IS_DIRECT_DEP_COND]


def simd_transpose_rd(hag):
    instructions = []
    loop_tabs = f"{ALL_LOOP_ID} + loop_op[0] - 1"
    src_loop_iter = ('loop_op', f'enumerate(cdlt.get_ops_by_type("loop"))')

    ## Source Read
    operand = "op.sources[0]"

    src_movement_type = f"'up' if {operand}.data_path[0] == 'DRAM' else 'down'"
    src_loop_iters = f"cdlt.inner_iter({operand}, loop_op[1], loop_op[0]) - 1"

    src_op_loc = f"op.get_operand_location({operand}.name)"

    src_ns_idx = f"(loop_op[1].loop_id % {ALL_LOOP_ID}) + ({operand}.get_mem_index({src_op_loc}) * {ALL_LOOP_ID}) if op.get_operand_location({operand}.name) != 'IMM' " \
                 f"else cdlt.temps.index({operand})"
    # src_base_sign_ext = f"({operand}.get_mem_offset({src_op_loc})//({operand}.dtype.bits()) + 1) if {src_op_loc} " \
    #                     f"!= 'IMM' else cdlt.temps.index({operand})"
    src_base_sign_ext = BASE_SIGN_EXT_TEMPLATE.format(OPERAND=operand, OP_LOC=src_op_loc)
    src_loop_stride = f"cdlt.inner_stride({operand}, loop_op[1], loop_op[0])"

    src_base_instr = hag.get_primitive_template("PERM_SET_BASE_ADDR")
    src_base_instr.set_print_tabs(ALL_LOOP_ID)
    src_base_instr.set_field_by_name('RD_WR', "RD")
    src_base_instr.set_field_flex_param('BASE_ADDR', src_base_sign_ext)
    instructions.append(src_base_instr)

    src_stride_instr = hag.get_primitive_template("PERM_SET_LOOP_STRIDE")
    src_stride_instr.set_print_tabs(loop_tabs)
    src_stride_instr.add_iterable(*src_loop_iter)
    src_stride_instr.add_condition(" and ".join(ENUM_LOOP_CONDS))
    src_stride_instr.set_field_by_name('RD_WR', "RD")
    src_stride_instr.set_field_flex_param('LOOP_INDEX_ID', src_ns_idx)
    src_stride_instr.set_field_flex_param('STRIDE', src_loop_stride)

    src_iter_instr = hag.get_primitive_template("PERM_SET_LOOP_ITER")
    src_iter_instr.set_print_tabs(loop_tabs)
    src_iter_instr.add_iterable(*src_loop_iter)
    src_iter_instr.add_condition(" and ".join(ENUM_LOOP_CONDS))
    src_iter_instr.set_field_by_name('RD_WR', "RD")
    src_iter_instr.set_field_flex_param('LOOP_INDEX_ID', src_ns_idx)
    src_iter_instr.set_field_flex_param('NUM_ITERS', src_loop_iters)
    src_stride_instr.add_base_instruction(src_iter_instr)
    instructions.append(src_stride_instr)

    return instructions

def simd_transpose_wr(hag):
    instructions = []
    simd_size = hag.get_subgraph_node("SIMD").dimensions[0]
    loop_tabs = f"{ALL_LOOP_ID} + loop_op[0]"
    dst_loop_iter = ('loop_op', f'enumerate(cdlt.get_ops_by_type("loop")[:-2] + [list(cdlt.get_ops_by_type("loop"))[-1]] + [list(cdlt.get_ops_by_type("loop"))[-2]])')
    transpose_axes = f"[i for i,s in enumerate(op.sources[0].shape_list) if s != op.dests[0].shape_list[i]]"
    # Previous dst_outer_iters
    # dst_outer_iters = f'cdlt.get_ops_by_type("loop")[-2].iter_count // {simd_size}'
    # New dst outer_iters
    dst_outer_iters = f'max([l.iter_count for i, l in enumerate(cdlt.get_ops_by_type("loop")) if i >= len(cdlt.get_ops_by_type("loop"))/2]) // {simd_size}'
    # end changes

    dst_outer_stride = f'cdlt.get_ops_by_type("loop")[-1].iter_count'

    operand = "op.dests[0]"
    src_operand = "op.sources[0]"

    dst_loop_iter_base = f"cdlt.inner_iter({operand}, loop_op[1], loop_op[0])"

    ## Changes for benchmarking
    # Original implementation
    # dst_loop_iters = f"({dst_loop_iter_base}-1) if loop_op[0] != list({dst_loop_iter[1]})[-1][0] else (({dst_loop_iter_base})//({dst_outer_iters})) - 1)"

    # This is option 1, which worked, but is more inaccurate relative to the number of iterations
    # alt_dst_iters = f"1 if {dst_outer_iters} == 0 else ((({dst_loop_iter_base})//({dst_outer_iters})) - 1)"
    # dst_loop_iters = f"({dst_loop_iter_base}-1) if loop_op[0] != list({dst_loop_iter[1]})[-1][0] else {alt_dst_iters}"

    # Option 2, which uses the lowest tile level to determine iterations, and is more accurate
    alt_dst_iters = f"loop_op[1].iter_count//{simd_size} - 1"
    cond = f"cdlt.param_tiling[2][cdlt.loop_param_map[loop_op[1].op_str]] != {simd_size}"
    dst_loop_iters = f"loop_op[1].iter_count-1 if {cond} else {alt_dst_iters}"

    ## End changes for benchmarking

    dst_op_loc = f"op.get_operand_location({operand}.name)"

    dst_ns_idx = f"(loop_op[1].loop_id % {ALL_LOOP_ID}) + ({operand}.get_mem_index({dst_op_loc}) * {ALL_LOOP_ID}) + 1 if op.get_operand_location({operand}.name) != 'IMM' " \
             f"else cdlt.temps.index({operand}) + 1"


    dst_base_sign_ext = BASE_SIGN_EXT_TEMPLATE.format(OPERAND=operand, OP_LOC=dst_op_loc)

    dst_loop_stride = f"cdlt.inner_stride({operand}, loop_op[1], loop_op[0])"

    dst_base_instr = hag.get_primitive_template("PERM_SET_BASE_ADDR")
    dst_base_instr.set_print_tabs(ALL_LOOP_ID)
    dst_base_instr.set_field_by_name('RD_WR', "WR")
    dst_base_instr.set_field_flex_param('BASE_ADDR', dst_base_sign_ext)
    instructions.append(dst_base_instr)
    ## Outer loop for dest

    outer_stride_instr = hag.get_primitive_template("PERM_SET_LOOP_STRIDE")
    outer_stride_instr.set_print_tabs(f"{ALL_LOOP_ID} + 1")
    outer_stride_instr.set_field_by_name('RD_WR', "WR")
    outer_stride_instr.set_field_value('LOOP_INDEX_ID', 0)
    outer_stride_instr.set_field_flex_param('STRIDE', f"{dst_outer_stride}")
    instructions.append(outer_stride_instr)

    outer_iter_instr = hag.get_primitive_template("PERM_SET_LOOP_ITER")
    outer_iter_instr.set_print_tabs(f"{ALL_LOOP_ID} + 1")
    outer_iter_instr.set_field_by_name('RD_WR', "WR")
    outer_iter_instr.set_field_value('LOOP_INDEX_ID', 0)
    outer_iter_instr.set_field_flex_param('NUM_ITERS', f"({dst_outer_iters}) - 1")
    instructions.append(outer_iter_instr)

    ## Inner loops for dest

    dst_stride_instr = hag.get_primitive_template("PERM_SET_LOOP_STRIDE")
    dst_stride_instr.set_print_tabs(loop_tabs)
    dst_stride_instr.add_iterable(*dst_loop_iter)
    dst_stride_instr.add_condition(" and ".join(ENUM_LOOP_CONDS))
    dst_stride_instr.set_field_by_name('RD_WR', "WR")
    dst_stride_instr.set_field_flex_param('LOOP_INDEX_ID', dst_ns_idx)
    dst_stride_instr.set_field_flex_param('STRIDE', dst_loop_stride)


    dst_iter_instr = hag.get_primitive_template("PERM_SET_LOOP_ITER")
    dst_iter_instr.set_print_tabs(loop_tabs)
    dst_iter_instr.add_iterable(*dst_loop_iter)
    dst_iter_instr.add_condition(" and ".join(ENUM_LOOP_CONDS))
    dst_iter_instr.set_field_by_name('RD_WR', "WR")
    dst_iter_instr.set_field_flex_param('LOOP_INDEX_ID', dst_ns_idx)
    dst_iter_instr.set_field_flex_param('NUM_ITERS', dst_loop_iters)
    dst_stride_instr.add_base_instruction(dst_iter_instr)
    instructions.append(dst_stride_instr)
    return instructions

def simd_transpose(hag):
    instructions = []
    src_op_loc = f"op.get_operand_location(op.sources[0].name)"
    dst_op_loc = f"op.get_operand_location(op.dests[0].name)"

    instructions += simd_transpose_rd(hag)
    instructions += simd_transpose_wr(hag)


    start_instr = hag.get_primitive_template("PERM_START")
    start_instr.set_print_tabs(f"{ALL_LOOP_ID}*2 + {ALL_LOOP_ID}//2 + 1")
    start_instr.set_field_flex_param('DST_NS_ID', dst_op_loc)
    start_instr.set_field_flex_param('SRC_NS_ID', src_op_loc)
    start_instr.set_field_by_name('SHUFFLE_BANKS', 'DO_SHUFFLE')
    instructions.append(start_instr)


    return instructions

def add_pow_loop(hag):
    # Params
    simd_size = hag.get_subgraph_node("SIMD").dimensions[0]



    # Index generation
    ns_idx = f"(loop_op.loop_id % {ALL_LOOP_ID}) + (operand.get_mem_index({OPERAND_LOC}) * {ALL_LOOP_ID}) if op.get_operand_location(operand.name) != 'IMM' else cdlt.temps.index(operand)"
    # base_sign_ext = f"operand.get_mem_offset({OPERAND_LOC})//(operand.dtype.bits()) if op.get_operand_location(operand.name) " \
    #                 f"!= 'IMM' else cdlt.temps.index(operand)"


    instructions = []
    pow_ns_idx = f"({ALL_LOOP_ID}) + (operand.get_mem_index({OPERAND_LOC}) * {ALL_LOOP_ID}) if op.get_operand_location(operand.name) != 'IMM' else cdlt.temps.index(operand)"

    macro_instr = hag.get_primitive_template("BASE_SIGN_EXT")
    macro_instr.set_print_tabs(ALL_LOOP_ID)
    macro_instr.add_iterable(*OPERAND_ITER)
    macro_instr.set_field_flex_param('NS_ID', OPERAND_LOC)
    macro_instr.set_field_flex_param('NS_INDEX_ID', pow_ns_idx)
    macro_instr.set_field_flex_param('IMM', BASE_SIGN_EXT)
    #
    sub_instr = hag.get_primitive_template("STRIDE_SIGN_EXT")
    sub_instr.set_print_tabs(ALL_LOOP_ID)
    sub_instr.add_iterable(*OPERAND_ITER)
    sub_instr.set_field_flex_param('NS_ID', OPERAND_LOC)
    sub_instr.set_field_flex_param('NS_INDEX_ID', pow_ns_idx)
    sub_instr.set_field_flex_param('IMM', '0')
    macro_instr.add_base_instruction(sub_instr)
    instructions.append(macro_instr)

    exp_tabs = f"len(cdlt.get_ops_by_type('loop'))"
    macro_instr = hag.get_primitive_template("SET_ITER")
    macro_instr.set_print_tabs(ALL_LOOP_ID)
    macro_instr.set_field_flex_param("LOOP_ID", f"0")
    macro_instr.set_field_flex_param("NUM_ITER",
                                     f"cdlt.required_params['exp'].value - 1")

    # Extra set index
    sub_instr = hag.get_primitive_template("SET_INDEX")
    set_index_fmt = "({all_loop_id}) + ({operand}.get_mem_index({op_loc}) * {all_loop_id}) if op.get_operand_location({operand}.name) != 'IMM' else cdlt.temps.index({operand})"

    sub_instr.set_print_tabs(ALL_LOOP_ID)
    sub_instr.set_field_flex_param("DST_NS_ID", "op.get_operand_location(op.dests[0].name)")
    sub_instr.set_field_flex_param("DST_INDEX_ID",
                                   set_index_fmt.format(all_loop_id=ALL_LOOP_ID, operand="op.dests[0]",
                                                        op_loc="op.get_operand_location(op.dests[0].name)"))
    sub_instr.set_field_flex_param("SRC1_NS_ID", "op.get_operand_location(op.sources[0].name)")
    sub_instr.set_field_flex_param("SRC1_INDEX_ID",
                                   set_index_fmt.format(all_loop_id=ALL_LOOP_ID, operand="op.sources[0]",
                                                        op_loc="op.get_operand_location(op.sources[0].name)"))
    sub_instr.set_field_flex_param("SRC2_NS_ID",
                                   f"'IMM' if len(op.sources) == 1 else op.get_operand_location(op.sources[1].name)")
    src2_idx = f"0 if len(op.sources) == 1 else " + set_index_fmt.format(all_loop_id=ALL_LOOP_ID,
                                                                         operand="op.sources[1]",
                                                                         op_loc="op.get_operand_location(op.sources[1].name)")
    sub_instr.set_field_flex_param("SRC2_INDEX_ID", src2_idx)
    macro_instr.add_base_instruction(sub_instr)
    instructions.append(macro_instr)
    if hag.meta_cfg['ADDR_GEN_TEST']:
        instructions += sw_addr_nops_nested("POW", hag)
    return instructions

def add_pow_compute(hag, rd_op_str):
    instructions = []
    instr = hag.get_primitive_template("MUL")
    instr.set_field_flex_param("DST_NS_ID", "op.get_operand_location(op.dests[0].name)")
    instr.set_field_flex_param("DST_INDEX_ID",
                               f"op.dests[0].get_mem_index(op.get_operand_location(op.dests[0].name))")
    instr.set_field_flex_param("SRC1_NS_ID", "op.get_operand_location(op.sources[0].name)")
    instr.set_field_flex_param("SRC1_INDEX_ID", rd_op_str.format(IDX=0))
    instr.set_field_flex_param("SRC2_NS_ID", "op.get_operand_location(op.sources[1].name)")
    instr.set_field_flex_param("SRC2_INDEX_ID", rd_op_str.format(IDX=1))
    instructions.append(instr)
    return instructions

def insert_dtype_cfg_from_cast(op_name, hag):
    instructions = []
    instr = hag.get_primitive_template("DTYPE_CFG")
    instr.set_field_flex_param("DTYPE", "str(op.dests[0].dtype.bits()) + op.dests[0].dtype.type")
    instr.set_field_flex_param("DST_BITS", "op.dests[0].dtype.exp")
    instr.set_field_flex_param("SRC1_BITS", "op.sources[0].dtype.exp")
    instr.set_field_flex_param("SRC2_BITS", "op.sources[0].dtype.exp")
    instructions.append(instr)
    return instructions

def sw_addr_nops(op_name, hag):
    instructions = []

    ## First, need to do BASE/SIGN_EXT



    ## Now, create the loop
    ## First loop
    all_loop_list = LOOP_ITER[1]
    other_constr = " and ".join(LOOP_CONDS + ["loop_op.op_str in op.dependencies"])

    tgt_loop_list = f"[loop_op.iter_count // cdlt.param_tiling[2][cdlt.loop_param_map[loop_op.op_str]] for loop_op in {all_loop_list} if {other_constr}]"
    loop_count = f"np.prod({tgt_loop_list})"
    filterd_operand_locs = "([op.get_operand_location(o.name) for o in op.operands if op.get_operand_location(o.name) != 'IMM'])"
    bin_un_count = f"{loop_count}*3 if len({filterd_operand_locs}) > 2 else {loop_count}*2"


    macro_instr = hag.get_primitive_template("BASE_SIGN_EXT")
    macro_instr.set_field_by_name('NS_ID', "VMEM1")
    macro_instr.set_print_tabs("op.loop_level - op.loop_level//2")
    macro_instr.set_field_flex_param('NS_INDEX_ID', "0")
    macro_instr.set_field_value('IMM', 1)
    #

    sub_instr = hag.get_primitive_template("STRIDE_SIGN_EXT")
    sub_instr.set_field_by_name('NS_ID', "VMEM1")
    sub_instr.set_print_tabs("op.loop_level - op.loop_level//2")
    sub_instr.set_field_flex_param('NS_INDEX_ID', "0")
    sub_instr.set_field_flex_param('IMM', f"0")
    macro_instr.add_base_instruction(sub_instr)

    sub_instr = hag.get_primitive_template("SET_ITER")
    sub_instr.set_print_tabs("op.loop_level - op.loop_level//2")
    sub_instr.set_field_flex_param("LOOP_ID", f"0")
    sub_instr.set_field_flex_param("NUM_ITER", loop_count)
    macro_instr.add_base_instruction(sub_instr)

    sub_instr = hag.get_primitive_template("SET_INDEX")
    sub_instr.set_print_tabs("op.loop_level - op.loop_level//2")
    sub_instr.set_field_by_name("DST_NS_ID", "VMEM1")
    sub_instr.set_field_flex_param("DST_INDEX_ID", "0")
    sub_instr.set_field_by_name("SRC1_NS_ID", "VMEM1")
    sub_instr.set_field_flex_param("SRC1_INDEX_ID", "0")
    sub_instr.set_field_by_name("SRC2_NS_ID", "VMEM1")
    sub_instr.set_field_flex_param("SRC2_INDEX_ID", '0')
    macro_instr.add_base_instruction(sub_instr)
    instructions.append(macro_instr)

    noop_instr = hag.get_primitive_template("NOP")
    noop_instr.set_print_tabs("op.loop_level - op.loop_level//2 + 1")
    instructions.append(noop_instr)

    ### Second Loop
    macro_instr = hag.get_primitive_template("BASE_SIGN_EXT")
    macro_instr.set_field_by_name('NS_ID', "VMEM1")
    macro_instr.set_print_tabs("op.loop_level - op.loop_level//2")
    macro_instr.set_field_flex_param('NS_INDEX_ID', "0")
    macro_instr.set_field_value('IMM', 1)
    #

    sub_instr = hag.get_primitive_template("STRIDE_SIGN_EXT")
    sub_instr.set_field_by_name('NS_ID', "VMEM1")
    sub_instr.set_print_tabs("op.loop_level - op.loop_level//2")
    sub_instr.set_field_flex_param('NS_INDEX_ID', "0")
    sub_instr.set_field_flex_param('IMM', f"0")
    macro_instr.add_base_instruction(sub_instr)
    sub_instr = hag.get_primitive_template("SET_ITER")
    sub_instr.set_print_tabs("op.loop_level - op.loop_level//2")
    sub_instr.set_field_flex_param("LOOP_ID", f"0")
    sub_instr.set_field_flex_param("NUM_ITER", loop_count)
    macro_instr.add_base_instruction(sub_instr)

    sub_instr = hag.get_primitive_template("SET_INDEX")
    sub_instr.set_print_tabs("op.loop_level - op.loop_level//2")
    sub_instr.set_field_by_name("DST_NS_ID", "VMEM1")
    sub_instr.set_field_flex_param("DST_INDEX_ID", "0")
    sub_instr.set_field_by_name("SRC1_NS_ID", "VMEM1")
    sub_instr.set_field_flex_param("SRC1_INDEX_ID", "0")
    sub_instr.set_field_by_name("SRC2_NS_ID", "VMEM1")
    sub_instr.set_field_flex_param("SRC2_INDEX_ID", '0')
    macro_instr.add_base_instruction(sub_instr)
    instructions.append(macro_instr)

    noop_instr = hag.get_primitive_template("NOP")
    noop_instr.set_print_tabs("op.loop_level - op.loop_level//2 + 1")
    instructions.append(noop_instr)


    ### Optional third
    third_cond = f"len({filterd_operand_locs}) > 2"
    macro_instr = hag.get_primitive_template("BASE_SIGN_EXT")
    macro_instr.add_condition(third_cond)
    macro_instr.set_field_by_name('NS_ID', "VMEM1")
    macro_instr.set_print_tabs("op.loop_level - op.loop_level//2")
    macro_instr.set_field_flex_param('NS_INDEX_ID', "0")
    macro_instr.set_field_value('IMM', 1)
    #

    sub_instr = hag.get_primitive_template("STRIDE_SIGN_EXT")
    sub_instr.add_condition(third_cond)
    sub_instr.set_field_by_name('NS_ID', "VMEM1")
    sub_instr.set_print_tabs("op.loop_level - op.loop_level//2")
    sub_instr.set_field_flex_param('NS_INDEX_ID', "0")
    sub_instr.set_field_flex_param('IMM', f"0")
    macro_instr.add_base_instruction(sub_instr)
    sub_instr = hag.get_primitive_template("SET_ITER")
    sub_instr.set_print_tabs("op.loop_level - op.loop_level//2")
    sub_instr.set_field_flex_param("LOOP_ID", f"0")
    sub_instr.set_field_flex_param("NUM_ITER", loop_count)
    macro_instr.add_base_instruction(sub_instr)

    sub_instr = hag.get_primitive_template("SET_INDEX")
    sub_instr.add_condition(third_cond)
    sub_instr.set_print_tabs("op.loop_level - op.loop_level//2")
    sub_instr.set_field_by_name("DST_NS_ID", "VMEM1")
    sub_instr.set_field_flex_param("DST_INDEX_ID", "0")
    sub_instr.set_field_by_name("SRC1_NS_ID", "VMEM1")
    sub_instr.set_field_flex_param("SRC1_INDEX_ID", "0")
    sub_instr.set_field_by_name("SRC2_NS_ID", "VMEM1")
    sub_instr.set_field_flex_param("SRC2_INDEX_ID", '0')
    macro_instr.add_base_instruction(sub_instr)
    instructions.append(macro_instr)

    noop_instr = hag.get_primitive_template("NOP")
    noop_instr.add_condition(third_cond)

    noop_instr.set_print_tabs("op.loop_level - op.loop_level//2 + 1")
    instructions.append(noop_instr)

    return instructions



def sw_addr_nops_nested(op_name, hag):
    instructions = []

    ## First, need to do BASE/SIGN_EXT

    loop_idx_offset = 0 if op_name != "POW" else 1


    ## Now, create the loop
    ## First loop
    all_loop_list = LOOP_ITER[1]
    other_constr = " and ".join(LOOP_CONDS + ["loop_op.op_str in op.dependencies"])
    ns_idx = f"(loop_op.loop_level % {ALL_LOOP_ID})"

    tgt_loop_list = f"[loop_op.iter_count // cdlt.param_tiling[2][cdlt.loop_param_map[loop_op.op_str]] for loop_op in {all_loop_list} if {other_constr}]"
    filtered_loops = f"[loop_op for loop_op in {all_loop_list} if {other_constr}]"
    loop_count = f"np.prod({tgt_loop_list})"
    filterd_operand_locs = "([op.get_operand_location(o.name) for o in op.operands if op.get_operand_location(o.name) != 'IMM'])"

    multiplier = f"(3 if len({filterd_operand_locs}) > 2 else 2)"
    multiplier_val = f"({multiplier} if loop_op.loop_id == {filtered_loops}[0].loop_id else 1)"

    lconds = " and ".join(LOOP_CONDS)

    macro_instr = hag.get_primitive_template("BASE_SIGN_EXT")
    macro_instr.set_print_tabs(ALL_LOOP_ID)
    macro_instr.add_iterable(*LOOP_ITER)
    macro_instr.add_condition(lconds)
    macro_instr.set_field_by_name('NS_ID', "VMEM1")
    macro_instr.set_print_tabs("op.loop_level - op.loop_level//2")
    macro_instr.set_field_flex_param('NS_INDEX_ID', ns_idx)
    macro_instr.set_field_value('IMM', 0)
    #

    sub_instr = hag.get_primitive_template("STRIDE_SIGN_EXT")
    sub_instr.set_print_tabs(ALL_LOOP_ID)
    sub_instr.add_iterable(*LOOP_ITER)
    sub_instr.add_condition(lconds)
    sub_instr.set_field_by_name('NS_ID', "VMEM1")
    sub_instr.set_field_flex_param('NS_INDEX_ID', ns_idx)
    sub_instr.set_field_flex_param('IMM', 0)
    macro_instr.add_base_instruction(sub_instr)
    instructions.append(macro_instr)

    ## Now generate loops

    macro_instr = hag.get_primitive_template("SET_ITER")
    macro_instr.set_print_tabs("loop_op.loop_level")
    macro_instr.add_iterable(*LOOP_ITER)
    macro_instr.add_condition(other_constr)
    macro_instr.set_field_flex_param("LOOP_ID", f"(loop_op.loop_level % {ALL_LOOP_ID}) + {loop_idx_offset}")
    macro_instr.set_field_flex_param("NUM_ITER", f"(loop_op.iter_count // cdlt.param_tiling[2][cdlt.loop_param_map[loop_op.op_str]])*{multiplier_val}")


    sub_instr = hag.get_primitive_template("SET_INDEX")
    sub_instr.set_print_tabs("loop_op.loop_level")
    sub_instr.add_iterable(*LOOP_ITER)
    sub_instr.add_condition(other_constr)
    sub_instr.set_field_by_name("DST_NS_ID", "VMEM1")
    sub_instr.set_field_flex_param("DST_INDEX_ID", ns_idx)
    sub_instr.set_field_by_name("SRC1_NS_ID", "VMEM1")
    sub_instr.set_field_flex_param("SRC1_INDEX_ID", ns_idx)
    sub_instr.set_field_by_name("SRC2_NS_ID", "VMEM1")
    sub_instr.set_field_flex_param("SRC2_INDEX_ID", ns_idx)
    macro_instr.add_base_instruction(sub_instr)
    instructions.append(macro_instr)

    # Compute instruction
    instr = hag.get_primitive_template("SET_INST")
    instr.add_condition("cdlt.op_id_counters['compute'] - 1 > op.op_id")
    instr.set_field_flex_param("SINGLE_NESTED", "1")
    instr.set_field_flex_param("NUM_INSTR", "1")
    instructions.append(instr)

    noop_instr = hag.get_primitive_template("NOP")
    instructions.append(noop_instr)


    return instructions

def simd_alu_template(op_name, hag: ArchitectureNode):
    if op_name == "TRANSPOSE":
        return simd_transpose(hag)
    instructions = []

    loop_idx_offset = 0 if op_name != "POW" else 1

    if op_name in DTYPE_CAST_NAMES:
        instructions += insert_dtype_cfg_from_cast(op_name, hag)
    ### Base and stride first
    instructions += base_sign_ext_gen(op_name, hag)


    if op_name == "POW":
        # Additional loop requires additional base/offset
      instructions += add_pow_loop(hag)

    ### iters and index
    other_constr = LOOP_CONDS + ["loop_op.op_str in op.dependencies"]
    macro_instr = hag.get_primitive_template("SET_ITER")
    macro_instr.set_print_tabs("loop_op.loop_level")
    macro_instr.add_iterable(*LOOP_ITER)
    macro_instr.add_condition(" and ".join(other_constr))
    macro_instr.set_field_flex_param("LOOP_ID", f"(loop_op.loop_level % {ALL_LOOP_ID}) + {loop_idx_offset}")
    macro_instr.set_field_flex_param("NUM_ITER", f"loop_op.iter_count // cdlt.param_tiling[2][cdlt.loop_param_map[loop_op.op_str]]")


    sub_instr = hag.get_primitive_template("SET_INDEX")
    set_index_fmt = "(loop_op.loop_level % {all_loop_id}) + ({operand}.get_mem_index({op_loc}) * {all_loop_id}) if op.get_operand_location({operand}.name) != 'IMM' else cdlt.temps.index({operand})"

    sub_instr.set_print_tabs("loop_op.loop_level")
    sub_instr.add_iterable(*LOOP_ITER)
    sub_instr.add_condition(" and ".join(other_constr))
    sub_instr.set_field_flex_param("DST_NS_ID", "op.get_operand_location(op.dests[0].name)")
    sub_instr.set_field_flex_param("DST_INDEX_ID",
                                   set_index_fmt.format(all_loop_id=ALL_LOOP_ID, operand="op.dests[0]",
                                                        op_loc="op.get_operand_location(op.dests[0].name)"))
    sub_instr.set_field_flex_param("SRC1_NS_ID", "op.get_operand_location(op.sources[0].name)")
    sub_instr.set_field_flex_param("SRC1_INDEX_ID",
                                   set_index_fmt.format(all_loop_id=ALL_LOOP_ID, operand="op.sources[0]",
                                                        op_loc="op.get_operand_location(op.sources[0].name)"))
    sub_instr.set_field_flex_param("SRC2_NS_ID",
                                   f"'IMM' if len(op.sources) == 1 else op.get_operand_location(op.sources[1].name)")
    src2_idx = f"0 if len(op.sources) == 1 else " + set_index_fmt.format(all_loop_id=ALL_LOOP_ID,
                                                                         operand="op.sources[1]",
                                                                         op_loc="op.get_operand_location(op.sources[1].name)")
    sub_instr.set_field_flex_param("SRC2_INDEX_ID", src2_idx)
    macro_instr.add_base_instruction(sub_instr)
    instructions.append(macro_instr)

    # Compute instruction
    instr = hag.get_primitive_template("SET_INST")
    instr.add_condition("cdlt.op_id_counters['compute'] - 1 > op.op_id")
    instr.set_field_flex_param("SINGLE_NESTED", "0 if op.num_loop_dependencies == 1 else 1")
    instr.set_field_flex_param("NUM_INSTR", "1")
    instructions.append(instr)
    rd_op_str = "op.sources[{IDX}].get_mem_index(op.get_operand_location(op.sources[{IDX}].name)) if op.get_operand_location(op.sources[{IDX}].name) != 'IMM' else cdlt.temps.index(op.sources[{IDX}])"

    if op_name != "POW":
        instr = hag.get_primitive_template(op_name)
        instr.add_condition("len(op.sources) > 1")
        instr.set_field_flex_param("DST_NS_ID", "op.get_operand_location(op.dests[0].name)")
        instr.set_field_flex_param("DST_INDEX_ID", f"op.dests[0].get_mem_index(op.get_operand_location(op.dests[0].name))")
        instr.set_field_flex_param("SRC1_NS_ID", "op.get_operand_location(op.sources[0].name)")
        instr.set_field_flex_param("SRC1_INDEX_ID", rd_op_str.format(IDX=0))
        instr.set_field_flex_param("SRC2_NS_ID", "op.get_operand_location(op.sources[1].name)")
        instr.set_field_flex_param("SRC2_INDEX_ID", rd_op_str.format(IDX=1))
        instructions.append(instr)

        instr = hag.get_primitive_template(op_name)
        instr.add_condition("len(op.sources) == 1")
        instr.set_field_flex_param("DST_NS_ID", "op.get_operand_location(op.dests[0].name)")
        instr.set_field_flex_param("DST_INDEX_ID", f"op.dests[0].get_mem_index(op.get_operand_location(op.dests[0].name))")
        instr.set_field_flex_param("SRC1_NS_ID", "op.get_operand_location(op.sources[0].name)")
        instr.set_field_flex_param("SRC1_INDEX_ID", rd_op_str.format(IDX=0))
        instr.set_field_flex_param("SRC2_NS_ID", "'IMM'")
        instr.set_field_value("SRC2_INDEX_ID", 0)
        instructions.append(instr)
    else:
        instructions += add_pow_compute(hag, rd_op_str)


    instructions += alu_noop(op_name, hag)
    instr = hag.get_primitive_template("SYNC_INST")
    all_obuf_ops = f"([o for o in cdlt.get_ops_by_type('compute') if 'OBUF' in o.source_locations])"
    # instr.add_condition("any([op.get_operand_location(o.name) == 'OBUF' for o in op.operands])")
    instr.add_condition(f"len({all_obuf_ops}) > 0 and op == {all_obuf_ops}[-1]")
    instr.set_field_by_name("COMPUTE_TARGET", "SIMD")
    instr.set_field_by_name("START_END", "END")
    instr.set_field_by_name("EXEC_BUF", "BUF")
    instr.set_field_flex_param("GROUP_NUM", "(cdlt.instance_id - 1) % 64")
    instr.set_field_flex_param("NUM_INSTR", "0")
    instructions.append(instr)

    if hag.meta_cfg['ADDR_GEN_TEST']:
        instructions += sw_addr_nops_nested(op_name, hag)

    return instructions



def base_sign_ext_gen(op_name, hag: ArchitectureNode):
    instructions = []

    # Index generation
    # ns_idx = f"(loop_op.loop_id % {ALL_LOOP_ID}) + (operand.get_mem_index({OPERAND_LOC}) * {ALL_LOOP_ID}) if op.get_operand_location(operand.name) != 'IMM' else cdlt.temps.index(operand)"

    ns_idx = f"(loop_op.loop_level % {ALL_LOOP_ID}) + (operand.get_mem_index({OPERAND_LOC}) * {ALL_LOOP_ID}) if op.get_operand_location(operand.name) != 'IMM' else cdlt.temps.index(operand)"
    # base_sign_ext = f"(operand.get_mem_offset({OPERAND_LOC})//(operand.dtype.bits()) + 1) if op.get_operand_location(operand.name) " \
    #                 f"!= 'IMM' else cdlt.temps.index(operand)"

    # base_sign_ext = f"(operand.get_mem_offset({OPERAND_LOC})//(hag.get_subgraph_node({OPERAND_LOC}).data_size) + 1) if op.get_operand_location(operand.name) " \
    #                 f"!= 'IMM' else cdlt.temps.index(operand)"
    # base_sign_ext = f"(operand.get_mem_offset({OPERAND_LOC})) if op.get_operand_location(operand.name) " \
    #                 f"!= 'IMM' else cdlt.temps.index(operand)"

    base_sign_ext_low = f"program.extract_bits({BASE_SIGN_EXT}, 16, 0)"
    base_sign_ext_high = f"program.extract_bits({BASE_SIGN_EXT}, 16, 16)"


    bitwidth = f"len(np.binary_repr({BASE_SIGN_EXT})) + int(np.signbit({BASE_SIGN_EXT}))"
    bitwidth_cond = f"{bitwidth} <= 16"
    single_base_ext_conds = LOOP_CONDS + [bitwidth_cond]
    multi_base_ext_conds = LOOP_CONDS + [f"not {bitwidth_cond}"]

    ## First, instructions for sign ext lower than 16 bits
    macro_instr = hag.get_primitive_template("BASE_SIGN_EXT")
    macro_instr.set_print_tabs(ALL_LOOP_ID)
    macro_instr.add_iterable(*OPERAND_ITER)
    macro_instr.add_iterable(*LOOP_ITER)
    macro_instr.add_condition(" and ".join(single_base_ext_conds))
    macro_instr.set_field_flex_param('NS_ID', OPERAND_LOC)
    macro_instr.set_field_flex_param('NS_INDEX_ID', ns_idx)
    macro_instr.set_field_flex_param('IMM', BASE_SIGN_EXT)

    sub_instr = hag.get_primitive_template("STRIDE_SIGN_EXT")
    sub_instr.set_print_tabs(ALL_LOOP_ID)
    sub_instr.add_iterable(*OPERAND_ITER)
    sub_instr.add_iterable(*LOOP_ITER)
    sub_instr.add_condition(" and ".join(single_base_ext_conds))
    sub_instr.set_field_flex_param('NS_ID', OPERAND_LOC)
    sub_instr.set_field_flex_param('NS_INDEX_ID', ns_idx)
    sub_instr.set_field_flex_param('IMM', LOOP_STRIDE)
    macro_instr.add_base_instruction(sub_instr)
    instructions.append(macro_instr)


    ## NExt, instructions for sign ext greater than 16 bits
    macro_instr = hag.get_primitive_template("BASE_LOW")
    macro_instr.set_print_tabs(ALL_LOOP_ID)
    macro_instr.add_iterable(*OPERAND_ITER)
    macro_instr.add_iterable(*LOOP_ITER)
    macro_instr.add_condition(" and ".join(multi_base_ext_conds))
    macro_instr.set_field_flex_param('NS_ID', OPERAND_LOC)
    macro_instr.set_field_flex_param('NS_INDEX_ID', ns_idx)
    macro_instr.set_field_flex_param('IMM', base_sign_ext_low)

    sub_instr = hag.get_primitive_template("BASE_HIGH")
    sub_instr.set_print_tabs(ALL_LOOP_ID)
    sub_instr.add_iterable(*OPERAND_ITER)
    sub_instr.add_iterable(*LOOP_ITER)
    sub_instr.add_condition(" and ".join(multi_base_ext_conds))
    sub_instr.set_field_flex_param('NS_ID', OPERAND_LOC)
    sub_instr.set_field_flex_param('NS_INDEX_ID', ns_idx)
    sub_instr.set_field_flex_param('IMM', base_sign_ext_high)
    macro_instr.add_base_instruction(sub_instr)

    sub_instr = hag.get_primitive_template("STRIDE_SIGN_EXT")
    sub_instr.set_print_tabs(ALL_LOOP_ID)
    sub_instr.add_iterable(*OPERAND_ITER)
    sub_instr.add_iterable(*LOOP_ITER)
    sub_instr.add_condition(" and ".join(multi_base_ext_conds))
    sub_instr.set_field_flex_param('NS_ID', OPERAND_LOC)
    sub_instr.set_field_flex_param('NS_INDEX_ID', ns_idx)
    sub_instr.set_field_flex_param('IMM', LOOP_STRIDE)
    macro_instr.add_base_instruction(sub_instr)
    instructions.append(macro_instr)

    return instructions


def alu_noop(op_name, hag):
    all_loop_id = ALL_LOOP_ID

    instructions = []
    if OP_COMPUTE_CYCLES[op_name] > 0:
        macro_instr = hag.get_primitive_template("BASE_SIGN_EXT")
        macro_instr.set_field_by_name('NS_ID', "VMEM1")
        macro_instr.set_print_tabs("op.loop_level - 1")
        macro_instr.set_field_flex_param('NS_INDEX_ID', "0")
        macro_instr.set_field_value('IMM', 1)
        #

        sub_instr = hag.get_primitive_template("STRIDE_SIGN_EXT")
        sub_instr.set_field_by_name('NS_ID', "VMEM1")
        sub_instr.set_print_tabs("op.loop_level - 1")
        sub_instr.set_field_flex_param('NS_INDEX_ID', "0")
        sub_instr.set_field_flex_param('IMM', f"0")
        macro_instr.add_base_instruction(sub_instr)

        noop_iters = f"{OP_COMPUTE_CYCLES[op_name]}"
        sub_instr = hag.get_primitive_template("SET_ITER")
        sub_instr.set_print_tabs("op.loop_level - 1")
        sub_instr.set_field_flex_param("LOOP_ID", f"op.loop_id % {all_loop_id}")
        sub_instr.set_field_flex_param("NUM_ITER", f"{noop_iters}")
        macro_instr.add_base_instruction(sub_instr)

        sub_instr = hag.get_primitive_template("SET_INDEX")
        sub_instr.set_print_tabs("op.loop_level + 1")
        sub_instr.set_field_by_name("DST_NS_ID", "VMEM1")
        sub_instr.set_field_flex_param("DST_INDEX_ID", "0")
        sub_instr.set_field_by_name("SRC1_NS_ID", "VMEM1")
        sub_instr.set_field_flex_param("SRC1_INDEX_ID", "0")
        sub_instr.set_field_by_name("SRC2_NS_ID", "VMEM1")
        sub_instr.set_field_flex_param("SRC2_INDEX_ID", '0')
        macro_instr.add_base_instruction(sub_instr)
        instructions.append(macro_instr)

        instr = hag.get_primitive_template("SET_INST")
        instr.set_field_flex_param("SINGLE_NESTED", "1")
        instr.set_field_flex_param("NUM_INSTR", "1")
        instructions.append(instr)

        noop_instr = hag.get_primitive_template("NOP")
        noop_instr.set_print_tabs("op.loop_level")
        instructions.append(noop_instr)
    return instructions