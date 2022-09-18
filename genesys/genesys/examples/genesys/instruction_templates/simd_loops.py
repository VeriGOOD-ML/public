from codelets.adl.graph import ArchitectureNode



def outer_simd_loops(hag: ArchitectureNode):
    instructions = []
    ld_simd_compute = f"(cdlt.loop_compute_op(op, src_op=operand))"
    st_simd_compute = f"(cdlt.loop_compute_op(op, dst_op=operand))"
    # program, cdlt, op, hag
    ld_stride_str = f"operand.get_offset(cdlt, op.loop_id, hag, {ld_simd_compute}.op_str, 'DRAM', write=False, outer_loop=True)*operand.dtype.bytes()"
    st_stride_str = f"operand.get_offset(cdlt, op.loop_id, hag, {st_simd_compute}.op_str, 'DRAM', write=True, outer_loop=True)*operand.dtype.bytes()"
    ld_operand_names = f"list(set([o.name for o in cdlt.filtered_read_operands('SIMD')]))"
    st_operand_names = f"list(set([o.name for o in cdlt.filtered_write_operands('SIMD')]))"
    ld_operand_str = f"[cdlt.get_operand(n) for n in {ld_operand_names}]"
    st_operand_str = f"[cdlt.get_operand(n) for n in {st_operand_names}]"
    simd_target = f'cdlt.is_loop_node_target(op, "SIMD")'
    not_dir_loop_dep = f'not cdlt.is_direct_loop_dep(op, "SIMD")'
    off_chip_operand = f'"DRAM" in operand.data_path'
    ld_check = f"(operand.has_transfer(['DRAM', operand.get_ld_storage_location(cdlt, 1), 'SIMD']))"
    st_check = f"(operand.has_transfer(['SIMD', operand.get_ld_storage_location(cdlt, 1), 'DRAM']))"
    ld_st_cond = f"(hag.is_adjacent(operand.get_ld_storage_location(cdlt, 1), 'SIMD'))"
    loop_conds = [simd_target, not_dir_loop_dep, off_chip_operand]
    ld_conds = loop_conds + [f"operand not in cdlt.outputs"]

    block_iter = ('operand', f'{ld_operand_str}')
    block_cond = " and ".join(ld_conds)

    macro_instr = hag.get_primitive_template("LD_CONFIG_BASE_LOOP_ITER")
    macro_instr.add_iterable(*block_iter)
    macro_instr.add_condition(f"{block_cond} and {ld_st_cond} and {ld_check}")
    macro_instr.set_field_flex_param("NS_ID", "operand.get_ld_storage_location(cdlt, 1)")
    macro_instr.set_field_flex_param("LOOP_INDEX_ID", f"op.loop_id")
    macro_instr.set_field_flex_param("NUM_ITERS", f"op.iter_count - 1")

    micro_instr = hag.get_primitive_template("LD_CONFIG_BASE_LOOP_STRIDE")
    micro_instr.add_iterable(*block_iter)
    macro_instr.add_condition(f"{block_cond} and {ld_st_cond} and {ld_check}")
    micro_instr.set_field_by_name("LSB_MSB", "LSB")
    micro_instr.set_field_flex_param("NS_ID", "operand.get_ld_storage_location(cdlt, 1)")
    micro_instr.set_field_flex_param("LOOP_INDEX_ID", f"op.loop_id")
    micro_instr.set_field_flex_param("STRIDE",
                                     f"program.extract_bits({ld_stride_str}, 16, 0)"
                                     )
    macro_instr.add_base_instruction(micro_instr)

    micro_instr = hag.get_primitive_template("LD_CONFIG_BASE_LOOP_STRIDE")
    micro_instr.add_iterable(*block_iter)
    macro_instr.add_condition(f"{block_cond} and {ld_st_cond} and {ld_check}")
    micro_instr.set_field_by_name("LSB_MSB", "MSB")
    micro_instr.set_field_flex_param("NS_ID", "operand.get_ld_storage_location(cdlt, 1)")
    micro_instr.set_field_flex_param("LOOP_INDEX_ID", f"op.loop_id")
    micro_instr.set_field_flex_param("STRIDE",
                                     f"program.extract_bits({ld_stride_str}, 16, 16)"
                                     )
    macro_instr.add_base_instruction(micro_instr)
    instructions.append(macro_instr)

    macro_instr = hag.get_primitive_template("ST_CONFIG_BASE_LOOP_ITER")
    loop_cond_str = " and ".join(loop_conds)

    macro_instr.add_iterable('operand', f'{st_operand_str}')
    macro_instr.add_condition(f"{loop_cond_str} and {ld_st_cond} and {st_check}")
    macro_instr.set_field_flex_param("NS_ID", "operand.get_ld_storage_location(cdlt, 1)")
    macro_instr.set_field_flex_param("LOOP_INDEX_ID", f"op.loop_id")
    macro_instr.set_field_flex_param("NUM_ITERS", f"op.iter_count - 1")

    micro_instr = hag.get_primitive_template("ST_CONFIG_BASE_LOOP_STRIDE")
    micro_instr.add_iterable('operand', f'{st_operand_str}')
    micro_instr.add_condition(f"{loop_cond_str} and {ld_st_cond} and {st_check}")
    micro_instr.set_field_by_name("LSB_MSB", "LSB")
    micro_instr.set_field_flex_param("NS_ID", "operand.get_ld_storage_location(cdlt, 1)")
    micro_instr.set_field_flex_param("LOOP_INDEX_ID", f"op.loop_id")
    micro_instr.set_field_flex_param("STRIDE", f"program.extract_bits({st_stride_str}, 16, 0)")
    macro_instr.add_base_instruction(micro_instr)

    micro_instr = hag.get_primitive_template("ST_CONFIG_BASE_LOOP_STRIDE")
    micro_instr.add_iterable('operand', f'{st_operand_str}')
    micro_instr.add_condition(f"{loop_cond_str} and {ld_st_cond} and {st_check}")
    micro_instr.set_field_by_name("LSB_MSB", "MSB")
    micro_instr.set_field_flex_param("NS_ID", "operand.get_ld_storage_location(cdlt, 1)")
    micro_instr.set_field_flex_param("LOOP_INDEX_ID", f"op.loop_id")
    micro_instr.set_field_flex_param("STRIDE", f"program.extract_bits({st_stride_str}, 16, 16)")
    macro_instr.add_base_instruction(micro_instr)
    instructions.append(macro_instr)

    return instructions


def inner_simd_loops(hag: ArchitectureNode):
    instructions = []
    is_direct_loop_dep = f"cdlt.is_direct_loop_dep(loop_op, 'SIMD')"

    return instructions