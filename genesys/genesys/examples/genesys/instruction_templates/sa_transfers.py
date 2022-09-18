from codelets.adl.graph import ArchitectureNode


def off_chip_transfer(ld_st, buffer_name, hag: ArchitectureNode):
    instructions = []
    ## FIX RULES:
    # LOOP ID >= 14
    # UNIQUE LOOPS PER DATA
    # At least 64 bytes per request
    # LD STORE req size, num bytes
    # FIND MAX REQ SIZE MULTIPLE OF BYTES
    # MAXIMIZE BW WITH CONSTRAINT THAT IC > bandwidth/input dtype width
    # IC*BITWIDTH_INPUT % bandwidth = 0
    # NUM_ITERS - 1
    ## FIXE RULES END
    if hag.meta_cfg['ASIC_CONFIG']:
        max_bits = f"16"
    else:
        max_bits = f"32"

    # TODO: Change to LOW/HIGH request

    ld_st_loop_str = f"hag.util_fns.get_ld_st_loop_id('{buffer_name}', len(op.sizes_for_node('{buffer_name}')) - 1, '{ld_st}')"
    data_width = f"hag.get_subgraph_node('DRAM').width"
    n_banks = f"hag.get_subgraph_node('{buffer_name}').banks"

    if buffer_name != "WBUF":

        ld_st_tabs = f"op.loop_level + len(op.sizes_for_node('{buffer_name}'))"

        loop_iter_str = f"dim_info[1][1] - 1"
        req_size_str = f"op.strides_iters({data_width}, divisor={n_banks}, max_bits={max_bits})[0][-1]"

        if hag.meta_cfg['ASIC_CONFIG']:
            denom_str = f"hag.get_subgraph_edge('DRAM', '{buffer_name}').bandwidth//8"
            stride_size_str = f"(dim_info[1][0]//({denom_str}))"
        else:
            stride_size_str = f"dim_info[1][0]"
        iterable_str = f"enumerate(zip(*op.strides_iters({data_width}, divisor={n_banks}, max_bits={max_bits})))"
        # END CHANGES

        stride_size_low = f"program.extract_bits({stride_size_str}, 16, 0)"
        stride_size_high = f"program.extract_bits({stride_size_str}, 16, 16)"

        loop_id_str = f"hag.util_fns.get_ld_st_loop_id('{buffer_name}', dim_info[0], '{ld_st}')"

        macro_instr = hag.get_primitive_template("SA_LOOP_CFG")
        macro_instr.add_iterable('dim_info', iterable_str)
        macro_instr.set_field_flex_param("LOOP_ID", loop_id_str)
        macro_instr.set_field_flex_param("NUM_ITERATIONS", f"{loop_iter_str}")
        macro_instr.set_print_tabs("op.loop_level + dim_info[0]")

        micro_instr = hag.get_primitive_template("SET_LOOP_STRIDE")
        micro_instr.add_iterable('dim_info', iterable_str)
        micro_instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
        micro_instr.set_field_by_name("ACCESS_TYPE", ld_st)
        micro_instr.set_field_by_name("BUFFER", f"{buffer_name}")
        micro_instr.set_field_flex_param("LOOP_ID", loop_id_str)
        micro_instr.set_field_flex_param("STRIDE", stride_size_low)
        micro_instr.set_print_tabs("op.loop_level + dim_info[0]")
        macro_instr.add_base_instruction(micro_instr)

        micro_instr = hag.get_primitive_template("SET_LOOP_STRIDE")
        micro_instr.add_iterable('dim_info', iterable_str)
        micro_instr.set_field_by_name("LOW_HIGH_BITS", "HIGH")
        micro_instr.set_field_by_name("ACCESS_TYPE", ld_st)
        micro_instr.set_field_by_name("BUFFER", f"{buffer_name}")
        micro_instr.set_field_flex_param("LOOP_ID", loop_id_str)
        micro_instr.set_field_flex_param("STRIDE", stride_size_high)
        micro_instr.set_print_tabs("op.loop_level + dim_info[0]")
        macro_instr.add_base_instruction(micro_instr)
        instructions.append(macro_instr)
    else:
        ld_st_tabs = f"op.loop_level + 1"
        req_size_str = f"op.strides_iters({data_width}, divisor={n_banks}, max_bits={max_bits}, contiguous=True)[0][-1]"
        ld_str_size = f"op.strides_iters({data_width}, divisor={n_banks}, max_bits={max_bits}, contiguous=True)[0][-1]"

        if hag.meta_cfg['ASIC_CONFIG']:
            denom_str = f"hag.get_subgraph_edge('DRAM', '{buffer_name}').bandwidth//8"
            stride_size_str = f"({ld_str_size})//{denom_str}"
        else:
            stride_size_str = f"({ld_str_size})"

        stride_size_low = f"program.extract_bits({stride_size_str}, 16, 0)"
        stride_size_high = f"program.extract_bits({stride_size_str}, 16, 16)"
        instr = hag.get_primitive_template("SA_LOOP_CFG")
        instr.set_field_flex_param("LOOP_ID", ld_st_loop_str)
        instr.set_field_flex_param("NUM_ITERATIONS", f"op.strides_iters({data_width}, divisor={n_banks}, "
                                                     f"max_bits={max_bits},"
                                                     f"contiguous=True)[1][-1] - 1")
        instructions.append(instr)

        instr = hag.get_primitive_template("SET_LOOP_STRIDE")
        instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
        instr.set_field_by_name("ACCESS_TYPE", ld_st)
        instr.set_field_by_name("BUFFER", f"{buffer_name}")
        instr.set_field_flex_param("LOOP_ID", ld_st_loop_str)
        instr.set_field_flex_param("STRIDE", stride_size_low)
        instructions.append(instr)

        instr = hag.get_primitive_template("SET_LOOP_STRIDE")
        instr.set_field_by_name("LOW_HIGH_BITS", "HIGH")
        instr.set_field_by_name("ACCESS_TYPE", ld_st)
        instr.set_field_by_name("BUFFER", f"{buffer_name}")
        instr.set_field_flex_param("LOOP_ID", ld_st_loop_str)
        instr.set_field_flex_param("STRIDE", stride_size_high)
        instructions.append(instr)

    ####
    # ITERS = tile_size / request_size / num_banks
    instr = hag.get_primitive_template("LD_ST")
    instr.set_field_by_name("ACCESS_TYPE", ld_st)
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_flex_param("LOOP_ID", ld_st_loop_str)

    # TEMP FIX
    instr.set_field_flex_param("REQUEST_SIZE", f"{req_size_str}//{n_banks}")
    # instr.set_field_flex_param("REQUEST_SIZE", f"0")
    instr.set_print_tabs(ld_st_tabs)


    instr.set_print_tabs(ld_st_tabs)
    instructions.append(instr)
    return instructions