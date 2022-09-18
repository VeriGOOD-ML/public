from codelets.adl.graph import ComputeNode

BASE_ADDR_STR = "program.extract_bits(relocation_table.get_base_by_name({OPERAND_NAME}), {NUM_BITS}, {POS})"

def imm_start_template(hag: ComputeNode):
    instructions = []
    # program.extract_bits({stride_size_str}, 16, 16)
    imm_val = "op.get_config_param_value('immediate_value')"
    bitwidth = f"len(np.binary_repr({imm_val})) + int(np.signbit({imm_val}))"
    bitwidth_cond = f"{bitwidth} <= 16"

    instr = hag.get_primitive_template("IMM_SIGN_EXT")
    instr.set_field_by_name("NS_ID", "IMM")
    instr.add_condition(bitwidth_cond)
    instr.set_field_flex_param("NS_INDEX_ID", f"op.get_config_param_value('index')")
    instr.set_field_flex_param("IMM", imm_val)
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_IMM_LOW")
    instr.add_condition(f"not ({bitwidth_cond})")
    instr.set_field_by_name('NS_ID', "IMM")
    instr.set_field_flex_param("NS_INDEX_ID", f"op.get_config_param_value('index')")
    instr.set_field_flex_param("IMM", f"program.extract_bits({imm_val}, 16, 0)")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_IMM_HIGH")
    instr.add_condition(f"not ({bitwidth_cond})")
    instr.set_field_by_name('NS_ID', "IMM")
    instr.set_field_flex_param("NS_INDEX_ID", f"op.get_config_param_value('index')")
    instr.set_field_flex_param("IMM", f"program.extract_bits({imm_val}, 16, 16)")
    instructions.append(instr)

    return instructions


def imm_end_template(hag: ComputeNode):
    instructions = []
    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "LOW")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "WBUF")
    instr.set_field_flex_param("BASE_ADDR",
                               BASE_ADDR_STR.format(OPERAND_NAME="cdlt.inputs[1].node_name", NUM_BITS="16",
                                                    POS="0"),
                               lazy_eval=True)
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "HIGH")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "WBUF")
    instr.set_field_flex_param("BASE_ADDR",
                               BASE_ADDR_STR.format(OPERAND_NAME="cdlt.inputs[1].node_name", NUM_BITS="16",
                                                    POS="16"),
                               lazy_eval=True)
    instructions.append(instr)
    return instructions