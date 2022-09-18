from codelets.adl.graph import ArchitectureNode, ComputeNode
from . import GENERATING_BENCH, BENCH_BASE_ADDR
# SIMD_BASE_ADDR = {"LD_VMEM1": 0, "LD_VMEM2": 1024 << 16, "ST_VMEM1": 1024 << 17, "ST_VMEM2": 1024 << 8}
# SIMD_BASE_ADDR = {"LD_VMEM1": 0, "LD_VMEM2": 1024 << 11, "ST_VMEM1": 1024 << 12, "ST_VMEM2": 1024 << 13}
SIMD_BASE_ADDR = {"LD_VMEM1": 0, "LD_VMEM2": 1024 << 10, "ST_VMEM1": 1024 << 11, "ST_VMEM2": 1024 << 12}
SIMD_BASE_ADDR_STR = str(SIMD_BASE_ADDR)
BASE_ADDR_STR_SIMD = f"program.extract_bits({SIMD_BASE_ADDR}[" + "'{LS}_' + relocation_table.get_namespace_by_name({OPERAND_NAME})], {NUM_BITS}, {POS})"


def codelet_start(hag: ArchitectureNode):
    instructions = []
    return instructions


def program_end(hag: ArchitectureNode):
    instructions = []
    return instructions


def program_start(hag: ArchitectureNode):
    instructions = []
    return instructions


def codelet_end(hag: ArchitectureNode):
    instructions = []
    instr = hag.get_primitive_template("BLOCK_END")
    # # TODO: Make sure this is evaluated after having gone through all codelets
    instr.set_field_flex_param("IS_END", "int(program.codelets[-1].instance_id == cdlt.instance_id)")
    instructions.append(instr)
    return instructions

def simd_start_template(hag: ComputeNode):

    instructions = []
    instr = hag.get_primitive_template("SYNC_INST")
    instr.set_field_by_name("COMPUTE_TARGET", "SIMD")
    instr.set_field_by_name("START_END", "START")
    instr.set_field_by_name("EXEC_BUF", "EXEC")
    instr.set_field_flex_param("GROUP_NUM", "(cdlt.instance_id - 1) % 64")
    # instr.set_field_flex_param("NUM_INSTR", "cdlt.num_instr", lazy_eval=True)
    instr.set_field_flex_param("NUM_INSTR", "cdlt.num_instr_by_group('SIMD')", lazy_eval=True)
    instructions.append(instr)

    # TODO: THis is a hotfix. need to more intelligently set the config for this later
    instr = hag.get_primitive_template("DTYPE_CFG")
    instr.set_field_flex_param("DTYPE", "str(cdlt.outputs[0].dtype.bits()) + cdlt.outputs[0].dtype.type")
    instr.set_field_flex_param("DST_BITS", "cdlt.outputs[0].dtype.exp")
    instr.set_field_flex_param("SRC1_BITS", "cdlt.outputs[0].dtype.exp")
    instr.set_field_flex_param("SRC2_BITS", "cdlt.outputs[0].dtype.exp")
    instructions.append(instr)

    block_iter = ('operand', f'cdlt.operands')
    block_cond = f'"SIMD" in operand.data_path and operand.data_path[0] == "DRAM" and operand not in cdlt.outputs'
    macro_instr = hag.get_primitive_template("LD_CONFIG_BASE_ADDR")
    macro_instr.add_iterable(*block_iter)
    macro_instr.add_condition(block_cond)
    macro_instr.set_field_by_name("LSB_MSB", "LSB")
    macro_instr.set_field_flex_param("NS_ID", "operand.get_ld_storage_location(cdlt, 1)")
    macro_instr.set_field_flex_param("LOOP_INDEX_ID", f"0")

    p1 = f"program.extract_bits(({SIMD_BASE_ADDR})["
    BASE_ADDR_STR_SIMD1 = "{EXTRACT}'{LS}_' + (operand.get_ld_storage_location(cdlt, 1))], {NUM_BITS}, {POS})"

    macro_instr.set_field_flex_param("BASE_ADDR",
                                     BASE_ADDR_STR_SIMD1.format(EXTRACT=p1, LS="LD", NUM_BITS="16",
                                                          POS="0"),
                                     )

    #
    micro_instr = hag.get_primitive_template("LD_CONFIG_BASE_ADDR")
    micro_instr.add_iterable(*block_iter)
    micro_instr.add_condition(block_cond)
    micro_instr.set_field_by_name("LSB_MSB", "MSB")
    micro_instr.set_field_flex_param("NS_ID", "operand.get_ld_storage_location(cdlt, 1)")
    micro_instr.set_field_flex_param("LOOP_INDEX_ID", f"0")
    micro_instr.set_field_flex_param("BASE_ADDR",
                                     BASE_ADDR_STR_SIMD1.format(EXTRACT=p1, LS="LD", NUM_BITS="16",
                                                          POS="16"),
                                     )
    macro_instr.add_base_instruction(micro_instr)
    instructions.append(macro_instr)

    macro_instr = hag.get_primitive_template("ST_CONFIG_BASE_ADDR")
    macro_instr.add_iterable('operand', f'cdlt.operands')
    macro_instr.add_condition(f'"SIMD" in operand.data_path and operand.data_path[-1] == "DRAM"')
    macro_instr.set_field_by_name("LSB_MSB", "LSB")
    macro_instr.set_field_flex_param("NS_ID", "operand.get_ld_storage_location(cdlt, 1)")
    macro_instr.set_field_flex_param("LOOP_INDEX_ID", f"0")
    # macro_instr.set_field_flex_param("BASE_ADDR",
    #                                  BASE_ADDR_STR.format(OPERAND_NAME="operand.node_name", NUM_BITS="16",
    #                                                       POS="0"),
    #                                  lazy_eval=True
    #                                  )
    macro_instr.set_field_flex_param("BASE_ADDR",
                                     BASE_ADDR_STR_SIMD1.format(EXTRACT=p1, LS="ST", NUM_BITS="16",
                                                          POS="0"),
                                     )
    #
    micro_instr = hag.get_primitive_template("ST_CONFIG_BASE_ADDR")
    micro_instr.add_iterable('operand', f'cdlt.operands')
    micro_instr.add_condition(f'"SIMD" in operand.data_path and operand.data_path[-1] == "DRAM"')
    micro_instr.set_field_by_name("LSB_MSB", "MSB")
    micro_instr.set_field_flex_param("NS_ID", "operand.get_ld_storage_location(cdlt, 1)")
    micro_instr.set_field_flex_param("LOOP_INDEX_ID", f"0")
    # micro_instr.set_field_flex_param("BASE_ADDR",
    #                                  BASE_ADDR_STR.format(OPERAND_NAME="operand.node_name", NUM_BITS="16",
    #                                                       POS="16"),
    #                                  lazy_eval=True)
    micro_instr.set_field_flex_param("BASE_ADDR",
                                     BASE_ADDR_STR_SIMD1.format(EXTRACT=p1, LS="ST", NUM_BITS="16",
                                                          POS="16"),
                                     )
    macro_instr.add_base_instruction(micro_instr)

    instructions.append(macro_instr)

    return instructions


def simd_end_template(hag: ComputeNode):
    #TODO: Add conditional block end instruction
    instructions = []

    instr = hag.get_primitive_template("SYNC_INST")
    instr.set_field_by_name("COMPUTE_TARGET", "SIMD")
    instr.set_field_by_name("START_END", "END")
    instr.set_field_by_name("EXEC_BUF", "EXEC")
    instr.set_field_flex_param("GROUP_NUM", "(cdlt.instance_id - 1) % 64")
    instr.set_field_flex_param("NUM_INSTR", "0")
    instructions.append(instr)
    return instructions


def sa_start_template(hag: ComputeNode):

    instructions = []
    instr = hag.get_primitive_template("SYNC_INST")
    instr.set_field_by_name("COMPUTE_TARGET", "SYSTOLIC_ARRAY")
    instr.set_field_by_name("START_END", "START")
    instr.set_field_by_name("EXEC_BUF", "EXEC")
    instr.set_field_flex_param("GROUP_NUM", "(cdlt.instance_id - 1) % 64")
    # Figure out what this is
    # instr.set_field_flex_param("NUM_INSTR", "cdlt.num_instr", lazy_eval=True)
    instr.set_field_flex_param("NUM_INSTR", "cdlt.num_instr_by_group('systolic_array')", lazy_eval=True)
    instructions.append(instr)

    if GENERATING_BENCH:
        instr = hag.get_primitive_template("SET_BASE_ADDR")
        instr.set_field_by_name("LOW_HIGH_ADDR", "LOW")
        instr.set_field_by_name("MEM_TYPE", "IMEM")
        instr.set_field_by_name("BUFFER", "IBUF")
        # TODO: Fix relocation table imem value
        instr.set_field_flex_param("BASE_ADDR",
                                 f"program.extract_bits({BENCH_BASE_ADDR['INSTR']},"
                                 " 16, 0)",
                                   lazy_eval=True)
        instructions.append(instr)

        instr = hag.get_primitive_template("SET_BASE_ADDR")
        instr.set_field_by_name("LOW_HIGH_ADDR", "HIGH")
        instr.set_field_by_name("MEM_TYPE", "IMEM")
        instr.set_field_by_name("BUFFER", "IBUF")
        instr.set_field_flex_param("BASE_ADDR",
                                 f"program.extract_bits({BENCH_BASE_ADDR['INSTR']},"
                                 " 16, 16)",
                                   lazy_eval=True)
        instructions.append(instr)
    else:
        instr = hag.get_primitive_template("SET_BASE_ADDR")
        instr.set_field_by_name("LOW_HIGH_ADDR", "LOW")
        instr.set_field_by_name("MEM_TYPE", "IMEM")
        instr.set_field_by_name("BUFFER", "IBUF")
        # TODO: Fix relocation table imem value
        instr.set_field_flex_param("BASE_ADDR",
                                   "program.extract_bits(relocation_table.get_relocation_base('INSTR_MEM', cdlt.cdlt_uid),"
                                   " 16, 0)",
                                   lazy_eval=True)
        instructions.append(instr)

        instr = hag.get_primitive_template("SET_BASE_ADDR")
        instr.set_field_by_name("LOW_HIGH_ADDR", "HIGH")
        instr.set_field_by_name("MEM_TYPE", "IMEM")
        instr.set_field_by_name("BUFFER", "IBUF")
        instr.set_field_flex_param("BASE_ADDR",
                                   "program.extract_bits(relocation_table.get_relocation_base('INSTR_MEM', cdlt.cdlt_uid),"
                                   " 16, 16)",
                                   lazy_eval=True)
        instructions.append(instr)

    instr = hag.get_primitive_template("LD_ST")
    instr.add_condition("cdlt.instance_id < len(program.codelets)")
    instr.set_field_by_name("ACCESS_TYPE", "LD")
    instr.set_field_by_name("MEM_TYPE", "IMEM")
    instr.set_field_by_name("BUFFER", "IBUF")
    instr.set_field_value("LOOP_ID", 0)
    instr.set_field_flex_param("REQUEST_SIZE", "program.codelets[cdlt.instance_id].num_instr", lazy_eval=True)
    instructions.append(instr)

    return instructions


def sa_end_template(hag: ComputeNode):
    #TODO: Add conditional block end instruction
    instructions = []
    instr = hag.get_primitive_template("SYNC_INST")
    instr.set_field_by_name("COMPUTE_TARGET", "SYSTOLIC_ARRAY")
    instr.set_field_by_name("START_END", "END")
    instr.set_field_by_name("EXEC_BUF", "EXEC")
    instr.set_field_flex_param("GROUP_NUM", "(cdlt.instance_id - 1) % 64")

    instr.set_field_value("NUM_INSTR", 0)
    instructions.append(instr)
    return instructions


