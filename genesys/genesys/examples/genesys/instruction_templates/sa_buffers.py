from codelets.adl.graph import ComputeNode
from . import GENERATING_BENCH, BENCH_BASE_ADDR

BASE_ADDR_STR = "program.extract_bits(relocation_table.get_base_by_name({OPERAND_NAME}), {NUM_BITS}, {POS})"


def ibuf_start_template(hag: ComputeNode):
    instructions = []
    if GENERATING_BENCH:
        instr = hag.get_primitive_template("SET_BASE_ADDR")
        instr.set_field_by_name("LOW_HIGH_ADDR", "LOW")
        instr.set_field_by_name("MEM_TYPE", "BUFFER")
        instr.set_field_by_name("BUFFER", "IBUF")
        # TODO: Fix relocation table imem value
        instr.set_field_flex_param("BASE_ADDR",
                                 f"program.extract_bits({BENCH_BASE_ADDR['IBUF']},"
                                 " 16, 0)",
                                   lazy_eval=True)
        instructions.append(instr)

        instr = hag.get_primitive_template("SET_BASE_ADDR")
        instr.set_field_by_name("LOW_HIGH_ADDR", "HIGH")
        instr.set_field_by_name("MEM_TYPE", "BUFFER")
        instr.set_field_by_name("BUFFER", "IBUF")
        instr.set_field_flex_param("BASE_ADDR",
                                 f"program.extract_bits({BENCH_BASE_ADDR['IBUF']},"
                                 " 16, 16)",
                                   lazy_eval=True)
        instructions.append(instr)
    else:
        instr = hag.get_primitive_template("SET_BASE_ADDR")
        instr.set_field_by_name("LOW_HIGH_ADDR", "LOW")
        instr.set_field_by_name("MEM_TYPE", "BUFFER")
        instr.set_field_by_name("BUFFER", "IBUF")
        instr.set_field_flex_param("BASE_ADDR",
                                   BASE_ADDR_STR.format(OPERAND_NAME="cdlt.inputs[0].node_name", NUM_BITS="16",
                                                        POS="0"),
                                   lazy_eval=True)
        instructions.append(instr)

        instr = hag.get_primitive_template("SET_BASE_ADDR")
        instr.set_field_by_name("LOW_HIGH_ADDR", "HIGH")
        instr.set_field_by_name("MEM_TYPE", "BUFFER")
        instr.set_field_by_name("BUFFER", "IBUF")
        instr.set_field_flex_param("BASE_ADDR",
                                   BASE_ADDR_STR.format(OPERAND_NAME="cdlt.inputs[0].node_name", NUM_BITS="16",
                                                        POS="16"),
                                   lazy_eval=True)
        instructions.append(instr)
    return instructions


def bbuf_start_template(hag: ComputeNode):
    instructions = []
    if GENERATING_BENCH:
        instr = hag.get_primitive_template("SET_BASE_ADDR")
        instr.set_field_by_name("LOW_HIGH_ADDR", "LOW")
        instr.set_field_by_name("MEM_TYPE", "BUFFER")
        instr.set_field_by_name("BUFFER", "BBUF")
        # TODO: Fix relocation table imem value
        instr.set_field_flex_param("BASE_ADDR",
                                 f"program.extract_bits({BENCH_BASE_ADDR['BBUF']},"
                                 " 16, 0)",
                                   lazy_eval=True)
        instructions.append(instr)

        instr = hag.get_primitive_template("SET_BASE_ADDR")
        instr.set_field_by_name("LOW_HIGH_ADDR", "HIGH")
        instr.set_field_by_name("MEM_TYPE", "BUFFER")
        instr.set_field_by_name("BUFFER", "BBUF")
        instr.set_field_flex_param("BASE_ADDR",
                                 f"program.extract_bits({BENCH_BASE_ADDR['BBUF']},"
                                 " 16, 16)",
                                   lazy_eval=True)
        instructions.append(instr)
    else:
        instr = hag.get_primitive_template("SET_BASE_ADDR")
        instr.set_field_by_name("LOW_HIGH_ADDR", "LOW")
        instr.set_field_by_name("MEM_TYPE", "BUFFER")
        instr.set_field_by_name("BUFFER", "BBUF")
        instr.set_field_flex_param("BASE_ADDR",
                                   BASE_ADDR_STR.format(OPERAND_NAME="cdlt.inputs[2].node_name", NUM_BITS="16",
                                                        POS="0"),
                                   lazy_eval=True)
        instructions.append(instr)

        instr = hag.get_primitive_template("SET_BASE_ADDR")
        instr.set_field_by_name("LOW_HIGH_ADDR", "HIGH")
        instr.set_field_by_name("MEM_TYPE", "BUFFER")
        instr.set_field_by_name("BUFFER", "BBUF")
        instr.set_field_flex_param("BASE_ADDR",
                                   BASE_ADDR_STR.format(OPERAND_NAME="cdlt.inputs[2].node_name", NUM_BITS="16",
                                                        POS="16"),
                                   lazy_eval=True)
        instructions.append(instr)
    return instructions


def obuf_start_template(hag: ComputeNode):
    instructions = []
    if GENERATING_BENCH:
        instr = hag.get_primitive_template("SET_BASE_ADDR")
        instr.set_field_by_name("LOW_HIGH_ADDR", "LOW")
        instr.set_field_by_name("MEM_TYPE", "BUFFER")
        instr.set_field_by_name("BUFFER", "OBUF")
        # TODO: Fix relocation table imem value
        instr.set_field_flex_param("BASE_ADDR",
                                 f"program.extract_bits({BENCH_BASE_ADDR['OBUF']},"
                                 " 16, 0)",
                                   lazy_eval=True)
        instructions.append(instr)

        instr = hag.get_primitive_template("SET_BASE_ADDR")
        instr.set_field_by_name("LOW_HIGH_ADDR", "HIGH")
        instr.set_field_by_name("MEM_TYPE", "BUFFER")
        instr.set_field_by_name("BUFFER", "OBUF")
        instr.set_field_flex_param("BASE_ADDR",
                                 f"program.extract_bits({BENCH_BASE_ADDR['OBUF']},"
                                 " 16, 16)",
                                   lazy_eval=True)
        instructions.append(instr)
    else:
        instr = hag.get_primitive_template("SET_BASE_ADDR")
        instr.set_field_by_name("LOW_HIGH_ADDR", "LOW")
        instr.set_field_by_name("MEM_TYPE", "BUFFER")
        instr.set_field_by_name("BUFFER", "OBUF")
        instr.set_field_flex_param("BASE_ADDR",
                                   BASE_ADDR_STR.format(OPERAND_NAME="cdlt.outputs[0].node_name", NUM_BITS="16", POS="0"),
                                   lazy_eval=True)

        instructions.append(instr)

        instr = hag.get_primitive_template("SET_BASE_ADDR")
        instr.set_field_by_name("LOW_HIGH_ADDR", "HIGH")
        instr.set_field_by_name("MEM_TYPE", "BUFFER")
        instr.set_field_by_name("BUFFER", "OBUF")
        instr.set_field_flex_param("BASE_ADDR",
                                   BASE_ADDR_STR.format(OPERAND_NAME="cdlt.outputs[0].node_name", NUM_BITS="16", POS="16"),
                                   lazy_eval=True)
        instructions.append(instr)
    return instructions


def wbuf_start_template(hag: ComputeNode):
    instructions = []
    if GENERATING_BENCH:
        instr = hag.get_primitive_template("SET_BASE_ADDR")
        instr.set_field_by_name("LOW_HIGH_ADDR", "LOW")
        instr.set_field_by_name("MEM_TYPE", "BUFFER")
        instr.set_field_by_name("BUFFER", "WBUF")
        # TODO: Fix relocation table imem value
        instr.set_field_flex_param("BASE_ADDR",
                                 f"program.extract_bits({BENCH_BASE_ADDR['WBUF']},"
                                 " 16, 0)",
                                   lazy_eval=True)
        instructions.append(instr)

        instr = hag.get_primitive_template("SET_BASE_ADDR")
        instr.set_field_by_name("LOW_HIGH_ADDR", "HIGH")
        instr.set_field_by_name("MEM_TYPE", "BUFFER")
        instr.set_field_by_name("BUFFER", "WBUF")
        instr.set_field_flex_param("BASE_ADDR",
                                 f"program.extract_bits({BENCH_BASE_ADDR['WBUF']},"
                                 " 16, 16)",
                                   lazy_eval=True)
        instructions.append(instr)

    else:
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