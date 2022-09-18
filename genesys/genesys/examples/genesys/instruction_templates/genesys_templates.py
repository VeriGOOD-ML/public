from examples.genesys.genesys_instructions import DTYPE_CAST_NAMES, \
    CMP_OP_NAMES, CALC_OP_NAMES, ALU_OP_NAMES
from examples.genesys.instruction_templates.loops import loop_template, loop_end_template
from examples.genesys.instruction_templates.sa_buffers import ibuf_start_template, bbuf_start_template, \
    obuf_start_template, wbuf_start_template
from examples.genesys.instruction_templates.sa_compute import sa_mvmul_template
from examples.genesys.instruction_templates.sa_transfers import off_chip_transfer
from examples.genesys.instruction_templates.simd_compute import simd_alu_template
from examples.genesys.instruction_templates.simd_memory import imm_start_template, imm_end_template
from examples.genesys.instruction_templates.simd_transfers import off_chip_transfer_simd
from examples.genesys.instruction_templates.start_end import codelet_start, program_end, program_start, codelet_end, \
    simd_start_template, simd_end_template, sa_start_template, sa_end_template

from functools import partial
BUFFER_ID_MAP = {'LD': {'IBUF': 0, 'WBUF': 1, 'OBUF': 2, 'BBUF': 3},
                 'ST': {'IBUF': 4, 'WBUF': 5, 'OBUF': 6, 'BBUF': 7},
                 }


SIMD_OP_NAMES = ALU_OP_NAMES + CALC_OP_NAMES + CMP_OP_NAMES + DTYPE_CAST_NAMES + ["POW", "TRANSPOSE"]


# Only need group id for INST_GROUP end, and need 1 for last group
# Only need 1 base addr instruction per memory
# Only 2 lds and 2 st allowed per instr group
# Only 16 groups supported
# Num iterations are minus 1
# Fix strides for outer loops
# add multiple loops
# tile loops for tile loops --> req size divided by lanes
#


GENESYS_TEMPLATES = {
    "program": {
        "start": program_start,
        "end": program_end,
    },
    "codelet": {
        "start": codelet_start,
        "end": codelet_end,
    },
    "config": {
        "systolic_array": {
            "start": sa_start_template,
            "end": sa_end_template
        },
        "WBUF": {
            "start": wbuf_start_template,
            "end": lambda x: []
        },
        "IBUF": {
            "start": ibuf_start_template,
            "end": lambda x: []
        },
        "BBUF": {
            "start": bbuf_start_template,
            "end": lambda x: []
        },
        "OBUF": {
            "start": obuf_start_template,
            "end": lambda x: []
        },
        "SIMD": {
            "start": simd_start_template,
            "end": simd_end_template
        },
        "IMM": {
            "start": imm_start_template,
            "end": imm_end_template
        },
    },
    "transfer": {
        ("DRAM", "WBUF"): partial(off_chip_transfer, "LD", "WBUF"),
        ("DRAM", "IBUF"): partial(off_chip_transfer, "LD", "IBUF"),
        ("DRAM", "BBUF"): partial(off_chip_transfer, "LD", "BBUF"),
        ("DRAM", "OBUF"): partial(off_chip_transfer, "LD", "OBUF"),
        ("OBUF", "DRAM"): partial(off_chip_transfer, "ST", "OBUF"),
        ("IBUF", "DRAM"): partial(off_chip_transfer, "ST", "IBUF"),
        ("WBUF", "DRAM"): partial(off_chip_transfer, "ST", "WBUF"),
        ("BBUF", "DRAM"): partial(off_chip_transfer, "ST", "BBUF"),
        ("DRAM", "VMEM1"): partial(off_chip_transfer_simd, "LD", "VMEM1"),
        ("DRAM", "VMEM2"): partial(off_chip_transfer_simd, "LD", "VMEM2"),
        ("VMEM1", "DRAM"): partial(off_chip_transfer_simd, "ST", "VMEM1"),
        ("VMEM2", "DRAM"): partial(off_chip_transfer_simd, "ST", "VMEM2"),
    },
    "loop": loop_template,
    "loop_end": loop_end_template,
    "compute": {
        ("pe_array", "MVMUL"): sa_mvmul_template,
        **{("SIMD", op_name): partial(simd_alu_template, op_name) for op_name in SIMD_OP_NAMES},
    }
}