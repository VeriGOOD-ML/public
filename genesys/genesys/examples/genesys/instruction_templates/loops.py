from codelets.adl.graph import ArchitectureNode
from examples.genesys.instruction_templates.sa_loops import inner_sa_loops, outer_sa_loops
from examples.genesys.instruction_templates.simd_loops import outer_simd_loops, inner_simd_loops

LOOPS_PER_LEVEL = 7


def loop_template(hag: ArchitectureNode):
    instructions = []
    # SYSTOLIC ARRAY LOOP INSTR

    # OUTER LOOP
    instructions += outer_sa_loops(hag)

    # INNER LOOP
    instructions += inner_sa_loops(hag)


    # SIMD ARRAY LOOP INSTR

    # OUTER LOOP
    instructions += outer_simd_loops(hag)
    # INNER LOOP INSTR
    instructions += inner_simd_loops(hag)


    return instructions


def loop_end_template(hag: ArchitectureNode):
    instructions = []
    return instructions