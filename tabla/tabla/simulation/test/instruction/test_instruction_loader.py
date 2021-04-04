from tablav2.simulation.instruction import InstructionLoader
from tablav2.simulation.pe import PE


def test():
    inst_loader = InstructionLoader()
    inst_filename = '../../../compilation_output/linear_55/compute-inst/instruction_memory.v'
    # inst_filename = '../../../compilation_output/design_5_7_reco_138_130_10/compute-inst/instruction_memory.v'

    pe_to_insts = inst_loader.parse(inst_filename)

    print(len(pe_to_insts))
    pe1_insts = pe_to_insts[2]

    pe = PE(2, 2, instructions=pe1_insts, debug=True)

    pe.print_instructions()


if __name__ == '__main__':
    test()
