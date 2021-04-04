from tablav2.simulation.tabla import Tabla
from tablav2.simulation.pe import PE
from tablav2.simulation.pu import PU
from tablav2.simulation.instruction import InstructionLoader
from tablav2.simulation.instruction import Operation, Operand, Dest, Instruction


def create_inst():
    pe_to_insts = {}

    instructions = []

    # ND0, NW0, +,
    op = Operation('+')
    op1 = Operand('ND', 0)
    op2 = Operand('NW', 0)
    instr = Instruction(op, op1, op2)
    instructions.append(instr)

    # NW0, ALU, *, NI0
    op = Operation('*')
    op1 = Operand('NW', 1)
    op2 = Operand('ALU')
    dest_NI = Dest('NI', 0)
    instr = Instruction(op, op1, op2, dest_NI=dest_NI)
    instructions.append(instr)

    op = Operation('DONE')
    op1 = Operand('DONE')
    instr = Instruction(op, op1)
    instructions.append(instr)

    pe_to_insts[0] = instructions

    return pe_to_insts


def test():
    config = {'num_pus': 1,
              'pes_per_pu': 1,
              'input_bitwidth': 16,
              'interim_bitwidth': 32}
    tabla = Tabla(config, debug=True)
    print(tabla)

    pe_to_insts = create_inst()
    pu2 = tabla.pus[0]
    pu2.head_pe.load_instructions(pe_to_insts[0])

    pu2.head_pe.print_instructions()

    pu2.head_pe.ND.write(0, 2)
    pu2.head_pe.NW.write(0, 3)
    pu2.head_pe.NW.write(1, 5)

    while not tabla.done_processing:
        tabla.run_one_cycle()

    tabla.print_statistics()


if __name__ == '__main__':
    test()
