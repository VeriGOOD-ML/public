from .instruction import Instruction


class InstructionMemory(object):
    """
    List of Instruction objects.
    """

    def __init__(self, instructions=[]):
        # instructions is a list of Instruction objects
        self.instructions = instructions

    def __str__(self):
        s = 'Instruction Memory:\n'
        for number, inst in enumerate(self.instructions):
            s += f'{number}: ' + inst.__str__() + '\n'
        return s

    def load_instructions(self, instructions):
        # Initializes this instruction memory with Instruction objects
        self.instructions = instructions

    def fetch(self, index):
        # Fetches the instruction for given index
        return self.instructions[index]

    @property
    def num_instructions(self):
        return len(self.instructions)


if __name__ == '__main__':
    from instruction import Operation, Operand, Dest

    inst_mem = InstructionMemory()
    print(inst_mem)

    op = Operation('+')
    op1 = Operand('ND', 0)
    op2 = Operand('NW', 1)
    dest_NI = Dest('NI', 0)
    instr = Instruction(op, op1, op2, dest_NI)

    instructions = [instr]

    inst_mem.load_instructions(instructions)
    print(inst_mem)

    inst = inst_mem.fetch(0)
    print(inst)

    print(inst_mem.num_instructions)
