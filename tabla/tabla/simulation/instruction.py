from pathlib import Path

from tabla.compiler.backend import OP_SELECT_WIDTH, OP_WIDTH, OP_SELECT_BIN, OP_CODE_BIN


class Operation(object):

    def __init__(self, op):
        self.op = op

    def __str__(self):
        return self.op

    @property
    def is_binary_op(self):
        if self.op in ['+', '-', '*', '/', '<', '<=', '>', '>=', '==', '!=']:
            return True
        else:
            return False

    @property
    def is_unary_op(self):
        if self.op in ['sigmoid', 'gaussian', 'sqrt', 'sigmoid_symmetric', 'log', 'pass']:
            return True
        else:
            return False

    @property
    def is_terminal_op(self):
        if self.op == 'DONE':
            return True
        else:
            return False


class Operand(object):
    DEFAULT = None

    def __init__(self, location, index=None):
        # location is one of: ND, NW, NI, NM, PENB, PEGB, PUNB, PUGB
        self.location = location

        # For ND, NW, NI, and NM, index refers to the index of the buffer.
        # For PEGB, index refers to PE ID.
        self.index = index

    def __str__(self):
        if self.location in ['PENB', 'PUNB', 'ALU']:
            index_str = ''
        else:
            index_str = str(self.index)
        s = f'{self.location}{index_str}'
        return s


class Dest(object):

    def __init__(self, location, index=None):
        self.location = location
        self.index = index

    def __str__(self):
        if self.location in ['PENB', 'PUNB']:
            index_str = ''
        else:
            index_str = str(self.index)
        s = f'{self.location}{index_str}'
        return s


class Instruction(object):

    def __init__(self, operation,
                 operand_1, operand_2=Operand.DEFAULT,
                 dest_NI=None, dest_NS=None, dest_PUNB=None, dest_PENB=None, dest_PEGB=None, dest_PUGB=None):
        self.operation = operation
        self.operand_1 = operand_1
        self.operand_2 = operand_2

        # Destination portion of Instruction
        # Note: If ALL of these are None, it means write to ALU (and subsequent instruction must have ALU as one of the Operands)
        self.dest_NI = dest_NI
        self.dest_NS = dest_NS
        self.dest_PUNB = dest_PUNB
        self.dest_PENB = dest_PENB
        self.dest_PEGB = dest_PEGB
        self.dest_PUGB = dest_PUGB

    def __str__(self):
        operand_2_str = '' if self.operand_2 is None else self.operand_2
        dest_NI_str = '' if self.dest_NI is None else self.dest_NI
        dest_NS_str = '' if self.dest_NS is None else self.dest_NS
        dest_PUNB_str = '' if self.dest_PUNB is None else self.dest_PUNB
        dest_PENB_str = '' if self.dest_PENB is None else self.dest_PENB
        dest_PEGB_str = '' if self.dest_PEGB is None else self.dest_PEGB
        dest_PUGB_str = '' if self.dest_PUGB is None else self.dest_PUGB
        s = f'{self.operand_1}, {operand_2_str} {self.operation}, {dest_NI_str} {dest_NS_str} {dest_PUNB_str} {dest_PENB_str} {dest_PEGB_str} {dest_PUGB_str}'
        return s


class InstructionLoader(object):
    OP_CODE = {
        0: 'DONE',  # Opcode to signify all instructions have been processed
        1: '+',
        2: '-',
        3: '*',
        4: '/',
        5: '<',
        6: '<=',
        7: '>',
        8: '>=',
        9: '==',
        10: '!=',
        16: 'sigmoid',
        17: 'gaussian',
        18: 'sqrt',
        19: 'sigmoid_symmetric',
        20: 'log',
        24: 'pass'
    }

    OP_LOCATION = {
        0: 'ZERO',
        1: 'ALU',
        2: 'NW',
        3: 'ND',
        4: 'NM',
        5: 'NI',
        6: ['PENB', 'PUNB'],
        7: ['PEGB', 'PUGB']
    }

    def __init__(self, debug=False):
        # bitwidht_config is a dictionary of bitwidths for each segment of instruction
        self.bitwidth_config = {}

        self.debug = debug

        # TODO minor: store the root directory of the compilation output of the benchmark (e.g. compilation_output/linear_55)


    def set_bitwidth_config(self, summary_filename):
        with open(summary_filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'Op1 index' in line:
                    begin = line.index(':') + 1
                    op_index_bits = int(line[begin :])
                elif 'NI index' in line:
                    begin = line.index(':') + 1
                    ni_index_bits = int(line[begin :])
                elif 'NS Index' in line:
                    begin = line.index(':') + 1
                    ns_index_bits = int(line[begin :])
                elif 'PEGB' in line:
                    begin = line.index(':') + 1
                    pe_bus_width = int(line[begin :])
                elif 'PUGB' in line:
                    begin = line.index(':') + 1
                    pu_bus_width = int(line[begin :])

            self.bitwidth_config = {
                'op_index_bits': op_index_bits,
                'ni_index_bits': ni_index_bits,
                'ns_index_bits': ns_index_bits,
                'pe_bus_width': pe_bus_width,
                'pu_bus_width': pu_bus_width
            }

            if self.debug:
                print(self.bitwidth_config)

    def parse(self, instruction_filename):
        """
        Parse instruction_memory.v file. Return a dictionary of PE ID (absolute_id)
        to a list of Instruction objects for that PE.
        """
        # First, set bitwidth config by parsing summary.txt file (inside parent directory of instruction_memory.v file
        summary_filename = f'{Path(instruction_filename).parent}/../summary.txt'
        self.set_bitwidth_config(summary_filename)

        # Then, read instruction_memory.v file and generate Instruction objects for each PE
        pe_to_insts = {}
        # pe_id = 0
        with open(instruction_filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if '(peId ==' in line:
                    # There's probably better way to do this
                    begin = line.index('==')
                    end = line.index(')')
                    pe_id_str = line[begin + 2 : end]
                    pe_id = int(pe_id_str)
                elif 'rdata = ' in line:
                    # if 'default' in line:
                    #     continue
                    begin = line.index("b")
                    end = line.index(';')
                    inst_binary = line[begin + 1 : end]

                    # Create Instruction Obejct
                    inst = self.parse_inst_binary(inst_binary)
                    # if pe_id == 7:
                    #     print(inst)

                    if pe_id in pe_to_insts:
                        pe_to_insts[pe_id].append(inst)
                    else:
                        pe_to_insts[pe_id] = [inst]

                # if pe_id == 8:
                #     exit()
        return pe_to_insts

    def parse_inst_binary(self, inst_binary):
        """
        Prase a single line of instruction binary, and generates an Instruction object.
        """
        ni_index_bits = self.bitwidth_config["ni_index_bits"]
        ns_index_bits = self.bitwidth_config["ns_index_bits"]
        op_index_bits = self.bitwidth_config["op_index_bits"]
        pe_bus_width = self.bitwidth_config["pe_bus_width"]
        pu_bus_width = self.bitwidth_config["pu_bus_width"]

        # NOTE Instruction Format:
        # instr_bin = f"{op_code}{op1}{op2}{ni_bin}{ns_bin}{neighbor_bin}{pegb_bin}{pugb_bin}"

        # Parse Operation
        begin = 0
        end = begin + OP_WIDTH
        op = self._parse_inst_binary(inst_binary, begin, end, 'operation')

        # Parse Op1
        begin = end
        end = begin + OP_SELECT_WIDTH + op_index_bits
        op1 = self._parse_inst_binary(inst_binary, begin, end, 'operand')

        # Parse Op2
        begin = end
        end = begin + OP_SELECT_WIDTH + op_index_bits
        op2 = self._parse_inst_binary(inst_binary, begin, end, 'operand')

        # Parse NI
        begin = end
        end = begin + ni_index_bits + 1
        dest_NI = self._parse_inst_binary(inst_binary, begin, end, 'ni')

        # Parse NS (Most likely means NW...)
        begin = end
        end = begin + ns_index_bits + 1
        dest_NS = self._parse_inst_binary(inst_binary, begin, end, 'ns')

        # Parse PUNB
        begin = end
        end = begin + 1
        dest_PUNB = self._parse_inst_binary(inst_binary, begin, end, 'punb')

        # Parse PENB
        begin = end
        end = begin + 1
        dest_PENB = self._parse_inst_binary(inst_binary, begin, end, 'penb')

        # Parse PEGB
        begin = end
        end = begin + pe_bus_width
        dest_PEGB = self._parse_inst_binary(inst_binary, begin, end, 'pegb')

        # Parse PUGB
        begin = end
        end = begin + pu_bus_width
        dest_PUGB = self._parse_inst_binary(inst_binary, begin, end, 'pugb')

        # Handle unary operators
        if op.is_unary_op:
            op2 = None

        instr = Instruction(op, op1, op2, dest_NI=dest_NI, dest_NS=dest_NS, dest_PUNB=dest_PUNB, dest_PENB=dest_PENB, dest_PEGB=dest_PEGB, dest_PUGB=dest_PUGB)
        return instr


    def _parse_inst_binary(self, inst_binary, begin, end, type):
        if type == 'operation':
            op_bin_str = inst_binary[begin : end]
            opcode_bin = int(op_bin_str, 2)
            opcode = InstructionLoader.OP_CODE[opcode_bin]
            return Operation(opcode)

        elif type == 'operand':
            operand_str = inst_binary[begin : end]
            location_str = operand_str[:OP_SELECT_WIDTH]
            index_str = operand_str[OP_SELECT_WIDTH : OP_SELECT_WIDTH + self.bitwidth_config['op_index_bits']]
            location_bin = int(location_str, 2)
            # Either PENB or PUNB
            if location_bin == 6:
                # If pe_pu_flag is 0, it means PENB. PUNB if 1.
                pe_pu_flag = int(index_str[-1])
                location_code = InstructionLoader.OP_LOCATION[location_bin]
                location_code = location_code[pe_pu_flag]
                return Operand(location_code)
            # Either PEGB or PUGB
            elif location_bin == 7:
                pe_pu_flag = int(index_str[-1])
                location_code = InstructionLoader.OP_LOCATION[location_bin]
                location_code = location_code[pe_pu_flag]
                index_str = index_str[:-1]
                index_code = int(index_str, 2)
                return Operand(location_code, index_code)
            else:
                location_code = InstructionLoader.OP_LOCATION[location_bin]
                index_code = int(index_str, 2)
                return Operand(location_code, index_code)

        elif type == 'ni':
            ni_str = inst_binary[begin : end]
            ni_bin = int(ni_str[0], 2)
            if ni_bin == 1:
                ni_index_str = ni_str[1: self.bitwidth_config['ni_index_bits'] + 1]
                ni_index = int(ni_index_str, 2)
                return Dest('NI', ni_index)

        elif type == 'ns':
            ns_str = inst_binary[begin : end]
            ns_bin = int(ns_str[0], 2)
            if ns_bin == 1:
                ns_index_str = ns_str[1: self.bitwidth_config['ns_index_bits'] + 1]
                ns_index = int(ns_index_str, 2)
                return Dest('NW', ns_index)

        elif type == 'punb':
            punb_str = inst_binary[begin : end]
            punb_bin = int(punb_str, 2)
            if punb_bin == 1:
                return Dest('PUNB')

        elif type == 'penb':
            penb_str = inst_binary[begin : end]
            penb_bin = int(penb_str, 2)
            if penb_bin == 1:
                return Dest('PENB')

        elif type == 'pegb':
            pegb_str = inst_binary[begin : end]
            pegb_bin = int(pegb_str[ :1], 2)
            if pegb_bin == 1:
                pe_id_str = pegb_str[1: self.bitwidth_config['pe_bus_width']]
                pe_id = int(pe_id_str, 2)
                return Dest('PEGB', pe_id)

        elif type == 'pugb':
            pugb_str = inst_binary[begin : end]
            pugb_bin = int(pugb_str[ :1], 2)
            if pugb_bin == 1:
                pu_id_str = pugb_str[1: self.bitwidth_config['pu_bus_width']]
                pu_id = int(pu_id_str, 2)
                return Dest('PUGB', pu_id)


if __name__ == '__main__':
    op = Operation('+')
    # print(op)

    op1 = Operand('ND', 0)
    # print(op1)

    op2 = Operand('NW', 1)
    # print(op2)

    dest_NI = Dest('NI', 0)
    # print(dest_NI)

    instr = Instruction(op, op1, op2, dest_NI)
    print(instr)
