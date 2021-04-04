from .buffer import Buffer, PEGBWriteBuffer, PUGBWriteBuffer, FIFO
from .flipflop import FlipFlop
from . import Data
from .bus import PENB
from .alu import ALU
from .instruction_memory import InstructionMemory
from .defaults import DEFAULT_NAMESPACE_BUFFER_SIZE, DEFAULT_BUS_BUFFER_SIZE, DEFAULT_INPUT_BITWIDTH, DEFAULT_INTERIM_BITWIDTH, DEFAULT_BUS_BITWIDTH


"""
PE has namespace buffers, ALU, and instruction memory.
"""
class PE(object):

    def __init__(self, absolute_id,
                 relative_id,
                 buffer_size=DEFAULT_NAMESPACE_BUFFER_SIZE,
                 buffer_interim_size=DEFAULT_NAMESPACE_BUFFER_SIZE,
                 input_bitwidth=DEFAULT_INPUT_BITWIDTH,
                 interim_bitwidth=DEFAULT_INTERIM_BITWIDTH,
                 bus_bitwidth=DEFAULT_BUS_BITWIDTH,
                 bus_buffer_size=DEFAULT_BUS_BUFFER_SIZE,
                 instructions=[],
                 is_head_pe=False,
                 debug=False):
        # PU-independent ID of this PE
        self.absolute_id = absolute_id

        # Relative ID (PU-dependent) for this PE within the PU it belongs to
        self.relative_id = relative_id

        # ALU for this PE
        self.alu = ALU()

        # Instruction Memory (ROM) for this PE
        self.instruction_memory = InstructionMemory(instructions)

        # Program Counter for Instruction Memory (Assume the address starts from 0)
        self.program_counter = 0

        # Namespace buffers
        self.ND = Buffer('ND', buffer_size, input_bitwidth, is_loaded=True, pe_id=self.absolute_id)
        self.NW = Buffer('NW', buffer_size, input_bitwidth, is_loaded=True, pe_id=self.absolute_id)
        self.NM = Buffer('NM', buffer_size, input_bitwidth, is_loaded=True, pe_id=self.absolute_id)
        self.NI = Buffer('NI', buffer_interim_size, interim_bitwidth, pe_id=self.absolute_id)

        self.bus_bitwidth = bus_bitwidth
        self.bus_buffer_size = bus_buffer_size

        # PE Neighbor Bus - this will be set by the PU that this PE belongs to
        self.penb = None

        # Read Buffer for PE Neighbor Bus (Note: there's no Write Buffer for PE Neighbor Bus)
        # self.penb_read_buffer = Buffer('PENB Read Buffer', self.bus_buffer_size, self.bus_bitwidth)
        self.penb_read_buffer = FIFO('PENB Read Buffer', self.bus_buffer_size, self.bus_bitwidth)

        # Read Buffer for PE Global Bus
        # self.pegb_read_buffer = Buffer('PEGB Read Buffer', self.bus_buffer_size, self.bus_bitwidth)
        self.pegb_read_buffer = FIFO('PEGB Read Buffer', self.bus_buffer_size, self.bus_bitwidth)

        # Write Buffer for PE Global Bus
        self.pegb_write_buffer = PEGBWriteBuffer('PEGB Write Buffer', self.bus_buffer_size, self.bus_bitwidth)

        # Head PE
        self.is_head_pe = is_head_pe

        # PU Neighbor Bus - this should only be createed for Head PE, and will be set by
        self.punb = None

        # PUNB Read Buffer, PUGB Read/Write Buffer should only be created for Head PE
        # Read Buffer for PU Neighbor Bus (Note: there's no Write Buffer for PU Neighbor Bus)
        # self.punb_read_buffer = Buffer('PUNB Read Buffer', self.bus_buffer_size, self.bus_bitwidth)
        self.punb_read_buffer = FIFO('PUNB Read Buffer', self.bus_buffer_size, self.bus_bitwidth)

        # Read Buffer for PU Global Bus
        # self.pugb_read_buffer = Buffer('PUGB Read Buffer', self.bus_buffer_size, self.bus_bitwidth)
        self.pugb_read_buffer = FIFO('PUGB Read Buffer', self.bus_buffer_size, self.bus_bitwidth)

        # Write buffer for PU Global Bus
        self.pugb_write_buffer = PUGBWriteBuffer('PUGB Write Buffer', self.bus_buffer_size, self.bus_bitwidth)

        # Cycle number
        self.cycle = 0

        # Flip-flops for each stage of the pipeline
        self.inst_flipflop = FlipFlop('Inst Flip Flop')
        self.decode_flipflop = FlipFlop('Decode Filp Flop')
        self.execute_flipflop = FlipFlop('Execute Flip Flop')
        self.initial_stall_cycle = -1
        self.stall_instr = None
        self.stall_instr_id = -1
        self.stall_dependencies = []
        # Flag to signify if pipeline has halted
        self.pipeline_stall = False

        # To hold value for ALU
        self.alu_operand = FlipFlop('ALU Operand Flip Flop')

        # Flag to signify whether this PE completed processing all instructions
        self.done_processing = False

        self.debug = debug


        self.NI_access_count_read = 0
        self.NI_access_count_write = 0
        self.ND_access_count_read = 0
        self.ND_access_count_write = 0
        self.NW_access_count_read = 0
        self.NW_access_count_write = 0
        self.NM_access_count_read = 0
        self.NM_access_count_write = 0
        self.PUNB_read_buffer_access_count = 0
        self.PENB_read_buffer_access_count = 0
        self.PEGB_write_buffer_access_count = 0
        self.PEGB_read_buffer_access_count = 0
        self.PUGB_write_buffer_access_count = 0
        self.PUGB_read_buffer_access_count = 0


    def __str__(self):
        head_str = 'HEAD' if self.is_head_pe is True else ''
        s = f'PE {self.relative_id} ({self.absolute_id}) {head_str}\n' + \
            f'\t{self.ND.__str__()}\n' + \
            f'\t{self.NW.__str__()}\n' + \
            f'\t{self.NM.__str__()}\n' + \
            f'\t{self.NI.__str__()}\n' + \
            f'\t{self.penb_read_buffer.__str__()}\n' + \
            f'\t{self.pegb_read_buffer.__str__()}\n' + \
            f'\t{self.pegb_write_buffer.__str__()}\n' + \
            f'\t{self.penb}'
        return s

    def read_operand_value(self, operand):
        """
        Given an Operand object, read the value from its location (e.g. NI, ND).
        """
        location = operand.location
        value = None
        if location == 'ND':
            index = operand.index
            value = self.ND.read(index)
            self.ND_access_count_read += 1
        elif location == 'NW':
            index = operand.index
            value = self.NW.read(index)
            self.NW_access_count_read += 1
        elif location == 'NI':
            index = operand.index
            value = self.NI.read(index)
            self.NI_access_count_read += 1
        elif location == 'NM':
            index = operand.index
            value = self.NM.read(index)
            self.NM_access_count_read += 1
        elif location == 'ALU':
            value = self.alu_operand.read(self.absolute_id)
        elif location == 'PENB':
            value = self.penb_read_buffer.pop()
            self.PENB_read_buffer_access_count += 1
        elif location == 'PEGB':
            value = self.pegb_read_buffer.pop()
            self.PEGB_read_buffer_access_count += 1
        elif location == 'PUNB':
            value = self.punb_read_buffer.pop()
            self.PUNB_read_buffer_access_count += 1
        elif location == 'PUGB':
            value = self.pugb_read_buffer.pop()
            self.PUGB_read_buffer_access_count += 1
        elif location == 'DONE':
            value = 'DONE' # Signifies all instructions have been processed


        if isinstance(value, Data):
            value = value.value

        return value


    def peek_operand_value(self, operand):
        """
        Similar to read_operand_value(), but only checks if data is available without setting its flag to False
        """
        location = operand.location
        data_available_flag = None
        if location == 'ND':
            index = operand.index
            data_available_flag = self.ND.peek(index)
        elif location == 'NW':
            index = operand.index
            data_available_flag = self.NW.peek(index)
        elif location == 'NI':
            index = operand.index
            data_available_flag = self.NI.peek(index)
        elif location == 'NM':
            index = operand.index
            data_available_flag = self.NM.peek(index)
        elif location == 'ALU':
            data_available_flag = self.alu_operand.peek(self.absolute_id)
        elif location == 'PENB':
            data_available_flag = self.penb_read_buffer.peek_fifo()
        elif location == 'PEGB':
            data_available_flag = self.pegb_read_buffer.peek_fifo()
        elif location == 'PUNB':
            data_available_flag = self.punb_read_buffer.peek_fifo()
        elif location == 'PUGB':
            data_available_flag = self.pugb_read_buffer.peek_fifo()
        elif location == 'DONE':
            data_available_flag = 'DONE' # Signifies all instructions have been processed
        return data_available_flag


    def fetch(self):
        """
        Instruction fetch stage. Fetches instruction and stores it in Inst Fetch Flip Flop.
        """

        # If program counter has gone through all instructions in instruction ROM,
        # don't do anything in this stage
        if self.program_counter >= self.instruction_memory.num_instructions:
            return

        if self.pipeline_stall == False:
            inst = self.instruction_memory.fetch(self.program_counter)

            self.inst_flipflop.write(inst, self.absolute_id, self.relative_id)

            self.program_counter += 1
            if self.debug:
                print(f'[Fetch] instruction fetched: {inst}')

            self.accessed = True

    def decode(self):
        """
        Instruction decode stage. Reads instruction from Inst Fetch Flip Flop
        and stores decoded data in Decode Flip Flop.
        """
        self.stall_dependencies = []

        inst = self.inst_flipflop.read(self.absolute_id)

        # inst is None if previous stage (Fetch) did not complete yet.
        # In that case, don't do anything in this stage.
        if inst == None:
            return

        op = inst.operation

        if op.is_terminal_op:
            inst_decoded = {'op': op,
                            'operand_1_val': None,
                            'operand_2_val': None,
                            'dest_NI': None,
                            'dest_NS': None,
                            'dest_PUNB': None,
                            'dest_PENB': None,
                            'dest_PEGB': None,
                            'dest_PUGB': None,
                            'dest_ALU': None}
            self.decode_flipflop.write(inst_decoded, self.absolute_id, self.relative_id)
            if self.debug:
                inst_decoded_str = '[Decode] instruction decoded: {'
                print(f'[Decode] instruction decoded: Terminal Instruction')
            return
        elif op.is_binary_op:
            operand_1_available = self.peek_operand_value(inst.operand_1)
            operand_2_available = self.peek_operand_value(inst.operand_2)

            # If either of the operands has not been read, stall the pipeline
            if not (operand_1_available and operand_2_available):
                # If stalling, write inst back to Inst Fetch Flip Flop
                if not operand_1_available:
                    self.stall_dependencies.append(str(inst.operand_1))

                if not operand_2_available:
                    self.stall_dependencies.append(str(inst.operand_2))

                self.inst_flipflop.write(inst, self.absolute_id, self.relative_id)

                if self.initial_stall_cycle < 0:
                    self.initial_stall_cycle = self.cycle
                if self.stall_instr_id < 0:
                    self.stall_instr_id = self.program_counter
                self.stall_instr = str(inst)
                self.pipeline_stall = True
                return
        else:
            operand_1_available = self.peek_operand_value(inst.operand_1)
            if not operand_1_available:
                # Handle pipeline stalls
                self.stall_dependencies.append(str(inst.operand_1))

                self.inst_flipflop.write(inst, self.absolute_id, self.relative_id)

                if self.initial_stall_cycle < 0:
                    self.initial_stall_cycle = self.cycle
                if self.stall_instr_id < 0:
                    self.stall_instr_id = self.program_counter
                self.stall_instr = str(inst)

                self.pipeline_stall = True
                return

        # If we're at this point, it means there's no pipeline stalls
        # Set the flag to true, so that Instruction Fetch stage can proceed
        self.pipeline_stall = False
        self.initial_stall_cycle = -1
        self.stall_instr_id = -1
        if op.is_binary_op:
            operand_1_val = self.read_operand_value(inst.operand_1)
            operand_2_val = self.read_operand_value(inst.operand_2)
        else:
            operand_1_val = self.read_operand_value(inst.operand_1)
            operand_2_val = None

        # NOTE We need to also include the destination portion of the Instruction
        # in the Decode Flip Flop (and subsequently Execute Flip Flop) so that
        # it can be read in the Write Back stage
        dest_NI = inst.dest_NI
        dest_NS = inst.dest_NS
        dest_PUNB = inst.dest_PUNB
        dest_PENB = inst.dest_PENB
        dest_PEGB = inst.dest_PEGB
        dest_PUGB = inst.dest_PUGB

        # NOTE THIS WAS WRONG - output always written to ALU
        # # Handle ALU dest (all 0's in the dest bits of instruction binary)
        # # Note: If ALL of these are None, it means write to ALU (and subsequent instruction must have ALU as one of the Operands)
        # dest_ALU = False
        # if dest_NI == None and \
        #    dest_PUNB == None and \
        #    dest_PENB == None and \
        #    dest_PEGB == None and \
        #    dest_PUGB == None:
        #     dest_ALU = True

        dest_ALU = True

        # NOTE Decode Flip Flop contains decoded instruction in a dictionary
        inst_decoded = {'op': op,
                        'operand_1_val': operand_1_val,
                        'operand_2_val': operand_2_val,
                        'dest_NI': dest_NI,
                        'dest_NS': dest_NS,
                        'dest_PUNB': dest_PUNB,
                        'dest_PENB': dest_PENB,
                        'dest_PEGB': dest_PEGB,
                        'dest_PUGB': dest_PUGB,
                        'dest_ALU': dest_ALU}
        self.decode_flipflop.write(inst_decoded, self.absolute_id, self.relative_id)

        if self.debug:
            inst_decoded_str = '[Decode] instruction decoded: '

            op_str = f'{str(op)}, '
            op1_str = f'{operand_1_val}, '
            op2_str = f'{operand_2_val}, ' if operand_2_val != None else ''
            dest_NI_str = f'{dest_NI}, ' if dest_NI != None else ''
            dest_NS_str = f'{dest_NS}, ' if dest_NS != None else ''
            dest_PUNB_str = f'{dest_PUNB}, ' if dest_PUNB != None else ''
            dest_PENB_str = f'{dest_PENB}, ' if dest_PENB != None else ''
            dest_PEGB_str = f'{dest_PEGB}, ' if dest_PEGB != None else ''
            dest_PUGB_str = f'{dest_PUGB}, ' if dest_PUGB != None else ''
            # dest_ALU_str = f'ALU' if dest_ALU == True else ''

            inst_decoded_str += op1_str + \
                                op2_str + \
                                op_str + \
                                dest_NI_str + \
                                dest_NS_str + \
                                dest_PUNB_str + \
                                dest_PENB_str + \
                                dest_PEGB_str + \
                                dest_PUGB_str
            print(inst_decoded_str)

        self.accessed = True


    def execute(self):
        """
        Instruction execute stage. Reads decoded instruction from Decode Flip Flop
        and executes the operation.
        """
        inst_decoded = self.decode_flipflop.read(self.absolute_id)

        # inst_decoded is None if previous stage (Decode) did not complete yet
        # In that case, don't do anything in this stage.
        if inst_decoded == None:
            return

        op = inst_decoded['op']

        operand_1_val = inst_decoded['operand_1_val']
        operand_2_val = inst_decoded['operand_2_val']
        if isinstance(operand_1_val, Data):
            operand_1_val = operand_1_val.value

        if isinstance(operand_2_val, Data):
            operand_2_val = operand_2_val.value
        out_value = self.alu.execute(op, operand_1_val, operand_2_val)

        # Add out_value to the inst_decoded dictionary, so that the Write Back
        # stage can read both the destination portion of the instruction and out_value
        inst_decoded['out_value'] = out_value
        self.execute_flipflop.write(inst_decoded, self.absolute_id, self.relative_id)

        if self.debug:
            dest_NI = inst_decoded['dest_NI']
            dest_PUNB = inst_decoded['dest_PUNB']
            dest_PENB = inst_decoded['dest_PENB']
            dest_PEGB = inst_decoded['dest_PEGB']
            dest_PUGB = inst_decoded['dest_PUGB']
            dest_ALU = inst_decoded['dest_ALU']

            inst_decoded_str = '[Execute] instruction executed: '

            op_str = f'{str(op)}, '
            op1_str = f'{operand_1_val}, '
            op2_str = f'{operand_2_val}, ' if operand_2_val != None else ''
            dest_NI_str = str(dest_NI) if dest_NI != None else ''
            dest_PUNB_str = str(dest_PUNB) if dest_PUNB != None else ''
            dest_PENB_str = str(dest_PENB) if dest_PENB != None else ''
            dest_PEGB_str = str(dest_PEGB) if dest_PEGB != None else ''
            dest_PUGB_str = str(dest_PUGB) if dest_PUGB != None else ''
            # dest_ALU_str = f'ALU' if dest_ALU == True else ''

            inst_decoded_str += op1_str + \
                                op2_str + \
                                op_str + \
                                dest_NI_str + \
                                dest_PUNB_str + \
                                dest_PENB_str + \
                                dest_PEGB_str + \
                                dest_PUGB_str
            print(inst_decoded_str)

        self.accessed = True

    def buffer_sizes(self):
        sizes = {}
        sizes['PENB_READ'] = len(self.penb_read_buffer.entries)
        sizes['PEGB_READ'] = len(self.pegb_read_buffer.entries)
        sizes['PUNB_READ'] = len(self.punb_read_buffer.entries)
        sizes['PUGB_READ'] = len(self.pugb_read_buffer.entries)
        sizes['PEGB_WRITE'] = len(self.pegb_write_buffer.entries)
        sizes['PUGB_WRITE'] = len(self.pugb_write_buffer.entries)

        if self.punb is not None:
            sizes['PUNB'] = int(self.punb.new_data_written)

        if self.penb is not None:
            sizes['PENB'] = int(self.penb.new_data_written)

        sizes['INSTR_FF'] = int(self.inst_flipflop.peek(self.absolute_id))
        sizes['DECODE_FF'] = int(self.decode_flipflop.peek(self.absolute_id))
        sizes['EXECUTE_FF'] = int(self.execute_flipflop.peek(self.absolute_id))
        return sizes


    def write_back(self):
        """
        Data write back stage. Reads value from Execute Flip Flop and
        stores it in the corresponding destination location.
        """
        inst_decoded = self.execute_flipflop.read(self.absolute_id)
        # print(inst_decoded)

        # inst_decoded is None if previous stage (Decode) did not complete yet
        # In that case, don't do anything in this stage.
        if inst_decoded == None:
            return

        out_value = inst_decoded['out_value']

        dest_NI = inst_decoded['dest_NI']
        dest_NS = inst_decoded['dest_NS']
        dest_PUNB = inst_decoded['dest_PUNB']
        dest_PENB = inst_decoded['dest_PENB']
        dest_PEGB = inst_decoded['dest_PEGB']
        dest_PUGB = inst_decoded['dest_PUGB']

        dest_ALU = inst_decoded['dest_ALU']

        if dest_NI != None:
            index = dest_NI.index
            self.NI.write(index, out_value, self.absolute_id, self.relative_id)
            if self.debug:
                print(f'[Write Back] data value {out_value} written back to NI {index}')
            self.NI_access_count_write += 1
            self.accessed = True
        if dest_NS != None:
            # NOTE Assuming NS means NW - other namesapces (ND, NM) don't get written to during exeuction
            index = dest_NS.index
            self.NW.write(index, out_value, self.absolute_id, self.relative_id)
            if self.debug:
                print(f'[Write Back] data value {out_value} written back to NW {index}')
            self.NW_access_count_write += 1
            self.accessed = True
        if dest_PUNB != None:
            # self.punb.write(out_value)
            self.punb.write(Data(out_value,
                                 abs_pe_id=self.punb.dest_head_pe.absolute_id,
                                 rel_pe_id=self.punb.dest_head_pe.relative_id))
            if self.debug:
                print(f'[Write Back] data value {out_value} written back to PUNB')
            # Don't count the writes to bus

            self.accessed = True
        if dest_PENB != None:
            self.penb.write(Data(out_value,
                                 abs_pe_id=self.penb.dest_pe.absolute_id,
                                 rel_pe_id=self.penb.dest_pe.relative_id))
            if self.debug:
                print(f'[Write Back] data value {out_value} written back to PENB')
            # Don't count the writes to bus

            self.accessed = True

        if dest_PEGB != None:
            dest_pe_id_relative = dest_PEGB.index
            # TODO only for debug purposes - remove return statements after done
            # self.accessed is set to True if PEGB Write Buffer push() was successful

            abs_id = (self.absolute_id - self.relative_id) + dest_pe_id_relative

            self.accessed = self.pegb_write_buffer.push(out_value, abs_id, dest_pe_id_relative)

            if self.accessed:
                self.PEGB_write_buffer_access_count += 1

            if self.debug:
                print(f'[Write Back] data value ({out_value}, PE {dest_pe_id_relative}) written back to PEGB Write Buffer')

        if dest_PUGB != None:
            dest_pu_id = dest_PUGB.index
            # TODO only for debug purposes - remove return statements after done
            # self.accessed is set to True if PUGB Write Buffer push() was successful
            dest_pe_id = dest_pu_id*len(self.punb.source_pu.pes)
            # self.accessed = self.pugb_write_buffer.push(out_value, dest_pu_id, dest_pu_id)
            self.accessed = self.pugb_write_buffer.push(out_value, dest_pe_id, 0)
            if self.accessed:
                self.PUGB_write_buffer_access_count += 1

            if self.debug:
                print(f'[Write Back] data value ({out_value}, PU {dest_pu_id}) written back to PUGB Write Buffer')


        # We always write to ALU flip flop
        self.alu_operand.write(out_value, self.absolute_id, self.relative_id)

        # if dest_ALU:
        #     self.alu_operand.write(out_value)
        #     if self.debug:
        #         print(f'[Write Back] data value {out_value} written back to ALU')
        #     self.accessed = True

        # Signify this PE is done procesing all instructions
        if inst_decoded['op'].is_terminal_op:
            self.done_processing = True
            # Need to set this here so simulator doesn't think it's stuck, even though at this point all instructions have been processed
            self.accessed = True
            if self.debug:
                print(f'[Write Back] Terminal Op')


    def run_one_cycle(self):
        """
        Run for one cycle. 4-stage pipeline: Fetch, decode, execute, and write back.
        Outputs of each stage is stored in a flip flop.
        Fetch -> flip flop -> Decode -> flip flop -> ALU -> flip flop -> Write back
        """
        # print(f'Cycle {self.cycle}:')

        # Initialize access count statistics to 0 in every cycle
        # self.NI_access_count_read = 0
        # self.NI_access_count_write = 0
        # self.ND_access_count_read = 0
        # self.ND_access_count_write = 0
        # self.NW_access_count_read = 0
        # self.NW_access_count_write = 0
        # self.NM_access_count_read = 0
        # self.NM_access_count_write = 0
        # self.PUNB_read_buffer_access_count = 0
        # self.PENB_read_buffer_access_count = 0
        # self.PEGB_write_buffer_access_count = 0
        # self.PEGB_read_buffer_access_count = 0
        # self.PUGB_write_buffer_access_count = 0
        # self.PUGB_read_buffer_access_count = 0

        # Debug - if ANY component has been accessed, set to True
        self.accessed = False

        # TODO Test this more
        if self.is_head_pe:
            assert self.punb != None
            self.punb.write_to_pu_read_buffer()

        # TODO Test this more
        assert self.penb != None
        self.penb.write_to_pe_read_buffer()

        # (4) Write back
        self.write_back()

        # (3) Execute instruction
        self.execute()

        # (2) Decode instruction
        self.decode()

        # (1) Fetch instruction from Instruction Memory
        self.fetch()

        if self.debug:
            if self.accessed is False:
                print(f'\tPE {self.relative_id}: nothing happened in cycle {self.cycle}')

        self.cycle += 1

        # Dictionary to hold access counts for each on-chip memory component in this PE
        # Example format: {"NI": {"read": 3, "write": 3, "data_size": 16}}
        access_stats = {'NI': {'read': self.NI_access_count_read,
                               'write': self.NI_access_count_write,
                               'data_size': self.NI.bitwidth},
                        'ND': {'read': self.ND_access_count_read,
                               'write': self.ND_access_count_write,
                               'data_size': self.ND.bitwidth},
                        'NW': {'read': self.NW_access_count_read,
                               'write': self.NW_access_count_write,
                               'data_size': self.NW.bitwidth},
                        'NM': {'read': self.NM_access_count_read,
                               'write': self.NM_access_count_write,
                               'data_size': self.NM.bitwidth},
                        'PUNB_Read_Buffer': {'read': self.PUNB_read_buffer_access_count,
                                             'data_size': self.punb_read_buffer.bitwidth},
                        'PENB_Read_Buffer': {'read': self.PENB_read_buffer_access_count,
                                             'data_size': self.penb_read_buffer.bitwidth},
                        'PEGB_Read_Buffer': {'read': self.PEGB_read_buffer_access_count,
                                             'data_size': self.pegb_read_buffer.bitwidth},
                        'PEGB_Write_Buffer': {'write': self.PEGB_write_buffer_access_count,
                                              'data_size': self.pegb_write_buffer.bitwidth},
                        'PUGB_Read_Buffer': {'read': self.PUGB_read_buffer_access_count,
                                             'data_sze': self.pugb_read_buffer.bitwidth},
                        'PUGB_Write_Buffer': {'write': self.PUGB_write_buffer_access_count,
                                              'data_size': self.pugb_write_buffer.bitwidth}}
        return access_stats


    def run_cycles(self, cycles):
        """
        Run instructions for the given number of cycles.
        """
        for i in range(cycles):
            self.run_one_cycle()

    def set_penb(self, penb):
        """
        PU sets PENB for this PE.
        """
        self.penb = penb

    def set_punb(self, punb):
        """
        Should only be set for Head PE.
        """
        self.punb = punb

    def load_instructions(self, instructions):
        self.instruction_memory.load_instructions(instructions)

    def print_instructions(self):
        print(self.instruction_memory)


if __name__ == '__main__':
    from instruction import Operation, Operand, Dest, Instruction

    op = Operation('+')
    op1 = Operand('ND', 0)
    op2 = Operand('NW', 1)
    dest_NI = Dest('NI', 0)
    instr = Instruction(op, op1, op2, dest_NI)
    instructions = [instr]

    pe = PE(9, 1, input_bitwidth=16, interim_bitwidth=32, instructions=instructions, debug=True)
    # print(pe)

    pe.ND.write(0, 2)
    pe.NW.write(1, 5)

    pe.run_cycles(5)
