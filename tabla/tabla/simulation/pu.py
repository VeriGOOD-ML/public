from .pe import PE
from .bus import PENB, PEGB
from .buffer import Buffer
from .bus_arbiter import PEGBArbiter
from .defaults import DEFAULT_NAMESPACE_BUFFER_SIZE, DEFAULT_BUS_BUFFER_SIZE, DEFAULT_INPUT_BITWIDTH, DEFAULT_INTERIM_BITWIDTH, DEFAULT_BUS_BITWIDTH


"""
PU has PEs, buses, bus arbiter.
"""
class PU(object):

    def __init__(self, id,
                 pes_per_pu,
                 pe_buffer_size=DEFAULT_NAMESPACE_BUFFER_SIZE,
                 buffer_interim_size=DEFAULT_NAMESPACE_BUFFER_SIZE,
                 input_bitwidth=DEFAULT_INPUT_BITWIDTH,
                 interim_bitwidth=DEFAULT_INTERIM_BITWIDTH,
                 bus_bitwidth=DEFAULT_BUS_BITWIDTH,
                 bus_buffer_size=DEFAULT_BUS_BUFFER_SIZE,
                 debug=False):
        self.id = id
        self.pes_per_pu = pes_per_pu

        # Size of namespace buffers for the PEs that belong to this PU
        self.pe_buffer_size = pe_buffer_size

        # Size of NI (namespace interim) buffer for the PEs that belong ot this PU
        self.buffer_interim_size = buffer_interim_size
        self.input_bitwidth = input_bitwidth
        self.interim_bitwidth = interim_bitwidth

        # Create PEs for this PU
        self.pes = []
        for i in range(pes_per_pu):
            relative_id = i
            absolute_id = pes_per_pu * id + relative_id
            pe = PE(absolute_id, relative_id, self.pe_buffer_size,
                    self.buffer_interim_size,
                    self.input_bitwidth,
                    self.interim_bitwidth,
                    debug=debug)
            self.pes.append(pe)

        # Set Head PE of this PU
        self.head_pe = self.pes[0]
        self.head_pe.is_head_pe = True

        # Set PENB's for each pair of PEs in this PU
        for i, pe in enumerate(self.pes[:-1]):
            source_pe = pe
            dest_pe = self.pes[i + 1]
            penb = PENB(source_pe, dest_pe, debug=debug)
            # print(penb)
            source_pe.set_penb(penb)
        # Set last PE's neighbor to be first PE
        last_pe = self.pes[-1]
        first_pe = self.pes[0]
        penb = PENB(last_pe, first_pe, debug=debug)
        # print(penb)
        last_pe.set_penb(penb)

        self.bus_bitwidth = bus_bitwidth
        self.bus_buffer_size = bus_buffer_size

        # PE Global Bus for the PEs that belong to this PU
        self.pegb = PEGB(self.pes, self.bus_bitwidth, debug=debug)

        # PE Global Bus Arbiter
        self.bus_arbiter = PEGBArbiter(self.pegb, debug=debug)

        self.cycle = 0
        self.debug = debug

    def __str__(self):
        pe_str = ''
        for pe in self.pes:
            pe_str += 'PE ' + str(pe.absolute_id) + ', '
        s = f'PU {self.id}\n' + \
            f'\t{pe_str}\n' + \
            f'\t{self.pegb.__str__()}'
        return s

    def buffer_sizes(self):
        sizes = {}
        for pe in self.pes:
            sizes[f'PE{pe.relative_id}'] = pe.buffer_sizes()
        sizes[f'PEGB'] = int(self.pegb.new_data_written)
        return sizes

    # Probably won't be used much
    def load_instructions_to_pe(self, pe_id_relative, instructions):
        pe = self.pes[pe_id_relative]
        pe.load_instructions(instructions)

    def run_one_cycle(self):
        # if self.debug:
        #     print(f'Cycle {self.cycle}')

        self.accessed = False

        # Dictionary to hold access counts for each on-chip memory component in this PE
        # Example format: {"PE0": pe access stats dictionary...}
        # Use PE absolute id for this
        pu_access_stats = {}

        self.bus_arbiter.run_one_cycle()
        self.accessed = self.bus_arbiter.accessed

        for pe in self.pes:
            if self.debug:
                print(f'PE {pe.relative_id}')

            pe_access_stats = pe.run_one_cycle()
            pu_access_stats[f'PE_{pe.absolute_id}'] = pe_access_stats

            if self.debug:
                if pe.done_processing:
                    print(f'\tPE {pe.relative_id}: DONE PROCESSING')
                else:
                    program_counter = pe.program_counter
                    num_insts = pe.instruction_memory.num_instructions
                    progress_percentage = int(program_counter / num_insts * 100)
                    print(f'\tPE {pe.relative_id} PC: {program_counter} out of {num_insts} total ({progress_percentage} %)')

            self.accessed = self.accessed or pe.accessed

        if self.debug:
            print()
            if self.done_processing:
                print(f'\t*** PU {self.id} DONE PROCESSING ***')
            elif self.accessed is False:
                print(f'\t*** PU {self.id}: Nothing happened in cycle {self.cycle} ***')

        self.cycle += 1

        if self.debug:
            print()


        return pu_access_stats


    def run_cycles(self, cycles):
        for i in range(cycles):
            self.run_one_cycle()



    @property
    def done_processing(self):
        """
        Returns True if this all PE's in this PU completed processing all instructions.
        """
        status = True
        for pe in self.pes:
            if not pe.done_processing:
                # if self.debug:
                #     print(f'\tPE {pe.absolute_id} did not complete processing all insts')
                return False
        return status

    def set_punb(self, punb):
        """
        Set the PUNB of Head PE.
        """
        self.head_pe.set_punb(punb)

    # TODO (Not important) use this in write_to_pu_read_buffer() function and test it
    @property
    def pugb_read_buffer(self):
        return self.head_pe.pugb_read_buffer

    # TODO (Not important) use this in read_from_pu_write_buffer() function and test it
    @property
    def pugb_write_buffer(self):
        return self.head_pe.pugb_write_buffer


if __name__ == '__main__':
    pu = PU(1, 8)
    print(pu)

    pe = pu.pes[1]
    print(pe)
