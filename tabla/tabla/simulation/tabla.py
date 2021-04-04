from .pu import PU
from .bus import PUNB, PUGB
from .bus_arbiter import PUGBArbiter
from .memory_interface import MemoryInterface
from .defaults import DEFAULT_NAMESPACE_BUFFER_SIZE, DEFAULT_BUS_BUFFER_SIZE, DEFAULT_INPUT_BITWIDTH, DEFAULT_INTERIM_BITWIDTH, DEFAULT_BUS_BITWIDTH


"""
Class that instantiates PUs, PU Global Bus, PUGB Bus Arbiter, and memory interface.
"""
class Tabla(object):

    def __init__(self, config, benchmark_dir, debug=False):

        # Config JSON. Has all the design parameters we need
        self.config = config

        # Design parameters
        self.num_pus = self.config['num_pus']
        self.pes_per_pu = self.config['pes_per_pu']
        self.input_bitwidth = self.config['input_bitwidth']
        self.interim_bitwidth = self.config['interim_bitwidth']

        # Other design parameters not required to be specified in config.json
        if 'bus_bitwidth' in self.config:
            self.bus_bitwidth = self.config['bus_bitwidth']
        else:
            self.bus_bitwidth = DEFAULT_BUS_BITWIDTH

        # Create PU's, which instantiates their cooresponding PE's
        self.pus = []
        for i in range(self.num_pus):
            pu = PU(i, self.pes_per_pu, debug=debug)
            self.pus.append(pu)

        # Set PUNB's for each pair of PUs
        for i, pu in enumerate(self.pus[:-1]):
            source_pu = pu
            dest_pu = self.pus[i + 1]
            punb = PUNB(source_pu, dest_pu, debug=debug)
            source_pu.set_punb(punb)
        # Set last PU's neighbor to first PU
        last_pu = self.pus[-1]
        first_pu = self.pus[0]
        punb = PUNB(last_pu, first_pu, debug=debug)
        last_pu.set_punb(punb)

        # Create PU Global Bus
        self.pugb = PUGB(self.pus, self.bus_bitwidth, debug=debug)

        # PUGB Bus Arbiter
        self.bus_arbiter = PUGBArbiter(self.pugb, debug)

        # Memory Interface
        self.memory_interface = MemoryInterface(self.config, benchmark_dir)

        self.cycle = 0
        self.debug = debug


    def __str__(self):
        s = f'Tabla: {self.num_pus} PUs, {self.pes_per_pu} PEs/PU, input bitwidth {self.input_bitwidth}, interim bitwidth {self.interim_bitwidth}'
        return s


    def load_instructions(self, pe_to_insts):
        """
        Load Instructions to each PE. pe_to_insts is a dictionary of PE ID
        (absolute_id) to list of Instructions, provided by InstructionLoader.
        """
        for pu in self.pus:
            pes = pu.pes
            for pe in pes:
                instructions = pe_to_insts[pe.absolute_id]
                pe.load_instructions(instructions)


    def run_one_cycle(self):
        if self.debug:
            print('=' * 60)
            print(f'\t\t\tCycle {self.cycle}')

        self.accessed = False

        # Dictionary to hold access counts for each on-chip memory component in this PE
        # Example format: {"PU0": pu access stats dictionary...}
        tabla_access_stats = {}

        self.accessed = self.bus_arbiter.run_one_cycle()

        for pu in self.pus:
            if self.debug:
                print(f'\t[PU {pu.id}]')

            pu_access_stats = pu.run_one_cycle()
            tabla_access_stats[f'PU_{pu.id}'] = pu_access_stats

            self.accessed = self.accessed or pu.accessed

        if self.accessed is False:
            print(f'Simulation got stuck in cycle {self.cycle}')
            stall_deps = []
            for pu in self.pus:
                for pe in pu.pes:
                    if pe.pipeline_stall and not pe.done_processing:
                        stall_deps.append(f"PE{pe.absolute_id}: {pe.stall_dependencies}, "
                                          f"Cycle: {pe.initial_stall_cycle}, "
                                          f"PC: {pe.stall_instr_id}, "
                                          f"Total Instructions: {pe.instruction_memory.num_instructions}, "
                                          f"Instr: {pe.stall_instr}, ")
            print(f"Stall dependencies:")
            print('\n'.join(stall_deps))
            exit()

        self.cycle += 1

        return tabla_access_stats

    def buffer_sizes(self):
        assert self.done_processing
        buffer_sizes = {}
        for pu in self.pus:
            buffer_sizes[f'PU{pu.id}'] = pu.buffer_sizes()
        buffer_sizes['PUGB'] = int(self.pugb.new_data_written)
        return buffer_sizes

    def run_cycles(self, cycles):
        for i in range(cycles):
            self.run_one_cycle()

    def get_offchip_access_stats(self):
        offchip_access_stats = self.memory_interface.get_offchip_access_stats()

        return offchip_access_stats

    @property
    def done_processing(self):
        status = True
        for pu in self.pus:
            if not pu.done_processing:
                return False
        return status

    def print_statistics(self):
        """
        Print number of cycles and memory access counts.
        """
        print(f'Cycles: {self.cycle}')
