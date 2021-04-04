import json
import pprint

from pathlib import Path

from tabla.simulation.tabla import Tabla
from tabla.simulation.instruction import InstructionLoader


class Simulator(object):

    def __init__(self, benchmark_dir, config_filename, debug=False):

        # Path to the compilation output of the benchmark
        self.benchmark_dir = benchmark_dir

        # Config filename
        self.config_filename = config_filename
        self.config = self.create_config()

        self.debug = debug

        # Instatiate Tabla architecture to simulate
        self.tabla = Tabla(self.config, self.benchmark_dir, self.debug)

        # Load instructions to Tabla
        inst_loader = InstructionLoader()
        inst_filename = f'{Path(benchmark_dir)}/compute-inst/instruction_memory.v'
        pe_to_insts = inst_loader.parse(inst_filename)
        self.tabla.load_instructions(pe_to_insts)


        # Simulator maintains a dictionary to store on chip memory access statistics metrics
        # Format:
        # {cycle0 : Tabla access stats dictionary...}
        self.access_stats = {}


    def create_config(self):
        with open(self.config_filename, 'r') as f:
            config = json.load(f)
            return config

    def run(self):

        # Memory Interface: Off-chip memory access
        offchip_access_stats = self.tabla.get_offchip_access_stats()
        self.access_stats['Offchip_access'] = offchip_access_stats
        temp_cycles = 0
        while not self.tabla.done_processing:
            # print(f'Cycle {self.tabla.cycle}')
            self.run_one_cycle()
            temp_cycles += 1
        print(f"Temp Cycles: {temp_cycles}")
        # final_buffer_sizes = self.tabla.buffer_sizes()
        # from pprint import pprint
        # pprint(final_buffer_sizes)


    def run_one_cycle(self):
        cycle = self.tabla.cycle
        tabla_access_stats = self.tabla.run_one_cycle()
        self.access_stats[f'Cycle_{cycle}'] = tabla_access_stats


    def run_cycles(self, cycles):
        for i in range(cycles):
            self.run_one_cycle()


    def print_statistics(self):
        offchip_read = self.access_stats['Offchip_access']['read']
        offchip_write = self.access_stats['Offchip_access']['write']

        last_cycle_stat = self.access_stats[f'Cycle_{self.tabla.cycle - 1}']
        # pprint.pprint(last_cycle_stat)
        nd_read_total = 0
        nd_write_total = 0
        ni_read_total = 0
        ni_write_total = 0
        nw_read_total = 0
        nw_write_total = 0
        nm_read_total = 0
        nm_write_total = 0

        for pu_key, pu in last_cycle_stat.items():
            for pe_key, pe in pu.items():
                nd_read = pe['ND']['read']
                nd_write = pe['ND']['write']
                ni_read = pe['NI']['read']
                ni_write = pe['NI']['write']
                nw_read = pe['NW']['read']
                nw_write = pe['NW']['write']
                nm_read = pe['NM']['read']
                nm_write = pe['NM']['write']

                nd_read_total += nd_read
                nd_write_total += nd_write
                ni_read_total += ni_read
                ni_write_total += ni_write
                nw_read_total += nw_read
                nw_write_total += nw_write
                nm_read_total += nm_read
                nm_write_total += nm_write



        print(f'Offchip READ: {offchip_read}')
        print(f'Offchip WRITE: {offchip_write}')
        print(f'ND READ: {nd_read_total}')
        print(f'ND WRITE: {nd_write_total}')
        print(f'NI READ: {ni_read_total}')
        print(f'NI WRITE: {ni_write_total}')
        print(f'NW READ: {nw_read_total}')
        print(f'NW WRITE: {nw_write_total}')
        print(f'NM READ: {nm_read_total}')
        print(f'NM WRITE: {nm_write_total}')

        print(f'Cycles: {self.tabla.cycle - 1}')

        print(self.tabla.memory_interface)


    def write_statistics(self):
        with open('stats_svm_.json', 'w') as f:
            json.dump(self.access_stats, f)


    # For debug purposes
    def only_debug_pu(self, pu_id):
        """
        Print debug statements for the given PU only.
        """
        # Set debug flag to False to all components
        # self.tabla.pugb.debug = False
        # self.tabla.bus_arbiter.debug = False
        for pu in self.tabla.pus:
            pu.debug = False
            pu.pegb.debug = False
            pu.bus_arbiter.debug = False
            for pe in pu.pes:
                pe.debug = False
                pe.penb.debug = False

        pu = self.tabla.pus[pu_id]
        pu.debug = True
        pu.pegb.debug = True
        pu.bus_arbiter.debug = True
        for pe in pu.pes:
            pe.debug = True
            pe.penb.debug = True
