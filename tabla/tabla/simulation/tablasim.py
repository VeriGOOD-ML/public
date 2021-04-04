import json
import os
import argparse
import pprint

from pytools import memoize_method
from typing import Dict, Union, List, Optional, Tuple

from backend import CYCLE_DELAYS
from backend.bus_arbiter import PEGBArbiter, PUGBArbiter
from backend.component import Component
from backend.instruction import Instruction
from backend.bus import Bus, BusItem, DataNotFoundException
from backend.bus import BusEmptyException
from backend.state import State
from backend.pu import PU
from backend.namespace import Namespace
from backend.schedule_objects import ScheduleEdge, ScheduleNode
from backend.pe import PE
from backend.tabla_template import TablaTemplate
from backend.schedule import Schedule
from backend.schedule_validation import validate_graph
from backend.memory_interface import get_input_weight_nodes, get_input_data_nodes, get_input_meta_nodes, set_values_to_nodes, find_dest_pe_id, print_pe_assignments, Dtype, MemoryInstructionGenerator, reorder_nodes


class TablaSim(object):

    def __init__(self, arch_scheduled: TablaTemplate, config, schedule, interactive_mode=False):
        # Scheduled architecture
        self.arch_scheduled = arch_scheduled[0]
        # Dictionary from PE to list of instructions
        self.instructions = self.get_instructions()
        # self.print_instructions()

        # Needed for instantiating an architecture for second time
        Component.reset_ids()
        self.architecture = TablaTemplate(config)
        self.pes = [pe for _, pe in self.architecture.category_component_dict["pe"].items() if isinstance(pe, PE)]
        self.buses = [bus for _, bus in self.architecture.category_component_dict["bus"].items() if
                      isinstance(bus, Bus)]
        self.pus = [pu for _, pu in self.architecture.category_component_dict["pu"].items() if isinstance(pu, PU)]

        # Set instructions to respective PE
        # self.set_instructions()

        # To keep track of cycles
        self.cycle = 0
        self.weight_read_cycles = 0
        self.data_read_cycles = 0

        self.interactive_mode = interactive_mode
        self.commands = {}

        # PE Global Bus Arbiters for each PU
        self.pegb_arbiters = {pu_id: PEGBArbiter(pu) for pu_id, pu in enumerate(self.pus)}

        # PU Global Bus Arbiter
        pugb = self.architecture.bus_map["PUGB"]
        self.pugb_arbiter = PUGBArbiter(pugb, self.architecture.num_pus)
        self.pugb = pugb

        self.schedule = schedule

    def get_instructions(self):
        instructions = {}
        scheduled_pes = [pe for _, pe in self.arch_scheduled.category_component_dict["pe"].items() if
                         isinstance(pe, PE)]
        for pe_id_scheduled, pe_scheduled in enumerate(scheduled_pes):
            pe_insts = pe_scheduled.all_instructions()
            if pe_insts != []:
                d = {'insts': pe_insts,
                     'inst_position': 0}
                instructions[pe_id_scheduled] = d

            # for cycle in range(pe_scheduled.max_cycle + 1):
            #     try:
            #         instr = pe_scheduled.get_instr(cycle)
            #         if instr is not None:
            #             if pe_id_scheduled not in instructions:
            #                 instructions[pe_id_scheduled] = [instr]
            #             else:
            #                 instructions[pe_id_scheduled].append(instr)
            #     except RuntimeError:
            #         continue
            #     except ValueError:
            #         # print('ValueError')
            #         continue
        return instructions

    def print_instructions(self):
        print()
        print("-" * 80)
        print(f"\t\t\tAll Instructions (Cycle {self.cycle}):")
        for pe_id in self.instructions:
            print(f"PE {pe_id}:")
            for index, inst in enumerate(self.instructions[pe_id]['insts']):
                #pe = self.pes[pe_id]
                #if pe.inst_position == index:
                if index == self.instructions[pe_id]['inst_position']:
                    print(f">> {inst}")
                else:
                    print(f"   {inst}")
            print()
        print("-" * 80)

    def print_current_instructions(self):
        print()
        print("-" * 80)
        print(f"\t\t\tCurrent Instructions to Run (Cycle {self.cycle}):")
        print("\t(Instruction will only run in this cycle if data sources are ready.)\n")
        for pe_id in self.instructions:
            print(f"PE {pe_id}:")
            # pe = self.pes[pe_id]
            #inst = pe.get_current_inst()
            inst = self.get_current_inst(pe_id)
            if inst is not None:
                print(inst)
                print()

    def get_current_inst(self, pe_id):
        current_pos = self.instructions[pe_id]['inst_position']
        insts = self.instructions[pe_id]['insts']

        if current_pos >= len(insts):
            return None
        elif len(insts) > 0:
            return insts[current_pos]
        else:
            return None

    def inst_complete(self, pe_id):
        if pe_id in self.instructions:
            insts = self.instructions[pe_id]['insts']
            if len(insts) == 0:
                return True
            elif self.instructions[pe_id]['inst_position'] >= len(insts):
                return True
            else:
                return False
        else:
            return True

    def find_pe_global_bus(self, pe_id):
        # Find the PE Global Bus for the given PE
        for bus in self.buses:
            if bus.component_subtype == "PEGB":
                if pe_id in bus.pes:
                    return bus

    def memory_interface(self, weight_file, input_data_file, meta_file):
        cycle = 0

        # Load weights to NW
        print("[Memory Interface] Initializing NW namespace...")

        # Get weight nodes
        weight_nodes = get_input_weight_nodes(self.schedule)

        # Set data values to DFG nodes
        set_values_to_nodes(weight_file, weight_nodes)

        # For each node, write its data to corresponding PE namespace
        for node in weight_nodes:
            pe = self.architecture.component_map[node.src_component]
            ns = pe.get_namespace(node.namespace_name)
            ns.insert_data(cycle, node.source_id, node.data_id, node.value)
        #print_pe_assignments(weight_nodes, self.architecture)
        print("Done!")

        # Load input data to ND
        print("[Memory Interface] Initializing ND namespace...")

        # Get input data nodes
        input_data_nodes = get_input_data_nodes(self.schedule)

        # Set data values to DFG nodes
        set_values_to_nodes(input_data_file, input_data_nodes)

        # For each node, write its data to corresponding PE namespace
        for node in input_data_nodes:
            pe = self.architecture.component_map[node.src_component]
            ns = pe.get_namespace(node.namespace_name)
            ns.insert_data(cycle, node.source_id, node.data_id, node.value)
        #print_pe_assignments(input_data_nodes, self.architecture)
        print("Done!")

        # Load metadata to NM
        print("[Memory Interface] Initializing NM namespace...")


        meta_nodes = get_input_meta_nodes(self.schedule)
        component_map = self.architecture.component_map
        dest_pes = {}
        for meta_node in meta_nodes:
            if meta_node.op_name == 'mu':
                for child in meta_node.children:
                    node = self.schedule.get_schedule_node(child)
                    # Find destination PE
                    dest_pe = component_map[node.component_id]

                    inst = str(node.get_instruction())
                    import re
                    match = re.search('NM\d*', inst)
                    nm = match.group()
                    index = int(nm[2:])

                    ns = dest_pe.get_namespace('NM')
                    ns.insert_data(cycle, node.component_id, meta_node.node_id, node.computed)

        # # Get meta data nodes
        # meta_nodes = get_input_meta_nodes(self.schedule)
        # with open(meta_file, 'r') as f:
        #     meta_data = int(f.readlines()[0])
        # for meta in meta_nodes:
        #     meta.value = meta_data

        # # For each node, write its data to corresponding PE namespace
        # for node in meta_nodes:
        #     pe = self.architecture.component_map[node.src_component]
        #     ns = pe.get_namespace(node.namespace_name)
        #     ns.insert_data(cycle, node.source_id, node.data_id, node.value)
        #print_pe_assignments(meta_nodes, self.architecture)
        print("Done!")
        print()

        # Calculate number of cycles took from reading weights and data
        self.weight_read_cycles = self.calculate_memory_interface_weight_read(weight_nodes)
        print(f'Weight read cycles: {self.weight_read_cycles}')

        self.data_read_cycles = self.calculate_memory_interface_data_read(input_data_nodes)
        print(f'Input data read cycles: {self.data_read_cycles}')

        # Increment cycles
        #self.cycle += self.data_read_cycles

    def calculate_memory_interface_weight_read(self, weight_nodes):
        cycles = 0
        return cycles

    def calculate_memory_interface_data_read(self, input_data_nodes):
        input_data_nodes = reorder_nodes(input_data_nodes, self.architecture)
        config = self.architecture.config
        meminst_gen = MemoryInstructionGenerator(input_data_nodes, Dtype.DATA, config['num_axi'], config['num_lanes'],
                                                 config['pes_per_lane'], self.architecture)
        meminst_gen.gen_inst('meminst.json')
        binary = meminst_gen.gen_binary('meminst.txt')
        lines = binary.strip().split('\n')
        extra_cycles = 0
        cycles = len(lines) + extra_cycles
        return cycles

    def get_source_val(self, src, pe, cycle):
        """If source is a global bus (e.g. PEGB), we need to read from Read Buffer."""
        # Get values
        if src.location == "PENB":
            bus = self.architecture.component_map[pe.neighbor_bus]
            val = bus.get_data(cycle, pe)
        elif src.location == "PEGB":
            src_pe_id = self.convert_relative_pe_id_to_absolute(src.index, pe)
            pe_id = pe.category_id
            bus = self.find_pe_global_bus(pe_id)
            print(f"get_source_val(): source PE {src_pe_id}")
            # If source value has just been written by Bus, can't read it until next cycle
            print(bus.pegb_read_buffer_written)
            for i, item in enumerate(bus.pegb_read_buffer_written):
                if src_pe_id == item['src'] and pe_id == item['dst'] and src.source_id == item['id']:
                    # Clear up this flag for next cycle
                    #bus.pegb_read_buffer_written = []
                    del bus.pegb_read_buffer_written[i]
                    raise PEGBDataWrittenInSameCycleException
            val = bus.get_data_from_pegb_read_buffer(cycle, src_pe_id, pe_id)
            # val.value is a tuple of form (val, src_pe_id, dest_pe_id)
            if val is not None:
                return val.value[0]
            else:
                raise DataNotFoundException
        elif src.location == "PUNB":
            containing_pu_id = pe.pu_id
            pu = self.architecture.component_map[containing_pu_id]
            punb_id = pu.neighbor_bus
            punb = self.architecture.component_map[punb_id]
            val = punb.get_data(cycle)
            if pe.component_id == 60:
                print(f"Data read: {val.value}")
        elif src.location == "PUGB":
            containing_pu_id = pe.pu_id
            pu = self.architecture.component_map[containing_pu_id]
            pu_id = pu.category_id

            for item in self.pugb.pugb_read_buffer_written:
                if src.index == item['src'] and pu_id == item['dst']:
                    print(f'{self.pugb.pugb_read_buffer_written}')
                    self.pugb.pugb_read_buffer_written = []
                    raise PUGBDataWrittenInSameCycleException
            val = self.pugb.get_data_from_pugb_read_buffer(cycle, src.index, pu_id)
            if val is not None:
                return val.value[0]
            else:
                raise DataNotFoundException
        elif src.location == "ALU":
            return pe.alu_data
        else:
            ns1 = pe.get_namespace(src.location)
            print(f'data id: {src.data_id}')
            val = ns1.get_data(cycle, src.data_id)
        return val.value

    def check_source_val_exists(self, src, pe, cycle):
        """Returns True if source value exists, False otherwise."""
        if src.location == "PENB":
            bus = self.architecture.component_map[pe.neighbor_bus]
            return bus.check_data_exists(cycle)
        elif src.location == "PEGB":
            src_pe_id = self.convert_relative_pe_id_to_absolute(src.index, pe)
            pe_id = pe.category_id
            bus = self.find_pe_global_bus(pe_id)
            print(f"check_source_val_exists(): source PE {src_pe_id}")

            # If source value has just been written by Bus, can't read it until next cycle
            print(bus.pegb_read_buffer_written)
            for i, item in enumerate(bus.pegb_read_buffer_written):
                if src_pe_id == item['src'] and pe_id == item['dst'] and src.source_id == item['id']:
                    # Clear up this flag for next cycle
                    #bus.pegb_read_buffer_written = []
                    del bus.pegb_read_buffer_written[i]
                    return False
            return bus.check_data_exists_from_pegb_read_buffer(cycle, src_pe_id, pe_id, src.source_id)
        elif src.location == "PUNB":
            containing_pu_id = pe.pu_id
            pu = self.architecture.component_map[containing_pu_id]
            punb_id = pu.neighbor_bus
            punb = self.architecture.component_map[punb_id]
            return punb.check_data_exists(cycle)
        elif src.location == "PUGB":
            return True
        elif src.location == "ALU":
            return True
        else:
            return True

    def get_cycle_delays_for_src(self, src):
        if src.location == "PEGB":
            # Read from PE Read Buffer
            cycle_delay = CYCLE_DELAYS["PE"]["PE"]
        elif src.location == "PENB":
            cycle_delay = CYCLE_DELAYS["PENB"]["PE"]
        elif src.location == "PUNB":
            cycle_delay = CYCLE_DELAYS["PUNB"]["PE"]
        else:
            cycle_delay = CYCLE_DELAYS["NAMESPACE"]["PE"]
        return cycle_delay

    def get_cycle_delays_for_dest(self, dest):
        if dest.location == "PEGB":
            cycle_delay = CYCLE_DELAYS["PE"]["PEGB"]
        elif dest.location == "PENB":
            cycle_delay = CYCLE_DELAYS["PE"]["PENB"]
        elif dest.location == "PUNB":
            cycle_delay = CYCLE_DELAYS["PE"]["PUNB"]
        else:
            cycle_delay = CYCLE_DELAYS["PE"]["NAMESPACE"]
        return cycle_delay

    def all_pes_done_executing(self):
        for pe in self.pes:
            #if not pe.inst_all_complete():
            if not self.inst_complete(pe.category_id):
                return False
        return True

    def print_help(self):
        help_str = ("\t\t\tHelp Menu\n"
                    "Enter 'p' to print current instructions in all PEs.\n"
                    "Enter 'p i' to print all instructions in every PE.\n"
                    "Enter 'p' followed by pe{id}.{namespace} to print valid items of that namespace.\n"
                    "\tFor example, p pe2.nw prints the valid items of NW namespace in PE 2.\n"
                    "\tIt is in Index: Value format.\n"
                    "Enter 'p' followed by pe{id}.{penb} to print the Read Buffer of PE Neighbor Bus for that PE.\n"
                    "\tFor example, p pe1.penb prints the Read Buffer of PE 1 Neighbor Bus. This is supposed to be "
                    "written by PE 0.\n"
                    "Enter 'p' followed by pe{id}.{pegb} to print the Read Buffer, Write Buffer, and Bus for PE "
                    "Global Bus for that PE.\n"
                    "\tFor example, p pe3.pegb prints Read Buffer, Write Buffer of PE3 Global Bus, as well as the Bus "
                    "value for PE GB.\n"
                    "Enter 'p' followed by pe{id}.insts to print all instructions for that PE. Current instruction is "
                    "preceded by >> sign.\n"
                    "\tFor example, p pe4.insts prints all instructions for PE4.\n"
                    "Enter 'p' followed by Component ID (Resource ID) if you have a component ID and would like to know"
                    " which component it represents.\n"
                    "\tFor example, p 24 prints the Component with ID 24.\n"
                    "Enter 'p c' to print current cycle count (Compute cycle ONLY).\n"
                    "Enter 'r' to run instructions for all PEs in current cycle.\n"
                    "\tThis increments the cycle count.\n"
                    "Enter 'c' to continue running all instructions until all instructions in every PE has completed.\n"
                    "Enter 'h' to print this help menu\n"
                    "Enter 'q' to quit.\n")
        print(help_str)

    def print_welcome_menu(self):
        print()
        print("=" * 80)
        print("\t\t\tWelcome to TablaSim")
        print("=" * 80)

    def print_pe_component(self, components):
        pe_name = components[0]
        if pe_name == "pe":
            for pe in self.pes:
                print(pe)
            print()
            return
        pe = self.pes[int(pe_name[2:])]
        if len(components) == 1:
            print(pe)
        elif components[1] in ["nw", "nd", "ni", "nm", "ng"]:
            ns = pe.get_namespace(components[1].upper())
            # TODO ad-hoc
            if not ns.cycle_state_exists(self.cycle):
                ns.generate_intermediate_states(self.cycle)
            ns_items = ns.get_cycle_storage()
            print(ns.component_subtype)
            print("-" * 5)
            for index, nw_item in enumerate(ns_items):
                if nw_item.is_valid():
                    print(f"{index}: value: {nw_item.value}, data id: {nw_item.data_id}")
        elif components[1] == "insts":
            #insts = pe.instructions
            insts = self.instructions[pe.category_id]['insts']
            for index, inst in enumerate(insts):
                #if pe.inst_position == index:
                if self.instructions[pe.category_id]['inst_position'] == index:
                    print(f">> {inst}")
                else:
                    print(f"   {inst}")
        elif components[1] == "penb":
            penb_id = pe.neighbor_bus
            penb = self.architecture.component_map[penb_id]
            # TODO ad-hoc
            if not penb.cycle_state_exists(self.cycle):
                buffer = penb.get_cycle_buffer(penb.max_cycle)
            else:
                buffer = penb.get_cycle_buffer(self.cycle)
            read_buffer = buffer["read"]
            print(f"Read Buffer for PE {pe.category_id}: [", end='')
            for item in read_buffer:
                print(item.value, end=', ')
            print("]")
        elif components[1] == "pegb":
            pegb_id = pe.global_bus
            pegb = self.architecture.component_map[pegb_id]
            if not pegb.cycle_state_exists(self.cycle):
                buffer = pegb.get_cycle_buffer(pegb.max_cycle)
            else:
                buffer = pegb.get_cycle_buffer(self.cycle)

            pe_buffer = buffer[pe.category_id]
            print(f"Read Buffer for PE {pe.category_id}: [", end='')
            for item in pe_buffer["read"]:
                print(item.value, end=', ')
            print("]")
            print(f"Write Buffer for PE {pe.category_id}: [", end='')
            for item in pe_buffer["write"]:
                print(item.value, end=', ')
            print("]")
            bus_val = buffer['pegb']
            if bus_val is not None:
                bus_val = bus_val.value
            print(f"Bus Value: {bus_val}")
        else:
            print(f"[ERROR] Unrecognized component name: {components[1]}")
        print()

    def print_pu_component(self, component):
        pu_name = component[0]
        if int(pu_name[2:]) >= len(self.pus):
            print(f"PU ID {pu_name[2:]} exceeds total number of PUs: {len(self.pus)}.")
            pass
        pu = self.pus[int(pu_name[2:])]
        if len(component) == 1:
            print(pu)
        elif component[1] == "pegb":
            pegb = pu.get_bus("PEGB")
            if not pegb.cycle_state_exists(self.cycle):
                buffer = pegb.get_cycle_buffer(pegb.max_cycle)
            else:
                buffer = pegb.get_cycle_buffer(self.cycle)

            for pe_id in pu.pe_ids:
                pe = pu.get_pe(pe_id)
                pe_buffer = buffer[pe.category_id]
                print(f"PE {pe.category_id}: ", end='')
                print(f"Read: [", end='')
                for item in pe_buffer["read"]:
                    print(item.value, end=', ')
                print("]", end='\t\t\t\t')
                print(f"Write: [", end='')
                for item in pe_buffer["write"]:
                    print(item.value, end=', ')
                print("]")
            bus_val = buffer['pegb']
            if bus_val is not None:
                bus_val = bus_val.value
            print(f"Bus Value: {bus_val}")
        elif component[1] == "pugb":
            pass
        elif component[1] == "punb":
            bus_id = pu.neighbor_bus
            punb = self.architecture.component_map[bus_id]
            if not punb.cycle_state_exists(self.cycle):
                buffer = punb.get_cycle_buffer(punb.max_cycle)
            else:
                buffer = punb.get_cycle_buffer(self.cycle)
            read_buffer = buffer["read"]
            print(f"Read Buffer for PU {pu.category_id} (written by PU {pu.category_id - 1}): [", end='')
            for item in read_buffer:
                print(item.value, end=', ')
            print("]\n")
        else:
            self.print_pe_component(component[1:])

    def print_pugb(self):
        pugb = self.architecture.bus_map["PUGB"]
        # TODO ad-hoc
        if not pugb.cycle_state_exists(self.cycle):
            pugb_buffer = pugb.get_cycle_buffer(pugb.max_cycle)
        else:
            pugb_buffer = pugb.get_cycle_buffer(self.cycle)
        for pu_id in pugb.pus:
            buffer = pugb_buffer[pu_id]
            print(f"PU {pu_id}: ", end='')
            print(f"Read: [", end='')
            for bus_item in buffer["read"]:
                print(bus_item.value, end=', ')
            print("]", end='\t\t\t\t')
            print(f"Write: [", end='')
            for bus_item in buffer["write"]:
                print(bus_item.value, end=', ')
            print("]")
        bus_val = pugb_buffer["bus"]
        if bus_val is not None:
            bus_val = bus_val.value
        print(f"Bus Value: {bus_val}")

    def print_components(self, components):
        # When no argument is given to the 'p' command, print all instructions.
        if len(components) == 0:
            self.print_current_instructions()
            return

        for name in components:
            top_level_name = name[:2]
            if name == "pugb":
                self.print_pugb()
            elif top_level_name == "pe":
                self.print_pe_component(name.split('.'))
            elif top_level_name == "pu":
                self.print_pu_component(name.split('.'))
            elif name == "c":
                print(f"Compute cycle {self.cycle}")
            elif name == "i":
                self.print_instructions()
            elif name.isdigit():
                component = self.architecture.component_map[int(name)]
                print(component)
            else:
                print(f"[ERROR] Unrecognized component name: {components}.")

    def run_interactive(self):
        prompt = "(TablaSim: Cycle {}) "
        command = input(prompt.format(self.cycle + self.data_read_cycles)).strip()
        while command != 'q':
            if command == '':
                pass
            elif command == 'h':
                self.print_help()
            elif command[0] == 'p':
                component_names = command.split()[1:]
                self.print_components(component_names)
            elif command == 'r':
                if self.all_pes_done_executing():
                    print("Simulation complete. No more instructions to execute.\n")
                else:
                    self.run_cycle(self.cycle)
                    self.cycle += 1
            elif command == 'c':
                self.run_non_interactive()
            command = input(prompt.format(self.cycle + self.data_read_cycles))

    def run_cycle(self, cycle):
        print("*" * 80)
        print(f"\t\t\tRunning Instructions in Cycle {cycle + self.data_read_cycles}")

        # Do the bus arbiter routine here
        for pu_id in self.pegb_arbiters:
            pegb_arbiter = self.pegb_arbiters[pu_id]
            pegb_arbiter.run_cycle(cycle)

        self.pugb_arbiter.run_cycle(cycle)

        #for pe_id, pe in enumerate(self.pes):
        for pe_id in self.instructions:
            try:
                pe = self.pes[pe_id]
                instr = self.get_current_inst(pe.category_id)

                #instr = pe.get_current_inst()
                if instr is not None:
                    print(f"\t[PE {pe_id}]")
                    print(instr)

                    # Get operation
                    op = instr.op_name
                    if op == "*":
                        # Get sources
                        src0 = instr.srcs[0]
                        src1 = instr.srcs[1]

                        if not self.check_source_val_exists(src0, pe, cycle):
                            raise SourceNotReadyException
                        if not self.check_source_val_exists(src1, pe, cycle):
                            raise SourceNotReadyException

                        val0 = self.get_source_val(src0, pe, cycle)
                        val1 = self.get_source_val(src1, pe, cycle)

                        # Do the operation
                        out_val = val0 * val1

                        # Get cycle delay
                        cycle_delay_read = max(self.get_cycle_delays_for_src(src0), self.get_cycle_delays_for_src(src1))
                        print(f"{src0.location} = {val0}, {src1.location} = {val1}")
                        print(f"output = {out_val}")
                    elif op == "+":
                        # Get sources
                        src0 = instr.srcs[0]
                        src1 = instr.srcs[1]

                        if not self.check_source_val_exists(src0, pe, cycle):
                            raise SourceNotReadyException
                        if not self.check_source_val_exists(src1, pe, cycle):
                            raise SourceNotReadyException

                        val0 = self.get_source_val(src0, pe, cycle)
                        val1 = self.get_source_val(src1, pe, cycle)

                        # Do the operation
                        out_val = val0 + val1

                        # Get cycle delay
                        cycle_delay_read = max(self.get_cycle_delays_for_src(src0), self.get_cycle_delays_for_src(src1))
                        print(f"{src0.location} = {val0}, {src1.location} = {val1}")
                        print(f"output = {out_val}")
                    elif op == "-":
                        # Get sources
                        src0 = instr.srcs[0]
                        src1 = instr.srcs[1]

                        if not self.check_source_val_exists(src0, pe, cycle):
                            raise SourceNotReadyException
                        if not self.check_source_val_exists(src1, pe, cycle):
                            raise SourceNotReadyException

                        val0 = self.get_source_val(src0, pe, cycle)
                        val1 = self.get_source_val(src1, pe, cycle)

                        # Do the operation
                        out_val = val0 - val1

                        # Get cycle delay
                        cycle_delay_read = max(self.get_cycle_delays_for_src(src0), self.get_cycle_delays_for_src(src1))
                        print(f"{src0.location} = {val0}, {src1.location} = {val1}")
                        print(f"output = {out_val}")
                    elif op == "pass":
                        # Get source
                        src0 = instr.srcs[0]

                        print(f"{src0.source_id}, {src0.source_type}, {src0.source_index}")

                        if not self.check_source_val_exists(src0, pe, cycle):
                            raise SourceNotReadyException

                        val0 = self.get_source_val(src0, pe, cycle)

                        # Do the operation
                        out_val = val0

                        # Get cycle delay
                        cycle_delay_read = self.get_cycle_delays_for_src(src0)
                        print(f"{src0.location} = {val0}")
                        print(f"output = {out_val}")
                    else:
                        print(f"Unsupported operation: {op}")
                        raise Exception

                    # Clear off previous ALU wire value
                    pe.alu_data = None

                    # Write output to destination(s)
                    for dest in instr.dests:
                        if dest.location == "PENB":
                            cycle_delay_write = self.get_cycle_delays_for_dest(dest)
                            cycle_delay = cycle_delay_read + cycle_delay_write
                            print(f"will be written to {dest.location}{dest.index} in cycle {cycle + cycle_delay}")
                            # Get the bus of the neighbor PE
                            pe_neighbor_id = self.architecture.get_pe_neighbor(pe.component_id)
                            pe_neighbor = self.architecture.component_map[pe_neighbor_id]
                            bus = self.architecture.component_map[pe_neighbor.neighbor_bus]
                            bus.add_data_to_neighbor_bus(cycle + cycle_delay, out_val)
                            # bus.add_buffer_data(cycle + cycle_delay, pe.component_id, pe.component_id, dest.dest_id,
                            #                     dest.data_id, out_val)
                        elif dest.location == "PEGB":
                            print(f"{dest.dest_id}")
                            bus = self.find_pe_global_bus(pe_id)
                            # First, write to write buffer (takes 1 cycle)
                            dest_pe_id = self.convert_relative_pe_id_to_absolute(dest.index, pe)
                            out_val = (out_val, pe.category_id, dest_pe_id, dest.dest_id)
                            print(f"output val = {out_val}, data id = {dest.data_id}")
                            print(f"will be written to PE {pe.category_id} Write Buffer in cycle {cycle + 1}")
                            bus.add_data_to_pegb_write_buffer(cycle + 1, pe.category_id, -1, -1, out_val)
                            # bus.add_buffer_data(cycle + 1, pe.component_id, pe.component_id, pe.category_id,
                            #                     dest.data_id, out_val)
                        elif dest.location == "PUNB":
                            cycle_delay_write = self.get_cycle_delays_for_dest(dest)
                            cycle_delay = cycle_delay_read + cycle_delay_write
                            print(f"will be written to {dest.location} in cycle {cycle + cycle_delay}")

                            pu_neighbor_id = self.architecture.get_pu_neighbor(pe.pu_id)
                            pu_neighbor = self.architecture.component_map[pu_neighbor_id]
                            punb = self.architecture.component_map[pu_neighbor.neighbor_bus]
                            punb.add_data_to_neighbor_bus(cycle + cycle_delay, out_val)

                            # punb.add_buffer_data(cycle + cycle_delay, pe.component_id, pe.component_id, dest.dest_id,
                            #                      dest.data_id, out_val)
                        elif dest.location == 'PUGB':
                            containing_pu_id = pe.pu_id
                            pu = self.architecture.component_map[containing_pu_id]
                            src_pu_id = pu.category_id
                            out_val = (out_val, src_pu_id, dest.index)
                            self.pugb.add_data_to_pugb_write_buffer(cycle + 1, src_pu_id, -1, -1, out_val)
                        else:
                            cycle_delay_write = self.get_cycle_delays_for_dest(dest)
                            cycle_delay = cycle_delay_read + cycle_delay_write
                            print(f"will be written to {dest.location}{dest.index} in cycle {cycle + cycle_delay}")
                            dest_ns = pe.get_namespace(dest.location)
                            print(f"dest data id: {dest.data_id}")
                            write_to_nw_flag = False

                            # NW updates need to overwrite old NW values.
                            if dest.location == "NW":
                                for src in instr.srcs:
                                    if src.location == "NW":
                                        dest_data_id = src.data_id
                                        print(f"NW dest data ID: {dest_data_id}")
                                        dest_ns.insert_data(cycle + cycle_delay, dest.dest_id, dest_data_id, out_val)
                                        write_to_nw_flag = True
                                        break
                            if not write_to_nw_flag:
                                dest_ns.insert_data(cycle + cycle_delay, dest.dest_id, dest.data_id, out_val)

                            # Output data value is written to the wire directly connected to the ALU
                            pe.alu_data = out_val
                    # Increment the instruction pointer for this PE
                    #pe.inst_position += 1
                    self.instructions[pe.category_id]['inst_position'] += 1
                    print()
            except ValueError:
                continue
            except BusEmptyException:
                print('Bus Empty, waiting for data to arrive. This instruction will not run in this cycle.\n')
                continue
            except DataNotFoundException:
                print('Data not found in read buffer\n')
                continue
            except PEGBDataWrittenInSameCycleException:
                print(f"PEGB Data already written in cycle {cycle}")
                continue
            except PUGBDataWrittenInSameCycleException:
                print(f"PUGB Data already written in cycle {cycle}")
                continue
            except SourceNotReadyException:
                print(f"Source operand not ready\n")
                continue
            finally:
                pe.cycle += 1
        print(f"\t\t\t\tCycle {cycle + self.data_read_cycles} Complete")
        print("*" * 80)

    def convert_relative_pe_id_to_absolute(self, relative_pe_id, pe):
        containing_pu_id = pe.pu_id
        pu = self.architecture.component_map[containing_pu_id]
        return pu.pe_ids[relative_pe_id]

    def run_non_interactive(self):
        # Execute instructions
        while not self.all_pes_done_executing():
            self.run_cycle(self.cycle + self.data_read_cycles)
            self.cycle += 1
        print("Simulation Complete!\n\n")
        self.max_cycle = self.cycle + self.data_read_cycles
        self.pe_utilization = self.architecture.pe_utilization()
        self.namespace_utilization = self.architecture.namespace_utilization()
        self.pu_utilization = self.architecture.pu_utilization()
        self.print_stats()

    def run(self, weight_file, input_data_file, meta_file):
        self.print_welcome_menu()
        self.memory_interface(weight_file, input_data_file, meta_file)
        if self.interactive_mode is True:
            print("Entering interactive mode...\n")
            self.print_help()
            self.run_interactive()
        else:
            self.run_non_interactive()

    def print_stats(self):
        print("=" * 80)
        print("\t\t\tSimulation Statistics")
        print("=" * 80)
        print(f"Total number of cycles: {self.max_cycle}")
        print(f"Memory interface input data read cycles: {self.data_read_cycles}")
        print(f"Memory interface weight read cycles: {self.weight_read_cycles}")
        print(f"Compute cycles: {self.cycle}")
        print(f"PE utilization: {self.pe_utilization}")
        print(f"PU utilization: {self.pu_utilization}")
        print(f"Namespace utilization:")
        for pe in self.namespace_utilization:
            print(f"{pe}: {self.namespace_utilization[pe]}")


class PEGBDataWrittenInSameCycleException(Exception):
    pass


class PUGBDataWrittenInSameCycleException(Exception):
    pass


class SourceNotReadyException(Exception):
    pass


def main(args):
    with open(args.config_file) as config_file:
        config = json.load(config_file)

    # Sorting algorithm used internally in scheduler
    sort_alg = "custom"

    # Instantiate an architecture for scheduling
    arch_scheduled = TablaTemplate(config)
    sched = Schedule()
    sched.load_dfg(args.dfg_file, sort_type=sort_alg)
    arch_scheduled = sched.schedule_graph(arch_scheduled)

    # Second architecture is instantiated by simulator
    simulator = TablaSim(arch_scheduled, config, sched, args.interactive_mode)
    simulator.run(args.weight_file, args.input_data_file, args.meta_file)


if __name__ == '__main__':
    # Should be project.rtml/tablav2
    compiler_rootdir = os.path.dirname(os.path.abspath(__file__)).rsplit("/", 1)[0]
    dfg_dir = os.path.join(compiler_rootdir, 'tests/test_dfgs')
    dfg_file_default = os.path.join(dfg_dir, "linear_dfg.json")
    config_file_default = os.path.join(compiler_rootdir, 'simulation/config.json')

    argparser = argparse.ArgumentParser(description='TABLA Simulator')
    argparser.add_argument('-d', '--dfg_file',
                           required=True,
                           help='DFG file')
    argparser.add_argument('-c', '--config_file',
                           default=config_file_default,
                           help='config.json file (default: project.rtml/tablav2/simulation/config.json')
    argparser.add_argument('-n', '--input_data_file',
                           required=True,
                           help='Input data file')
    argparser.add_argument('-w', '--weight_file',
                           required=True,
                           help='Weight file')
    argparser.add_argument('-m', '--meta_file',
                           required=True,
                           help='Metadata (learning rate) file')
    argparser.add_argument('-i', '--interactive_mode',
                           default=True,
                           help='Interactive mode to run program cycle by cycle. Default: True. To disable it, '
                                'set this flag to False.')
    args = argparser.parse_args()

    main(args)
