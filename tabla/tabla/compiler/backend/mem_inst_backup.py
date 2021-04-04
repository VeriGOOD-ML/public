import json
import os
from typing import List
from numpy.random import randint
import numpy as np

import argparse
from enum import Enum

from backend import ScheduleNode, Schedule
from backend import TablaTemplate
from backend import OP_SELECT_WIDTH, OP_WIDTH, MEM_INTERFACE_WIDTH, BUS_WIDTH
from backend import PE


class Lane(object):
    """A Lane is a memory interface component that connects a set of PEs together. Once data is read through AXI, it is
    fed to Lanes before being written to its corresponding PEs.

    TODO Make this inherit from Component class.
    """

    def __init__(self, laneid: int, peids: List[int]):
        """

        Parameters
        ----------
        laneid : int
                Unique ID assigned to this Lane.
        peids : List[int]
                IDs of PEs attached to this Lane.
        """
        self.laneid = laneid
        self.peids = peids

    def get_relpeid(self, peid: int) -> int:
        """Given a PE ID, returns the relative offset in this Lane.
        """
        if peid in self.peids:
            return self.peids.index(peid)
        else:
            raise Exception("PE (ID: {:d}) does not exist in this lane!".format(peid))

    def __str__(self) -> str:
        return f'Lane {self.laneid}: PE IDs: {self.peids}'


class LaneGenerator(object):
    """A class to manage Lanes. Given the total number of lanes and PEs per Lane, Generates the Lanes accordingly.
    """

    def __init__(self, architecture: TablaTemplate, nlanes: int = 16, pes_per_lane: int = 4):
        """

        Parameters
        ----------
        nlanes : int
                Number of Lanes to be generated.
        pes_per_lane : int
                Number of PEs attached to each Lane.
        """
        self.architecture = architecture
        self.nlanes = nlanes
        self.pes_per_lane = pes_per_lane

    def init_lanes(self):
        lanes = []
        for base_peid in range(self.nlanes):
            lanes.append(Lane(base_peid, [base_peid + self.nlanes * i for i in range(self.pes_per_lane)]))
        return lanes

    def get_lanes_by_shift_amount(self, batch: List):
        """Given a batch, figure out shift amounts for each Lane, and group these by shift amount.
        """
        lanes_by_shift = {}
        for curr_lane, data in enumerate(batch):
            if data is not None:
                component_map = self.architecture.component_map
                dest_pe = component_map[data.src_component]
                dest_pe_id = dest_pe.category_id
                # print(f"Lane: {curr_lane}, Data: {data._edge_name}, Namespace: {data.namespace_name}, PE: {dest_pe_id}")
                # print(data._edge_name, dest_pe_id, data.path)
                # for comp in data.path:
                #     if component_map[comp].component_type == "pe":
                #         print("PE ID: ", component_map[comp].category_id)
                dest_lane_id = self.get_dest_laneid(dest_pe_id)
                shift_amount = self.get_shift_amount(curr_lane, dest_lane_id)

            if shift_amount in lanes_by_shift:
                lanes_by_shift[shift_amount].append((dest_lane_id, dest_pe_id))
            else:
                lanes_by_shift[shift_amount] = [(dest_lane_id, dest_pe_id)]
            # print("pos: {:d}, dest_lane_id: {:d}, shift to left: {:d}".format(curr_lane, dest_lane_id, shift_amount))
        return lanes_by_shift

    def get_shift_amount(self, curr_lane_id: int, dest_lane_id: int) -> int:
        """Given a current Lane position and destination Lane, calculate the shift amount (left shift only).

        Parameters
        ----------
        curr_lane_id : int
                Current Lane position.
        dest_lane_id : int
                Destination Lane position.

        Returns
        -------
        Shift amount
        """
        if curr_lane_id >= dest_lane_id:
            shift_amount = curr_lane_id - dest_lane_id
        else:
            shift_amount = self.nlanes - (dest_lane_id - curr_lane_id)
        return shift_amount

    def get_dest_laneid(self, pe_id):
        return pe_id % self.nlanes

    def get_lane(self, lanes, lane_id):
        return lanes[lane_id]


class AXI(object):
    """AXI Master.
    """

    def __init__(self, id, axi_size: int = 64, axi_read_cycle: int = 4):
        self.id = id
        # these two variables determine how many read instructions are required
        self.axi_size = axi_size  # number of data elements read by each AXI
        self.axi_read_cycle = axi_read_cycle  # number of data elements read in one cycle
        self.lanes = []
        self.data = []  # all data
        self.data_by_cycle = []  # all data grouped by cycle (4 per cycle)

    def set_lanes(self, lanes):
        self.lanes = lanes

    def __str__(self):
        lanes = ''
        for lane in self.lanes:
            lanes += str(lane) + '\n'
        return f'AXI {self.id}:\n{lanes}'


class AXIController(object):
    """Reads data from DDR through AXI.
    """

    def __init__(self, axi_list, architecture):
        self.axi_list = axi_list
        # TODO The following two lines are too ad-hoc.
        self.axi_size = self.axi_list[0].axi_size
        self.axi_read_cycle = self.axi_list[0].axi_read_cycle
        self.architecture = architecture

    @property
    def max_cycle(self):
        cycle = 0
        for axi in self.axi_list:
            if len(axi.data_by_cycle) > cycle:
                cycle = len(axi.data_by_cycle)

        return cycle

    def assign_axi(self, data: List[ScheduleNode]):
        """Assigns each data element to corresponding AXI master.

        TODO This is buggy if data size is greater than `self.axi_size * len(self.axi_list)`.
        """
        axis = len(data) // self.axi_size
        r = len(data) % self.axi_size
        for i in range(axis):
            self.axi_list[i % 4].data.extend(data[i * self.axi_size: i * self.axi_size + self.axi_size])
        if r > 0:
            if axis == 0:
                self.axi_list[0].data.extend(data[:])
            else:
                i += 1
                self.axi_list[i % 4].data.extend(data[i * self.axi_size:])

    def assign_weights_to_pe(self, weight_nodes):
        for weight_node in weight_nodes:
            # Find destination PE
            component_map = self.architecture.component_map
            dest_pe = component_map[weight_node.src_component]
            dest_pe_id = dest_pe.category_id

            # Put the node in the PE
            dest_pe.weight_nodes.append(weight_node)

    def print_axi_contents(self):
        for axi in self.axi_list:
            print(f'AXI {axi.id}:')
            for lane in axi.lanes:
                print(f'Lane {lane.laneid}:')
                for pe_id in lane.peids:
                    print(f'PE ID {pe_id}: ', end='')
                    pe = self.architecture.cat_component_map['pe'][pe_id]
                    for data in pe.weight_nodes:
                        print(f'{data._edge_name}', end=', ')
                    print()
                print()
            print()

    def gen_matrix_for_axi(self, axi_id):
        axi = self.axi_list[axi_id]

        lane_data = []
        for lane in axi.lanes:
            weight_data = []
            for pe_id in lane.peids:
                pe = self.architecture.cat_component_map['pe'][pe_id]
                values = [node.value for node in pe.weight_nodes]
                weight_data.extend(values)
            lane_data.append(weight_data)
        return lane_data

    def find_max_number_of_weights(self, lanes):
        max_num = -1
        for lane in lanes:
            num_weights = len(lane)
            if num_weights > max_num:
                max_num = num_weights
        return max_num

    def put_placeholder(self, weight_matrix, pe_index, lane_index, num_placeholder):
        """This is only used for weights."""
        values = weight_matrix[pe_index, lane_index]
        concatenated = np.append(values, np.zeros((num_placeholder,), dtype=int))
        weight_matrix[pe_index, lane_index] = concatenated

    def divide_axi_data_by_cycle(self):
        import math
        """Groups AXI data by cycle. Every AXI cna read 4 data elements at a time."""
        for axi in self.axi_list:
            cycls = len(axi.data) // self.axi_read_cycle
            r = len(axi.data) % self.axi_read_cycle
            for i in range(cycls):
                axi.data_by_cycle.append(axi.data[i * self.axi_read_cycle: i * self.axi_read_cycle + self.axi_read_cycle])
            if r > 0:
                if cycls == 0:
                    axi.data_by_cycle.append(axi.data[:])
                else:
                    i += 1
                    axi.data_by_cycle.append(axi.data[i * self.axi_read_cycle:])

    def get_axi_data_for_cycle(self, cycle: int):
        """Reads all data from every AXI master in the given cycle.
        """
        batch = []
        for axi in self.axi_list:
            if cycle >= len(axi.data_by_cycle):
                continue
            else:
                batch.extend(axi.data_by_cycle[cycle])
        return batch

    def get_axi_head_data(self):
        """Gets head data from each axi"""
        batch = []
        for axi in self.axi_list:
            head_data = axi.data_by_cycle.pop(0)
            batch.extend(head_data)
        return batch

    def peek_axi_head_data(self):
        batch = []
        for axi in self.axi_list:
            if len(axi.data_by_cycle) == 0:
                batch.extend([None, None, None, None])
            else:
                head_data = axi.data_by_cycle[0]
                batch.extend(head_data)
        return batch

    def write_axi_data(self, axi_dir):
        for axi in self.axi_list:
            print(f'AXI {axi.id}:')
            filepath = os.path.join(axi_dir, f"axi_{axi.id}.txt")
            with open(filepath, 'w') as f:
                for item in axi.data:
                    f.write(f'{item.value}\n')
                    print(f'{item._edge_name}', end=', ')
                print()
            print()

    def write_weights_from_axi(self, data, filename):
        """Write data from each AXI to file."""
        with open(filename, 'w') as f:
            for values in np.transpose(data):
                for item in values:
                    f.write(f'{item}\n')

class MemoryInstruction(object):
    """Memory Interface Instructions.
    TODO  Make this inherit from Component class.

    """

    def __init__(self, op: str, shift_amt: int = 0, nlanes: int = 16):
        self.op = op
        if self.op == 'read':
            self.axi_read_flags = [False, False, False, False]
        else:
            self.shiftamount = shift_amt
            self.lanes = []
            for i in range(nlanes):
                self.lanes.append({
                    "laneid": i,
                    "relpe": 0,
                    "valid": 0
                })

    def set_laneinst_at(self, laneid: int, relpeid: int):
        """Sets the valid flag to True for the given relative PE of the given Lane. This means the shift amount
        specified in the shift instruction will be applied to this Lane.
        """
        self.lanes[laneid]["relpe"] = relpeid
        self.lanes[laneid]["valid"] = 1

    def toDict(self):
        if self.op == 'read':
            d = {
                'op': self.op,
                'axi_read_flags': self.axi_read_flags
            }
        else:
            d = {
                "op": self.op,
                "shift": self.shiftamount,
                "lanes": self.lanes
            }
        return d

    def __str__(self):
        return json.dumps(self.toDict(), sort_keys=False, indent=2)


class MemoryInstructionGenerator(object):
    """Generates MemoryInstructions, given the number of data elements to read from DDR, number of AXI masters, number
    of Lanes, and number of PEs per Lane.

    """
    # Configurations for binary instruction format.
    peid_bits = 2
    valid_bits = 1
    opcode_bits = 2
    shift_bits = 4
    namespace_bits = 2
    axi_read_flag_bits = 4
    total_bits = 56
    nsbin = {
        "instructions": 0,
        "data": 1,
        "weight": 2,
        "meta": 3
    }
    opbin = {
        "read": 0,
        "shift": 1,
        "wfi": 2,
        "loop": 3
    }

    def __init__(self, data, dtype, n_axi: int, n_lanes: int, pes_per_lane: int, architecture: TablaTemplate):
        """
        Maintains a list of MemoryInstruction.

        Parameters
        ----------
        data :
                Data elements to read from DDR.
        n_axi : int
                The number of AXI slaves.
        n_lanes : int
                The number of Lanes.
        pes_per_lane : int
                The number of PEs per Lane.

        """
        self.insts = []

        self.axi_list = [AXI(i) for i in range(n_axi)]
        self.axi_controller = AXIController(self.axi_list, architecture)
        if dtype == Dtype.WEIGHT:
            self.axi_controller.assign_weights_to_pe(data)
        elif dtype == Dtype.DATA:
            self.axi_controller.assign_axi(data)
            self.axi_controller.divide_axi_data_by_cycle()

        # TODO Lane objects should ideally be passed as part of hardware description.
        self.lane_gen = LaneGenerator(architecture, n_lanes, pes_per_lane)
        self.lanes = self.lane_gen.init_lanes()
        self.assign_lanes_to_axi()
        for axi in self.axi_list:
            print(axi)

        self.architecture = architecture

        # AXI read flag for read instruction - keep track of what current ND index should be
        self.current_nd_index = {pe.category_id: 0 for pe in self.architecture.category_component_dict['pe'].values() if isinstance(pe, PE)}

    def assign_lanes_to_axi(self):
        lanes_per_axi = 4
        for axi_index, axi in enumerate(self.axi_list):
            lanes = self.lanes[axi_index * lanes_per_axi : axi_index * lanes_per_axi + lanes_per_axi]
            axi.set_lanes(lanes)

    def gen_single_lane_bin(self, lane):
        peid = lane["relpe"]
        valid = lane["valid"]
        bin_str = format(peid, '0' + str(MemoryInstructionGenerator.peid_bits) + 'b') + format(valid, '0' + str(MemoryInstructionGenerator.valid_bits) + 'b')
        return bin_str

    def gen_axi_read_flags_bin(self, axi_read_flags):
        reverse_order = axi_read_flags[::-1]
        flag_bits_str = ''.join('1' if axi_flag else '0' for axi_flag in reverse_order)
        #return flag_bits_str.zfill(MemoryInstructionGenerator.total_bits - MemoryInstructionGenerator.opcode_bits)
        return flag_bits_str

    def gen_lanes_bin(self, lanes):
        bin = ""
        for lane in reversed(lanes):
            bin += self.gen_single_lane_bin(lane)
        bin += self.gen_ns_bin("data")
        return bin

    def gen_ns_bin(self, ns):
        return format(MemoryInstructionGenerator.nsbin[ns], '0' + str(MemoryInstructionGenerator.namespace_bits) + 'b')

    def gen_opcode_bin(self, op):
        return format(MemoryInstructionGenerator.opbin[op], '0' + str(MemoryInstructionGenerator.opcode_bits) + 'b')

    def gen_shift_bin(self, shift):
        return format(shift, '0' + str(MemoryInstructionGenerator.shift_bits) + 'b')

    def gen_shift_inst(self, shift_amount, affected_lanes, lanes):
        inst = MemoryInstruction("shift", shift_amount)
        for lane in affected_lanes:
            lane_id = lane[0]
            pe_id = lane[1]
            lane_obj = self.lane_gen.get_lane(lanes, lane_id)
            rel_pe_id = lane_obj.get_relpeid(pe_id)
            inst.set_laneinst_at(lane_id, rel_pe_id)
        return inst

    def gen_read_inst(self, batch):
        inst = MemoryInstruction("read")
        num_axi = 4
        lanes_per_axi = 4
        for axi_id in range(num_axi):
            start = axi_id*lanes_per_axi
            end = axi_id*lanes_per_axi + lanes_per_axi
            axi_data = batch[start:end]
            for data in axi_data:
                if data == None:
                    # Means this AXI has completed reading off all data
                    break
                pe = self.architecture.component_map[data.src_component]
                ns = pe.get_namespace('ND')
                index = ns.find_data_index(data.data_id)
                if index != self.current_nd_index[pe.category_id]:
                    break
            else:  # only executed if inner loop did not break
                inst.axi_read_flags[axi_id] = True
                for data in axi_data:
                    pe = self.architecture.component_map[data.src_component]
                    ns = pe.get_namespace('ND')
                    self.current_nd_index[pe.category_id] += 1
                axi = self.axi_controller.axi_list[axi_id]
                axi.data_by_cycle.pop(0)
        return inst

    def gen_wfi_inst(self):
        return MemoryInstruction("wfi")

    def gen_loop_inst(self):
        return MemoryInstruction("loop")

    def gen_inst(self, filepath):
        """Generates MemoryInstructions and writes it to a JSON file. For each data batch, there is:
            - 1 read instruction
            - n shift instructions
        Once all batches have been processed, following instructions are run:
            - 1 wfi instruction
            - 1 loop instruction

        Returns
        -------

        """
        filepath = filepath or "."
        instrs = []
        for cycle in range(self.axi_controller.max_cycle):
            # instrs.append(self.gen_read_inst())
            # data_read = self.axi_controller.get_axi_data_for_cycle(cycle)

            data_read = self.axi_controller.peek_axi_head_data()
            print(f'Cycle {cycle}:')
            data_batch_str = 'Data batch:'
            print(f'{data_batch_str:<16}', end='')
            for item in data_read:
                if item is not None:
                    print(f'{item.value:>4}', end=', ')
                else:
                    value_str = 'None'
                    print(f'{value_str:>4}', end=', ')
            print()
            dest_pe_str = 'Dest PE:'
            print(f'{dest_pe_str:<16}', end='')
            for item in data_read:
                if item is not None:
                    component_map = self.architecture.component_map
                    dest_pe = component_map[item.src_component]
                    dest_pe_id = dest_pe.category_id
                    print(f'{dest_pe_id:>4}', end=', ')
                else:
                    value_str = 'None'
                    print(f'{value_str:>4}', end=', ')
            print()
            dest_lane_str = 'Dest Lane:'
            print(f'{dest_lane_str:<16}', end='')
            for item in data_read:
                if item is not None:
                    component_map = self.architecture.component_map
                    dest_pe = component_map[item.src_component]
                    dest_pe_id = dest_pe.category_id
                    dest_lane_id = self.lane_gen.get_dest_laneid(dest_pe_id)
                    print(f'{dest_lane_id:>4}', end=', ')
                else:
                    value_str = 'None'
                    print(f'{value_str:>4}', end=', ')
            print()
            shift_amount_str = 'Shift Amount:'
            print(f'{shift_amount_str:<16}', end='')
            for curr_lane, item in enumerate(data_read):
                if item is not None:
                    component_map = self.architecture.component_map
                    dest_pe = component_map[item.src_component]
                    dest_pe_id = dest_pe.category_id
                    dest_lane_id = self.lane_gen.get_dest_laneid(dest_pe_id)
                    shift_amount = self.lane_gen.get_shift_amount(curr_lane, dest_lane_id)
                    print(f'{shift_amount:>4}', end=', ')
                else:
                    value_str = 'None'
                    print(f'{value_str:>4}', end=', ')
            print()
            lanes_by_shift = self.lane_gen.get_lanes_by_shift_amount(data_read)
            print(f'Lanes by shift (lane_id, pe_id): {lanes_by_shift}')
            print(f'Num of shifts: {len(lanes_by_shift)}\n')

            # Read instruction
            instrs.append(self.gen_read_inst(data_read))

            for shift_amount in lanes_by_shift:
                affected_lanes = lanes_by_shift[shift_amount]
                inst = self.gen_shift_inst(shift_amount, affected_lanes, self.lanes)
                instrs.append(inst)
        instrs.append(self.gen_wfi_inst())
        instrs.append(self.gen_loop_inst())
        self.insts = instrs

        with open(filepath, 'w') as f:
            f.write(json.dumps([i.toDict() for i in instrs], sort_keys=False, indent=2))

    def gen_binary(self, filepath):
        """Writes binary instructions to a file.

        Note: This should be called after gen_inst() has been called.

        Returns
        -------

        """
        filepath = filepath or "."
        binary = ""
        for inst in self.insts:
            if inst.op == 'read':
                b = self.gen_opcode_bin(inst.op) + \
                    self.gen_axi_read_flags_bin(inst.axi_read_flags)
                b = b.zfill(MemoryInstructionGenerator.total_bits)
            else:
                b = self.gen_lanes_bin(inst.lanes) + self.gen_opcode_bin(inst.op) + self.gen_shift_bin(
                    inst.shiftamount)
            binary += b + '\n'

        with open(filepath, 'w') as f:
            f.write(binary)


class WeightConfigGenerator(object):
    """Generates meta information on weights that's needed for RTL.

    TODO It is still unclear how the config file generated by this class is used in RTL.
    """
    def __init__(self, architecture: TablaTemplate):
        self.architecture = architecture
        self.n_lanes = self.architecture.config['num_lanes']
        self.pes_per_lane = self.architecture.config['pes_per_lane']

    def gen_weightconf(self, wnodes, filepath):
        """Generates weight config file, based on the wnodes from DFG.

        TODO `wnodes` needs to be refactored so that it works with Tabla v2 backend.
        """
        lc = [[0 for col in range(self.pes_per_lane)]
              for row in range(self.n_lanes)]
        for wnode in wnodes:
            component_map = self.architecture.component_map
            dest_pe = component_map[wnode.src_component]
            dest_pe_id = dest_pe.category_id
            laneid = dest_pe_id % self.n_lanes
            pe_offset = dest_pe_id // self.n_lanes
            lc[laneid][pe_offset] += 1
        # print(lc)
        s = self._gen_templ(lc)

        with open(filepath, "w") as f:
            f.write(s)

    def _gen_templ(self, lc):
        define = "`define"
        wcpe = "WEIGHT_COUNT_PE_"
        wcl = "WEIGHT_COUNT_LANE_"
        count = "16'h"
        WEIGHT_COUNT = "//WEIGHT_COUNT"
        LANE = "//LANE{:d}"

        linef_pe = define + " " + wcpe + "{:d}" + " " + count + "{:x}"
        linef_lane = define + " " + wcl + "{:d}" + " " + count + "{:x}"

        s = ""
        s += WEIGHT_COUNT
        s += "\n"
        for lid, lane in enumerate(lc):
            lane_count = 0
            s += LANE.format(lid)
            s += "\n"
            for pe_offset, pe_count in enumerate(lane):
                peid = pe_offset * self.n_lanes + lid
                s += linef_pe.format(peid, pe_count)
                s += "\n"
                lane_count += pe_count
            s += linef_lane.format(lid, lane_count)
            s += "\n"
        return s


def print_pe_assignments(data_nodes, architecture):
    for node in data_nodes:
        pe = architecture.component_map[node.src_component]
        ns = pe.get_namespace(node.namespace_name)
        try:
            index = ns.find_data_index(node.data_id)
        except AssertionError:
            print(ns.ns_items.keys())
            print(f'Data id = {node.data_id}')
            exit()
        print(f"{node._edge_name}, PE {find_dest_pe_id(architecture, node.src_component)}, val = {node.value}, data_id = {node.data_id}, index = {index}")
    print()


def find_dest_pe_id(architecture, comp_id):
    component_map = architecture.component_map
    dest_pe = component_map[comp_id]
    dest_pe_id = dest_pe.category_id
    return dest_pe_id


def get_input_weight_nodes(schedule, architecture):
    weights = {}
    for edge in schedule._dfg_edges:
        if edge.is_src_edge and edge.namespace_name == 'NW':
            if edge._edge_name not in weights:
                weights[edge._edge_name] = edge
                #print(f"{edge._edge_name}, {edge.src_component}, {edge.namespace_name}, {edge.dtype}, {edge.is_src_edge}")
    weights = list(weights.values())
    # print_weights(weights, architecture)
    return weights


def get_input_data_nodes(schedule):
    input_data = {}
    for edge in schedule._dfg_edges:
        if edge.is_src_edge and edge.namespace_name == 'ND':
            if edge._edge_name not in input_data:
                input_data[edge._edge_name] = edge
                #print(f"{edge._edge_name}, {edge.src_component}, {edge.namespace_name}, {edge.dtype}, {edge.is_src_edge}")
    input_data = list(input_data.values())
    return input_data


def get_input_meta_nodes(schedule):
    meta = {}
    for edge in schedule._dfg_edges:
        if edge.is_src_edge and edge.namespace_name == 'NM':
            if edge.src_component not in meta:
                meta[edge.src_component] = edge
    meta = list(meta.values())
    return meta


def write_to_input_file(data, filepath):
    with open(filepath, 'w') as f:
        for item in data:
            f.write(str(item))
            f.write('\n')


def write_pe_files(data_file, data, mem_interface_artifacts_dir, architecture, dtype):
    """Given input data list (NW or ND), write each data point to corresponding PE file."""

    data_values = []
    with open(data_file, 'r') as f:
        for index, line in enumerate(f):
            data_point = int(line)
            data_values.append(data_point)
    #print(data_values)

    pes = {}
    for index, node in enumerate(data):
        pe = architecture.component_map[node.src_component]
        pe_id = pe.category_id
        data_point = data_values[index]
        if pe_id not in pes:
            pes[pe_id] = [data_point]
        else:
            pes[pe_id].append(data_point)
    # print(pes)

    pe_dir = os.path.join(mem_interface_artifacts_dir, "pe")
    if not os.path.exists(pe_dir):
        os.makedirs(pe_dir)

    if dtype == Dtype.DATA:
        pe_weight_data_dir = os.path.join(pe_dir, "input_data")
        if not os.path.exists(pe_weight_data_dir):
            os.makedirs(pe_weight_data_dir)
    elif dtype == Dtype.WEIGHT:
        pe_weight_data_dir = os.path.join(pe_dir, "weight")
        if not os.path.exists(pe_weight_data_dir):
            os.makedirs(pe_weight_data_dir)

    for pe_id in pes:
        filename = os.path.join(pe_weight_data_dir, f"pe_{pe_id}.txt")
        with open(filename, 'w') as f:
            for item in pes[pe_id]:
                f.write(str(item) + '\n')


def set_values_to_nodes(data_file, data_nodes):
    data_values = []
    with open(data_file, 'r') as f:
        for index, line in enumerate(f):
            data_point = int(line)
            data_values.append(data_point)

    for index, node in enumerate(data_nodes):
        data_point = data_values[index]
        node.value = data_point


class MetadataLoadGenerator(object):
    def __init__(self, architecture):
        self.architecture = architecture

    def generate_pe_instructions(self, schedule: Schedule, arch: TablaTemplate, filename, debug="values"):
        if not schedule.is_dfg_loaded():
            raise RuntimeError(f"Schedule has not loaded a DFG yet.")
        pes = [pe for _, pe in arch.category_component_dict["pe"].items() if isinstance(pe, PE)]
        config = self.compute_instr_widths(arch.config, debug=debug)
        address_width = int(np.ceil(np.log2(arch.max_instr)))

        pe_blocks = ["generate\n"]
        max_pe = -1
        for pe_id, pe in enumerate(pes):
            if pe.component_id <= max_pe:
                raise RuntimeError(f"Not iterating over components in correct sequence")
            else:
                max_pe = pe.component_id
            pe_str = f"if(peId == {pe_id}) begin\n" \
                     f"\talways @(*) begin\n" \
                     f"\t\tcase(address)\n"
            pe_meta_nodes = pe.meta_nodes
            for index, meta_node in enumerate(pe_meta_nodes):
                bin_str = f"\t\t\t{address_width}'d{index} : rdata = {config['instr_len']}'b{np.binary_repr(meta_node.value, width=config['instr_len'])};\n"
                pe_str += bin_str
            pe_str += f"\t\t\tdefault : rdata = {config['instr_len']}'b{np.binary_repr(0, width=config['instr_len'])};\n"
            pe_str += f"\t\tendcase\n"
            pe_str += f"\tend\n"
            pe_str += f"end\n"
            pe_str += f"\n"
            pe_blocks.append(pe_str)
        pe_blocks.append("endgenerate")
        self.write_instr_file(pe_blocks, filename)

    def get_data_values(self, instr, schedule):
        src_vals = []
        assert instr.node_id in schedule.dfg_node_map
        instr_node = schedule.dfg_node_map[instr.node_id]
        for idx, i in enumerate(instr.srcs):
            if i.data_id in schedule.dfg_node_map and i.location != "ALU":
                src_vals.append(str(schedule.dfg_node_map[i.data_id].computed))
            elif instr_node.parents[idx] in schedule.dfg_node_map:
                src_vals.append(str(schedule.dfg_node_map[instr_node.parents[idx]].computed))
            else:
                print(f"Instruction source without node: {i.data_id} \t {instr}")

        src_annotation = ", ".join(src_vals)
        dest_annotation = str(schedule.dfg_node_map[instr.node_id].computed)

        return src_annotation, dest_annotation

    def compute_instr_widths(self, config, debug=False):
        op_index_bits = int(np.log2(max([config["namespace_size"], config["namespace_interim_size"]])))
        pe_bus_width = int(np.log2(config["pes_per_pu"])) + 1
        pu_bus_width = int(np.log2(config["num_pes"] / config["pes_per_pu"])) + 1
        instr_len = OP_WIDTH + (
                op_index_bits * 4) + OP_SELECT_WIDTH * 2 \
                    + MEM_INTERFACE_WIDTH + BUS_WIDTH + pu_bus_width + pe_bus_width
        new_sum = OP_WIDTH + 2 * (op_index_bits + OP_SELECT_WIDTH) + 2 * (
                    1 + op_index_bits) + 2 + pe_bus_width + pu_bus_width
        if debug:
            print(f"Calc:\n"
                  f"\tOp code: {OP_WIDTH}\n"
                  f"\tOp1: {op_index_bits + OP_SELECT_WIDTH}\n"
                  f"\tOp2: {op_index_bits + OP_SELECT_WIDTH}\n"
                  f"\tNI: {1 + op_index_bits}\n"
                  f"\tNS: {1 + op_index_bits}\n"
                  f"\tNieghbors: {2}\n"
                  f"\tPEGB: {pe_bus_width}\n"
                  f"\tPUGB: {pu_bus_width}\n"
                  f"prev sum: {instr_len}\n"
                  f"New: {new_sum}")
        return {"op_index_bits": op_index_bits, "pe_bus_width": pe_bus_width,
                "pu_bus_width": pu_bus_width, "instr_len": instr_len}

    def write_instr_file(self, pe_instrs, filename):
        with open(filename, "w") as f:
            f.write("".join(pe_instrs))

    def assign_meta_to_pe(self, meta_nodes):
        for meta_node in meta_nodes:
            # Find destination PE
            component_map = self.architecture.component_map
            dest_pe = component_map[meta_node.src_component]

            # Put the node in the PE
            dest_pe.meta_nodes.append(meta_node)


class Dtype(Enum):
    DATA = 1
    WEIGHT = 2


def generate_memory_instr(basepath, schedule, architecture, config, input_data_file, weight_file, meta_file, debug=False):
    mem_dir = f"{basepath}/mem-inst"
    # weight_file = f"{basepath}/mem-inst/input_weights.txt"
    # input_data_file = f"{basepath}/mem-inst/input_data.txt"
    # meta_file = f"{basepath}/mem-inst/meta.txt"
    axi_dir = f"{basepath}/mem-inst/axi"
    axi_input_data_dir = f"{axi_dir}/input_data"
    axi_weight_dir = f"{axi_dir}/weights"


    if not os.path.exists(axi_dir):
        os.makedirs(axi_dir)
    if not os.path.exists(axi_weight_dir):
        os.makedirs(axi_weight_dir)
    print(f'Generating instructions for Weight Data')
    # Get a list of weights (DFG nodes)
    weights = get_input_weight_nodes(schedule, architecture)

    # Set data values to DFG nodes
    set_values_to_nodes(weight_file, weights)

    print_pe_assignments(weights, architecture)

    # Print weight to PE assignment for each AXI
    meminst_gen = MemoryInstructionGenerator(weights, Dtype.WEIGHT, config['num_axi'], config['num_lanes'],
                                             config['pes_per_lane'], architecture)
    meminst_gen.axi_controller.print_axi_contents()

    # Put placeholder values for each AXI and write to AXI files
    print('After filling in placeholders')
    for axi_index in range(config['num_axi']):
        weight_data = meminst_gen.axi_controller.gen_matrix_for_axi(axi_index)
        max_num_weights = meminst_gen.axi_controller.find_max_number_of_weights(weight_data)
        for lane in weight_data:
            if len(lane) < max_num_weights:
                num_placeholders = max_num_weights - len(lane)
                lane.extend([0 for _ in range(num_placeholders)])
        weight_data = np.array(weight_data)
        print(f'AXI {axi_index}')
        print(weight_data)
        print()
        meminst_gen.axi_controller.write_weights_from_axi(weight_data,
                                                          os.path.join(axi_weight_dir, f'axi_{axi_index}.txt'))

    # TODO Write weights to corresponding PE files
    # write_pe_files(weight_file, weights, mem_interface_artifacts_dir, architecture, Dtype.WEIGHT)

    # Generate weight config file (weightInst.txt)
    wconf_gen = WeightConfigGenerator(architecture)
    wconf_gen.gen_weightconf(weights, os.path.join(mem_dir, 'weight_insts.txt'))

    print(f'Generating instructions for Input Data')
    # Get a list of input data (DFG nodes)
    input_data_nodes = get_input_data_nodes(schedule)
    print(len(input_data_nodes))
    # Set data values to DFG nodes
    set_values_to_nodes(input_data_file, input_data_nodes)
    print_pe_assignments(input_data_nodes, architecture)

    meminst_gen = MemoryInstructionGenerator(input_data_nodes, Dtype.DATA, config['num_axi'], config['num_lanes'],
                                             config['pes_per_lane'], architecture)
    # Write AXI data to file
    axi_input_data_dir = os.path.join(axi_dir, 'input_data')
    if not os.path.exists(axi_input_data_dir):
        os.makedirs(axi_input_data_dir)
    meminst_gen.axi_controller.write_axi_data(axi_input_data_dir)

    # Generate memory instructions
    meminst_gen.gen_inst(os.path.join(mem_dir, 'meminst.json'))
    meminst_gen.gen_binary(os.path.join(mem_dir, 'meminst.txt'))

    # Write input data to corresponding PE files
    write_pe_files(input_data_file, input_data_nodes, mem_dir, architecture, Dtype.DATA)

    print(f'Generating Verilog file for metadata')
    # Generate Verilog files for loading metadata
    meta_nodes = get_input_meta_nodes(schedule)

    # Set meta value to meta nodes
    with open(meta_file, 'r') as f:
        meta_data = int(f.readlines()[0])
    for meta in meta_nodes:
        meta.value = meta_data
    print_pe_assignments(meta_nodes, architecture)

    meta_gen = MetadataLoadGenerator(architecture)
    meta_gen.assign_meta_to_pe(meta_nodes)
    meta_loader = os.path.join(mem_dir, 'meta.v')
    meta_gen.generate_pe_instructions(schedule, architecture, meta_loader)


def main(args):
    with open(args.config_file) as config_file:
        config = json.load(config_file)

    # Instantiate an architecture for scheduling
    architecture = TablaTemplate(config)
    schedule = Schedule()
    schedule.load_dfg(args.dfg_file)
    schedule.schedule_graph(architecture)

    print(f'Generating instructions for Weight Data')
    # Get a list of weights (DFG nodes)
    weights = get_input_weight_nodes(schedule, architecture)
    weight_file = args.weight_file
    if weight_file is None:
        weight_file = "input_weights.txt"
        n_data_points = len(weights)
        #weight_data = [n for n in range(n_data_points)]
        weight_data = np.random.randint(0, 5, n_data_points)
        write_to_input_file(weight_data, weight_file)

    # Set data values to DFG nodes
    set_values_to_nodes(weight_file, weights)
    print_pe_assignments(weights, architecture)

    # Print weight to PE assignment for each AXI
    meminst_gen = MemoryInstructionGenerator(weights, Dtype.WEIGHT, config['num_axi'], config['num_lanes'],
                                             config['pes_per_lane'], architecture)
    meminst_gen.axi_controller.print_axi_contents()

    # Prepare directories to write outputs in
    mem_interface_artifacts_dir = args.output_directory
    if not os.path.exists(mem_interface_artifacts_dir):
        os.makedirs(mem_interface_artifacts_dir)
    axi_dir = os.path.join(mem_interface_artifacts_dir, 'axi')
    if not os.path.exists(axi_dir):
        os.makedirs(axi_dir)
    axi_weight_dir = os.path.join(axi_dir, 'weights')
    if not os.path.exists(axi_weight_dir):
        os.makedirs(axi_weight_dir)

    # Put placeholder values for each AXI and write to AXI files
    print('After filling in placeholders')
    for axi_index in range(config['num_axi']):
        weight_data = meminst_gen.axi_controller.gen_matrix_for_axi(axi_index)
        max_num_weights = meminst_gen.axi_controller.find_max_number_of_weights(weight_data)
        for lane in weight_data:
            if len(lane) < max_num_weights:
                num_placeholders = max_num_weights - len(lane)
                lane.extend([0 for _ in range(num_placeholders)])
        weight_data = np.array(weight_data)
        print(f'AXI {axi_index}')
        print(weight_data)
        print()
        meminst_gen.axi_controller.write_weights_from_axi(weight_data,
                                                          os.path.join(axi_weight_dir, f'axi_{axi_index}.txt'))

    # TODO Write weights to corresponding PE files
    # write_pe_files(weight_file, weights, mem_interface_artifacts_dir, architecture, Dtype.WEIGHT)

    # Generate weight config file (weightInst.txt)
    wconf_gen = WeightConfigGenerator(architecture)
    wconf_gen.gen_weightconf(weights, os.path.join(mem_interface_artifacts_dir, 'weight_insts.txt'))

    print(f'Generating instructions for Input Data')
    # Get a list of input data (DFG nodes)
    input_data_nodes = get_input_data_nodes(schedule)
    input_data_file = args.input_data_file
    if input_data_file is None:
        input_data_file = "input_data.txt"
        n_data_points = len(input_data_nodes)
        #input_data = [n for n in range(n_data_points)]
        input_data = np.random.randint(0, 5, 784)
        input_data = np.append(input_data, [500])
        write_to_input_file(input_data, input_data_file)

    # Set data values to DFG nodes
    set_values_to_nodes(input_data_file, input_data_nodes)
    print_pe_assignments(input_data_nodes, architecture)

    meminst_gen = MemoryInstructionGenerator(input_data_nodes, Dtype.DATA, config['num_axi'], config['num_lanes'],
                                             config['pes_per_lane'], architecture)
    # Write AXI data to file
    axi_input_data_dir = os.path.join(axi_dir, 'input_data')
    if not os.path.exists(axi_input_data_dir):
        os.makedirs(axi_input_data_dir)
    meminst_gen.axi_controller.write_axi_data(axi_input_data_dir)

    # Generate memory instructions
    meminst_gen.gen_inst(os.path.join(mem_interface_artifacts_dir, 'meminst.json'))
    meminst_gen.gen_binary(os.path.join(mem_interface_artifacts_dir, 'meminst.txt'))

    # Write input data to corresponding PE files
    write_pe_files(input_data_file, input_data_nodes, mem_interface_artifacts_dir, architecture, Dtype.DATA)

    print(f'Generating Verilog file for metadata')
    # Generate Verilog files for loading metadata
    meta_nodes = get_input_meta_nodes(schedule)
    meta_file = args.meta_file
    if meta_file is None:
        meta_file = 'meta.txt'
        meta_data = [1]
        write_to_input_file(meta_data, meta_file)

    # Set meta value to meta nodes
    for meta in meta_nodes:
        meta.value = meta_data[0]
    print_pe_assignments(meta_nodes, architecture)

    meta_gen = MetadataLoadGenerator(architecture)
    meta_gen.assign_meta_to_pe(meta_nodes)
    meta_loader = os.path.join(mem_interface_artifacts_dir, 'meta.v')
    meta_gen.generate_pe_instructions(schedule, architecture, meta_loader)


# TODO Add a module to generate: config.list, weightInst.txt, active_pes.json, special_modules.json, inst_info.txt

if __name__ == "__main__":
    compiler_rootdir = os.path.dirname(os.path.abspath(__file__)).rsplit("/", 1)[0]
    dfg_dir = os.path.join(compiler_rootdir, 'tests/test_dfgs')
    dfg_file_default = os.path.join(dfg_dir, "linear_dfg.json")
    config_file_default = os.path.join(compiler_rootdir, 'tests/config.json')
    input_weights_file_dir = os.path.join(config_file_default, 'tests/inputs')
    mem_interface_artifacts_dir = os.path.join(compiler_rootdir, 'backend/mem_interface_artifacts')

    argparser = argparse.ArgumentParser(description='Memory Interface Instructino Generator')
    argparser.add_argument('-d', '--dfg_file',
                           default=dfg_file_default,
                           help='DFG file (default: project.rtml/tablav2/tests/test_dfgs/linear_dfg.json')
    argparser.add_argument('-c', '--config_file',
                           default=config_file_default,
                           help='config.json file (default: project.rtml/tablav2/tests/config.json')
    argparser.add_argument('-w', '--weight_file',
                           help='weights.txt file')
    argparser.add_argument('-i', '--input_data_file',
                           help='data.txt file')
    argparser.add_argument('-m', '--meta_file',
                           help='Metadata (learning rate) file')
    argparser.add_argument('-o', '--output_directory',
                           default=mem_interface_artifacts_dir,
                           help='Directory where all memory-related files are written.')
    args = argparser.parse_args()

    main(args)
