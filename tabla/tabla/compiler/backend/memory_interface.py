import copy
import json
import os
import math
from typing import List
from numpy.random import randint
import numpy as np

import argparse
from enum import Enum
import pickle
from . import ScheduleNode, Schedule
from . import TablaTemplate
from . import OP_SELECT_WIDTH, OP_WIDTH, MEM_INTERFACE_WIDTH, BUS_WIDTH
from . import PE
import pprint
from collections import defaultdict


class Lane(object):
    """
    A Lane is a memory interface component that connects a set of PEs together. Once data is read through AXI, it is
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
    """
    Given the total number of lanes and PEs per Lane, generates Lane objects accordingly.
    """

    def __init__(self, architecture: TablaTemplate, nlanes: int = 16, pes_per_lane: int = 4, n_axi = 4):
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
        self.n_axi = n_axi
        self.lanes_per_axi = int(self.nlanes / self.n_axi)

    def init_lanes(self):
        """
        Creates Lane objects.
        """
        lanes = []
        for lane_id in range(self.nlanes):
            # IDs of PEs that belong to this Lane
            pe_ids = []
            for i in range(self.pes_per_lane):
                pe_id = i * self.nlanes + lane_id
                pe_ids.append(pe_id)
            lane = Lane(lane_id, pe_ids)
            lanes.append(lane)
        return lanes

    def get_lanes_by_shift_amount(self, batch: List, axi_read_flags: List):
        """
        Given a batch, figure out shift amounts for each Lane, and group these by shift amount.
        """
        lanes_by_shift = {}
        for curr_lane, data in enumerate(batch):
            if data is not None:
                component_map = self.architecture.component_map
                dest_pe = component_map[data.src_component]
                dest_pe_id = dest_pe.category_id
                dest_lane_id = self.get_dest_laneid(dest_pe_id)
                shift_amount = self.get_shift_amount(curr_lane, dest_lane_id)

                axi_id = curr_lane // self.lanes_per_axi

                ns = dest_pe.get_namespace(data.namespace_name)
                index = ns.find_data_index(data.data_id)

                # print(f'AXI ID: {axi_id}, Read flag: {axi_read_flags[axi_id]}, curr lane: {curr_lane}, data: {data._edge_name}')
                if axi_read_flags[axi_id] is True:
                    if shift_amount in lanes_by_shift:
                        lanes_by_shift[shift_amount].append((dest_lane_id, dest_pe_id, index))
                    else:
                        lanes_by_shift[shift_amount] = [(dest_lane_id, dest_pe_id, index)]
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

    def __init__(self, id, axi_size=64, axi_read_cycle: int = 4):
        self.id = id
        # these two variables determine how many read instructions are required
        self.axi_size = axi_size  # number of data elements read by each AXI in a single burst
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
    """
    Reads data from FPGA DDR through AXI.
    """

    def __init__(self, axi_list, architecture, debug=False):
        self.axi_list = axi_list
        # TODO The following two lines are too ad-hoc.
        self.axi_size = self.axi_list[0].axi_size
        self.axi_read_cycle = self.axi_list[0].axi_read_cycle
        self.architecture = architecture
        self.debug = debug

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
        """Finds maximum number of weights from all Lanes.
        TODO We can shorten this with max(lanes, key=lambda x: len(x))
        """
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
        """Groups AXI data by cycle. Every AXI cna read 4 data elements at a time."""
        for axi in self.axi_list:
            cycls = len(axi.data) // self.axi_read_cycle
            r = len(axi.data) % self.axi_read_cycle
            for i in range(cycls):

                axi.data_by_cycle.append(axi.data[i * self.axi_read_cycle: i * self.axi_read_cycle + self.axi_read_cycle])
            if self.debug:
                print(f"Axi data length: {len(axi.data)}\n"
                      f"R val: {r}\n"
                      f"Index: {(i+1) * self.axi_read_cycle}\n"
                      f"I: {i}")
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
                # If less than 4 values are read, fill in with None
                if len(head_data) < 4:
                    fill_count = 4 - len(head_data)
                    head_data.extend([None] * fill_count)
                batch.extend(head_data)
        return batch

    def write_axi_data(self, axi_dir):
        for axi in self.axi_list:
            if self.debug:
                print(f'AXI {axi.id}:')
            filepath = os.path.join(axi_dir, f"axi_{axi.id}.txt")
            with open(filepath, 'a') as f:
                for item in axi.data:
                    f.write(f'{item.value}\n')
                    if self.debug:
                        print(f'{item._edge_name}', end=', ')


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

    def __init__(self, op: str, cycle: int, batch_count: int, shift_amt: int = 0, nlanes: int = 16):
        self.op = op
        self.cycle = cycle
        self.batch_count = batch_count
        if self.op == 'read':
            self.axi_read_flags = [False, False, False, False]
        else:
            self.shiftamount = shift_amt
            self.lanes = []
            for i in range(nlanes):
                self.lanes.append({
                    "laneid": i,
                    "relpe": 0,
                    "valid": 0,
                    "dest_pe": None,
                    "dest_ns_index": None
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
                'cycle': self.cycle,
                'batch_count': self.batch_count,
                'axi_read_flags': self.axi_read_flags
            }
        else:
            d = {
                "op": self.op,
                "cycle": self.cycle,
                'batch_count': self.batch_count,
                "shift": self.shiftamount,
                "lanes": self.lanes
            }
        return d

    def __str__(self):
        return json.dumps(self.toDict(), sort_keys=False, indent=2)


class MemoryInstructionGenerator(object):
    """
    Generates MemoryInstruction's, given the number of data elements to read from DDR, number of AXI masters, number
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

    def __init__(self, data, dtype, n_axi: int, n_lanes: int, pes_per_lane: int, architecture: TablaTemplate, debug=False):
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
        self.debug = debug
        self.insts = []

        # AXI objects
        self.axi_list = [AXI(i) for i in range(n_axi)]
        self.axi_controller = AXIController(self.axi_list, architecture, debug=debug)
        if dtype == Dtype.WEIGHT:
            self.axi_controller.assign_weights_to_pe(data)
        elif dtype == Dtype.DATA:
            self.axi_controller.assign_axi(data)
            self.axi_controller.divide_axi_data_by_cycle()

        # TODO Lane objects should ideally be passed as part of hardware description.
        self.lane_gen = LaneGenerator(architecture, n_lanes, pes_per_lane, n_axi)

        # Lane objects with corresponding PEs
        self.lanes = self.lane_gen.init_lanes()

        # Number of Lane objects per AXI
        # NOTE only multiples of 2 allowed
        self.lanes_per_axi = int(n_lanes / n_axi)
        self.num_axi = n_axi
        self.assign_lanes_to_axi()
        if self.debug:
            for axi in self.axi_list:
                print(axi)

        self.architecture = architecture

        # AXI read flag for read instruction - keep track of what current ND index should be
        self.current_nd_index = {pe.category_id: 0 for pe in self.architecture.category_component_dict['pe'].values() if isinstance(pe, PE)}

    def assign_lanes_to_axi(self):
        for axi_index, axi in enumerate(self.axi_list):
            lanes = self.lanes[axi_index * self.lanes_per_axi : axi_index * self.lanes_per_axi + self.lanes_per_axi]
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

    def gen_shift_inst(self, shift_amount, affected_lanes, lanes, cycle, batch_count):
        inst = MemoryInstruction("shift", cycle, batch_count, shift_amount)
        for lane in affected_lanes:
            lane_id = lane[0]
            pe_id = lane[1]
            ns_index = lane[2]
            lane_obj = self.lane_gen.get_lane(lanes, lane_id)
            rel_pe_id = lane_obj.get_relpeid(pe_id)
            inst.set_laneinst_at(lane_id, rel_pe_id)
            inst.lanes[lane_id]["dest_pe"] = pe_id
            inst.lanes[lane_id]["dest_ns_index"] = ns_index
        return inst

    # def gen_read_inst(self, batch, cycle, batch_count):
    #     inst = MemoryInstruction("read", cycle, batch_count)
    #
    #     axi_read_flags = self.determine_axi_read_flags(batch)
    #     print(f'AXI read flags: {axi_read_flags}')
    #     for axi_id in range(self.num_axi):
    #         # If flag is True:
    #         if axi_read_flags[axi_id]:
    #             inst.axi_read_flags[axi_id] = True
    #             # Update PE namespace next expected indices
    #             # pop values
    #
    # def determine_axi_read_flags(self, batch):
    #     flags = [False] * self.num_axi
    #
    #     # First criteria: check if AXI has completed reading off all data
    #     for axi_id in range(self.num_axi):
    #         axi_data = self._get_axi_data_batch(batch, axi_id)
    #         if axi_data == [None] * self.lanes_per_axi:
    #             flags[axi_id] = False
    #
    #     # Second criteria: If at least one data point's index does not match expected index value, set this AXI flag to False
    #     for axi_id in range(self.num_axi):
    #         axi_data = self._get_axi_data_batch(batch, axi_id)
    #         for data in axi_data:
    #             print(data)
    #             if data is not None:
    #                 pe = self.architecture.component_map[data.src_component]
    #                 ns = pe.get_namespace('ND')
    #                 index = ns.find_data_index(data.data_id)
    #                 # If this data point's destination index does not match currently expected index value, we need to
    #                 # set the AXI read flag to False
    #                 if index != self.current_nd_index[pe.category_id]:
    #                     print(f'data: {data._edge_name}, index: {index}, current index: {self.current_nd_index[pe.category_id]}')
    #                     print()
    #                     flags[axi_id] = False
    #
    #
    #     return flags
    #
    # def _get_axi_data_batch(self, batch, axi_id):
    #     start = axi_id * self.lanes_per_axi
    #     end = axi_id * self.lanes_per_axi + self.lanes_per_axi
    #     axi_data = batch[start:end]
    #     return axi_data

    def _get_dest_pe_id(self, data_point):
        component_map = self.architecture.component_map
        dest_pe = component_map[data_point.src_component]
        dest_pe_id = dest_pe.category_id
        return dest_pe_id

    def _get_dest_pe_index(self, shift_inst, batch):
        shift = shift_inst.shiftamount

        data_points_with_shift_amount = []
        for curr_lane, data in enumerate(batch):
            if data is not None:
                component_map = self.architecture.component_map
                dest_pe = component_map[data.src_component]
                dest_pe_id = dest_pe.category_id

                dest_lane_id = self.lane_gen.get_dest_laneid(dest_pe_id)
                shift_amount = self.lane_gen.get_shift_amount(curr_lane, dest_lane_id)
                if shift_amount == shift:
                    data_points_with_shift_amount.append(data)

        for data_point in data_points_with_shift_amount:
            for data in batch:
                if data is None:
                    continue
                if data_point == data:
                    continue
                if self._get_dest_pe_id(data) == self._get_dest_pe_id(data_point):
                    pe = self.architecture.component_map[data_point.src_component]
                    ns = pe.get_namespace(data_point.namespace_name)
                    index = ns.find_data_index(data_point.data_id)

                    return index
        return -1


    def _get_shift_tuples(self, shift_inst, batch, read_inst):
        shift = shift_inst.shiftamount

        data_points_with_shift_amount = []
        all_tuples = []
        read_len = len(read_inst.axi_read_flags)
        for curr_lane, data in enumerate(batch):
            if data is not None:
                component_map = self.architecture.component_map
                dest_pe = component_map[data.src_component]
                dest_pe_id = dest_pe.category_id

                dest_lane_id = self.lane_gen.get_dest_laneid(dest_pe_id)
                shift_amount = self.lane_gen.get_shift_amount(curr_lane, dest_lane_id)
                reading_val = True if read_inst.axi_read_flags[curr_lane // read_len ] == 1 else False
                if shift_amount == shift:
                    data_points_with_shift_amount.append(data)
                    ns = dest_pe.get_namespace(data.namespace_name)
                    index = ns.find_data_index(data.data_id)
                    all_tuples.append((dest_pe_id, index, shift, reading_val))

        return all_tuples

    def sort_shifts(self, shift_tuples, cycle):
        from itertools import permutations
        conditions = []
        shifts = list(set([i[2] for i in shift_tuples]))
        pe_groupings = defaultdict(list)
        debug_conds = []
        for p in shift_tuples:
            if p not in pe_groupings[p[0]]:
                pe_groupings[p[0]].append(p)
        groupings = []
        for pe_id, value in pe_groupings.items():
            g_sorted = sorted(value, key=lambda j: j[1])
            for idx, shift in enumerate(g_sorted):
                if not shift[3]:
                    continue
                if idx > 0:
                    a = shift[2]
                    b = g_sorted[idx - 1][2]
                    groupings.append((a, b))
                    conditions.append((lambda x: x.index(a) > x.index(b)))
                    t = (lambda x: (x.index(a), x.index(b), a, b))
                    debug_conds.append((f"{a} > {b}", t))

        perms = list(permutations(shifts))

        cond_fn = lambda x, a1, a2: x.index(a1) > x.index(a2)
        for p in perms:
            works = True
            if not all([cond_fn(p, c[0], c[1]) for c in groupings]):
                works = False
            if works:
                return p, True
        print(f"Didnt find a working combination in cycle: {cycle}\n"
              f"Shift tuples: {shift_tuples}\n"
              f"Shifts: {shifts}\n"
              f"")
        return shifts, False

    def reorder_shifts(self, shift_insts, batch, read_inst, cycle=-1):

        shift_tuples = []
        for idx, s in enumerate(shift_insts):
            shift_tuples += self._get_shift_tuples(s, batch, read_inst)

        sorted_shifts, valid_order = self.sort_shifts(shift_tuples, cycle)
        return sorted(shift_insts, key=lambda inst: sorted_shifts.index(inst.shiftamount)), valid_order

    def split_shift_instructions(self, shift_insts, valid_order, data_batch):
        # if not valid_order:

        # TODO change this back to if not
        if not valid_order:

            # order = {}
            # for inst in shift_insts:
            #     print(inst)
            #     shift_amount = inst.shiftamount
            #     valid_lanes = []
            #     for lane in inst.lanes:
            #         if lane['valid'] is 1:
            #             valid_lanes.append((lane['dest_pe'], lane['dest_ns_index']))

            #     order[shift_amount] = valid_lanes

            # print(order)

            # exit()
            return shift_insts
        else:
            return shift_insts

    def gen_read_inst(self, batch, cycle, batch_count):
        inst = MemoryInstruction("read", cycle, batch_count)
        for axi_id in range(self.num_axi):
            start = axi_id * self.lanes_per_axi
            end = axi_id * self.lanes_per_axi + self.lanes_per_axi
            axi_data = batch[start:end]
            # If an AXI has completed reading off all data, then skip
            if axi_data == [None] * self.lanes_per_axi:
                continue
            for i, data in enumerate(axi_data):
                # print(data)
                if data is not None:
                    pe = self.architecture.component_map[data.src_component]
                    ns = pe.get_namespace('ND')
                    index = ns.find_data_index(data.data_id)
                    # If this data point's destination index does not match currently expected index value, we need to
                    # set the AXI read flag to False

                    if index != self.current_nd_index[pe.category_id]:
                        # print('Data index and PE ND index mismatch:')
                        # print(f'data: {data._edge_name}, index: {index}, current index: {self.current_nd_index[pe.category_id]}')

                        # Decrement the ND index values of previous PEs
                        for j in range(i):
                            data_point = axi_data[j]
                            pe = self.architecture.component_map[data_point.src_component]
                            self.current_nd_index[pe.category_id] -= 1
                        break

                    self.current_nd_index[pe.category_id] += 1
            else:  # only executed if inner loop did not break
                inst.axi_read_flags[axi_id] = True
                for data in axi_data:
                    if data is not None:
                        pe = self.architecture.component_map[data.src_component]
                        #ns = pe.get_namespace('ND')
                        #self.current_nd_index[pe.category_id] += 1
                axi = self.axi_controller.axi_list[axi_id]
                popped = axi.data_by_cycle.pop(0)
                # print('Popped data:')
                # for d in popped:
                #     if d is not None:
                #         print(f'{d._edge_name}', end=', ')
                #     else:
                #         print(None, end=', ')
                # print()
        return inst

    def gen_wfi_inst(self, cycle, batch_count):
        return MemoryInstruction("wfi", cycle, batch_count)

    def gen_loop_inst(self, cycle, batch_count):
        return MemoryInstruction("loop", cycle, batch_count)

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
        batches = []
        lanes_by_shifts_list = []
        #for cycle in range(self.axi_controller.max_cycle):
        cycle = 0
        batch_count = 0
        nshifts = -1
        stopping = 0
        cond = 10
        while not self.axi_controller.peek_axi_head_data() == [None] * self.lanes_per_axi * self.num_axi:
            # instrs.append(self.gen_read_inst())
            #data_read = self.axi_controller.get_axi_data_for_cycle(cycle)
            stopping +=1
            data_read = self.axi_controller.peek_axi_head_data()
            if self.debug:
                print(f'Cycle {cycle}:')
                print(f'Batch count: {batch_count}')
            data_batch_str = 'Data batch:'
            if self.debug:
                print(f'{data_batch_str:<16}', end='')
            for item in data_read:
                if item is not None:
                    if self.debug:
                        print(f'{item._edge_name:>8}', end=', ')

                else:
                    value_str = 'None'
                    if self.debug:
                        print(f'{value_str:>8}', end=', ')
            if self.debug:
                print()
            dest_pe_str = 'Dest PE:'
            if self.debug:
                print()
                print(f'{dest_pe_str:<16}', end='')
            for item in data_read:
                if item is not None:
                    component_map = self.architecture.component_map
                    dest_pe = component_map[item.src_component]
                    dest_pe_id = dest_pe.category_id
                    if self.debug:
                        print(f'{dest_pe_id:>8}', end=', ')
                else:
                    value_str = 'None'
                    if self.debug:
                        print(f'{value_str:>8}', end=', ')
            dest_pe_str = 'Dest PE index:'
            if self.debug:
                print()
                print(f'{dest_pe_str:<16}', end='')
            for item in data_read:
                if item is not None:
                    component_map = self.architecture.component_map
                    dest_pe = component_map[item.src_component]
                    dest_pe_id = dest_pe.category_id
                    ns = dest_pe.get_namespace(item.namespace_name)
                    index = ns.find_data_index(item.data_id)
                    if self.debug:
                        print(f'{index:>8}', end=', ')
                else:
                    value_str = 'None'
                    if self.debug:
                        print(f'{value_str:>8}', end=', ')
            dest_lane_str = 'Dest Lane:'
            if self.debug:
                print()
                print(f'{dest_lane_str:<16}', end='')
            for item in data_read:
                if item is not None:
                    component_map = self.architecture.component_map
                    dest_pe = component_map[item.src_component]
                    dest_pe_id = dest_pe.category_id
                    dest_lane_id = self.lane_gen.get_dest_laneid(dest_pe_id)
                    if self.debug:
                        print(f'{dest_lane_id:>8}', end=', ')
                else:
                    value_str = 'None'
                    if self.debug:
                        print(f'{value_str:>8}', end=', ')
            shift_amount_str = 'Shift Amount:'
            if self.debug:
                print()
                print(f'{shift_amount_str:<16}', end='')
            for curr_lane, item in enumerate(data_read):
                if item is not None:
                    component_map = self.architecture.component_map
                    dest_pe = component_map[item.src_component]
                    dest_pe_id = dest_pe.category_id
                    dest_lane_id = self.lane_gen.get_dest_laneid(dest_pe_id)
                    shift_amount = self.lane_gen.get_shift_amount(curr_lane, dest_lane_id)
                    if self.debug:
                        print(f'{shift_amount:>8}', end=', ')
                else:
                    value_str = 'None'
                    if self.debug:
                        print(f'{value_str:>8}', end=', ')

            # Read instruction
            read_inst = self.gen_read_inst(data_read, cycle, batch_count)
            instrs.append(read_inst)

            axi_read_flag_str = 'AXI read flags:'
            if self.debug:
                print()
                print(f'{axi_read_flag_str:<16}', end='')
            for flag in read_inst.axi_read_flags:
                flag = 'True' if flag == 1 else 'False'
                if self.debug:
                    print(f'{flag:>38}', end=', ')

            # Needed for mem inst binary annotation
            data_read_copy = pickle.loads(pickle.dumps(data_read))
            batches.append(data_read_copy)
            shift_insts = []

            lanes_by_shift = self.lane_gen.get_lanes_by_shift_amount(data_read, read_inst.axi_read_flags)
            if self.debug:
                print()
                print(f'Lanes by shift (lane_id, pe_id, ND index): {lanes_by_shift}')
                print(f'Num of shifts: {len(lanes_by_shift)}')
            nshifts = len(lanes_by_shift)
            cycle += 1
            for shift_amount in lanes_by_shift:
                affected_lanes = lanes_by_shift[shift_amount]
                inst = self.gen_shift_inst(shift_amount, affected_lanes, self.lanes, cycle, batch_count)
                #instrs.append(inst)
                cycle += 1
                shift_insts.append(inst)

            batch_count += 1

            lanes_by_shifts_list.append(lanes_by_shift)
            if self.debug:
                print('Shift inst order before reordering (only showing shift amounts):')
                for shift_inst in shift_insts:
                    print(f'{shift_inst.shiftamount}', end=', ')
                print(f"\n")

            # Order shift instructions so that data going to same PE are read in correct order
            shifts_reordered, valid_order = self.reorder_shifts(shift_insts, data_read, read_inst, cycle=cycle)
            if self.debug:
                print('FLAG')
                print(f'Valid order: {valid_order}')
                for shift_inst in shifts_reordered:
                    print(f'{shift_inst.shiftamount}', end=', ')
                print()

            # If any of the target lanes for a given shift amount have target index values that are not expected yet,
            # we need to split it up and send it later
            shifts_reordered = self.split_shift_instructions(shifts_reordered, valid_order, data_read)
            if self.debug:
                for shift_inst in shifts_reordered:
                    print(f'{shift_inst.shiftamount}', end=', ')
                print('\n')

            instrs.extend(shifts_reordered)
            if self.debug:
                print('After reordering shifts:')
                for shift_inst in shifts_reordered:
                    print(f'{shift_inst.shiftamount}', end=', ')
                print('\n')

        instrs.append(self.gen_wfi_inst(cycle, batch_count))
        instrs.append(self.gen_loop_inst(cycle + 1, batch_count))
        self.insts = instrs

        with open(filepath, 'w') as f:
            f.write(json.dumps([i.toDict() for i in instrs], sort_keys=False, indent=2))

        return batches, lanes_by_shifts_list

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

        return binary

    def write_to_verilog(self, binary, filepath, batches, lanes_by_shift_list):
        """Writes instruction binary in Verilog"""

        num_instructions = len(self.insts)
        print(f"{num_instructions} total memory instructions")
        # Calculate the number of address bits required from the total number of instructions
        addr_width = math.ceil(math.log2(num_instructions))

        prefix = f"""`timescale 1ns/1ps

module ROM_ASIC #(
// Parameters
    parameter   DATA_WIDTH          = 16,
    parameter   ADDR_WIDTH          = {addr_width},
    parameter   INIT                = "weight.txt",
    parameter   TYPE                = "block",
    parameter   ROM_DEPTH           = 1<<ADDR_WIDTH
) (
// Port Declarations
    input  wire                         CLK,
    input  wire                         RESET,
    input  wire  [ADDR_WIDTH-1:0]       ADDRESS,
    input  wire                         ENABLE,
    output reg   [DATA_WIDTH-1:0]       DATA_OUT,
    output reg                          DATA_OUT_VALID
);

// ******************************************************************
// Internal variables
// ******************************************************************

  localparam DEPTH = ROM_DEPTH;

  reg     [DATA_WIDTH-1:0]        rdata;
  wire     [ADDR_WIDTH-1:0]        address;

  assign address = ADDRESS;


  // `include "instructions.v"   // TODO
  always @(*) begin
	case(address)
/*****************************************************************************************/
"""

        binary = binary.split('\n')[:-1]
        always_block = []
        template = "{non_inst:<17}{inst}"
        read_inst_count = 0
        lanes_by_shift_count = 0
        for i, inst in enumerate(self.insts):
            non_inst = f"{addr_width}'d{i}: rdata = "
            inst_bin = binary[i]
            inst_str = str(MemoryInstructionGenerator.total_bits) + "'b" + inst_bin + ';'

            annotation = []
            if inst.op == 'read':
                batch = batches[read_inst_count]

                axi_flags = inst.axi_read_flags
                inst_annotation = f'// read {axi_flags}'
                batch_annotation = f'// {list(map(lambda x: x._edge_name if x is not None else None, batch))}'
                data_value_annotation = f'// Data values: {list(map(lambda x: x.value if x is not None else None, batch))}'
                dest_pe_annotation = f'// Dest PEs: {list(map(lambda x: find_dest_pe_id(self.architecture, x.src_component) if x is not None else None, batch))}'

                annotation.append('//')
                annotation.append(inst_annotation)
                annotation.append(batch_annotation)
                annotation.append(data_value_annotation)
                annotation.append(dest_pe_annotation)

                lanes_by_shift = lanes_by_shift_list[read_inst_count]
                read_inst_count += 1
            elif inst.op == 'shift':
                #lanes_by_shift = lanes_by_shift_list[read_inst_count]
                lanes = lanes_by_shift[inst.shiftamount]
                lane_ids = list(set(map(lambda x: x[0], lanes)))
                lanes_by_shift_count += 1
                inst_annotation = f'// shift amount: {inst.shiftamount}, Lanes IDs: {lane_ids}'

                annotation.append('//')
                annotation.append(inst_annotation)

            elif inst.op == 'wfi':
                annotation.append('//')
                annotation.append('// wfi')
            elif inst.op == 'loop':
                annotation.append('//')
                annotation.append('// loop')

            always_block.extend(annotation)
            always_block.append(template.format(non_inst=non_inst, inst=inst_str))

        always_block_str = '\n'.join(always_block)

        suffix = """/****************************************************************************************/
default: rdata = 56'b00000000000000000000000000000000000000000000000001110000;

	endcase
	end

    //reg     [ADDR_WIDTH-1:0]        address;

// ******************************************************************
// Read Logic
// ******************************************************************

    always @ (posedge CLK)
    begin : READ_VALID
        if (RESET) begin
            DATA_OUT_VALID <= 1'b0;
        end else if (ENABLE) begin
            DATA_OUT_VALID <= 1'b1;
        end
    end



 always @(posedge CLK) begin
    if (ENABLE)
        DATA_OUT <= rdata;
end

endmodule
"""

        verilog = prefix + always_block_str + suffix

        with open(filepath, 'w') as f:
            f.write(verilog)
        return num_instructions


class WeightConfigGenerator(object):
    """
    Generates meta information on weights that's needed for RTL.

    TODO It is still unclear how the config file generated by this class is used in RTL.
    """
    def __init__(self, architecture: TablaTemplate):
        self.architecture = architecture
        self.n_lanes = self.architecture.config['num_lanes']

        # Number of PE's per Lane should be calcualted based on the total number of PE's and Lanes
        # NOTE num_pes and num_lanes must be powers of 2
        self.pes_per_lane = int(self.architecture.config['num_pes'] / self.architecture.config['num_lanes'])

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
        WEIGHT_COUNT_MACRO = "WEIGHT_COUNT_MACRO(lanes,pe)"

        # linef_pe = define + " " + wcpe + "{:d}" + " " + count + "{:x}"
        # linef_lane = define + " " + wcl + "{:d}" + " " + count + "{:x}"

        linef_pe = "(lanes == {lane_id} && pe == {rel_pe_id}) ? 16'd{count} : \\"

        s = []
        #s = ""
        s.append(define + " " + WEIGHT_COUNT_MACRO + " (\\")
        #s += "\n"
        for lid, lane in enumerate(lc):
            lane_count = 0
            #s += LANE.format(lid)
            #s += "\n"
            for pe_offset, pe_count in enumerate(lane):
                peid = pe_offset * self.n_lanes + lid
                #s += linef_pe.format(peid, pe_count)
                s.append(linef_pe.format(lane_id=lid, rel_pe_id=pe_offset, count=pe_count))
                #s += "\n"
                lane_count += pe_count
            #s += linef_lane.format(lid, lane_count)
            s.append(linef_pe.format(lane_id=lid, rel_pe_id=len(lane), count=lane_count))
            #s += "\n"
        #s += ")"

        # last line should only have lane count
        last_line = s.pop()
        lane_count = last_line[last_line.index('?') + 1 : last_line.index(':')].strip()
        s.append(lane_count + ')')

        string = '\n'.join(s)

        return string


def print_pe_assignments(data_nodes, architecture, debug=False):
    for node in data_nodes:
        pe = architecture.component_map[node.src_component]
        ns = pe.get_namespace(node.namespace_name)
        try:
            index = ns.find_data_index(node.data_id)
            if debug:
                print(f"{node._edge_name}, PE {find_dest_pe_id(architecture, node.src_component)}, val = {node.value}, data_id = {node.data_id}, index = {index}")
        except AssertionError:
            if debug:
                print(ns.ns_items.keys())
                print(f'Data id = {node.data_id}')
            #exit()
        #print(f"{node._edge_name}, PE {find_dest_pe_id(architecture, node.src_component)}, val = {node.value}, data_id = {node.data_id}, index = {index}")
    print()


def find_dest_pe_id(architecture, comp_id):
    component_map = architecture.component_map
    dest_pe = component_map[comp_id]
    dest_pe_id = dest_pe.category_id
    return dest_pe_id


def get_input_weight_nodes(schedule):
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
    # meta = {}
    # for edge in schedule._dfg_edges:
    #     if edge.is_src_edge and edge.namespace_name == 'NM':
    #         print(edge._edge_name)
    #         if edge.src_component not in meta:
    #             meta[edge.src_component] = edge
    # meta = list(meta.values())

    meta_nodes = []
    for node in schedule._dfg_nodes:
        if node.dtype == 'param':
            meta_nodes.append(node)
    return meta_nodes

    # print('-' * 32)
    # for edge in schedule._dfg_edges:
    #     if edge.is_src_edge:
    #         print(edge.edge_name, edge.namespace_name, edge.dtype)
    # print('-' * 32)
    #
    # print('*' * 32)
    # for node in schedule._dfg_nodes:
    #     if node.dtype == 'param' or node.dtype == 'constant':
    #         print(node.dtype, node.namespace_name, node.computed, node.op_name)
    # print('*' * 32)
    # return meta


def get_output_weight_nodes(schedule, architecture, debug=False):
    outputs = {}
    for i, n in enumerate(schedule._dfg_nodes):
        if n.dtype == "state":
            if n.parents == [0]:
                pass
            else:
                assert n.children == [1]
                pe = architecture.component_map[n.component_id]
                nid = n.parents[0] if schedule.get_schedule_node(n.parents[0]).is_data_node else n.parents[1]
                dindex = pe.get_namespace("NW").find_data_index(nid)
                outputs[n.node_id] = (n.computed, pe.category_id, dindex, schedule.get_schedule_node(nid).op_name)
    output_arr = sorted([k for k in outputs.keys()], key=lambda x: (outputs[x][2], outputs[x][1]))
    output_arr = [outputs[k]for k in output_arr]
    return output_arr

    # output_weight_nodes = []
    # for node in schedule._dfg_nodes:
    #     if node.is_sink_node():
    #         pe = architecture.component_map[node.component_id]
    #         ns = pe.get_namespace(node.namespace_name)
    #         try:
    #             index = ns.find_data_index(node.node_id)
    #             output_weight_nodes.append(node)
    #             if debug:
    #                 print(f"PE {node._cat_comp_id}, Index: {index}, computed: {node.computed}")
    #         except AssertionError:
    #             if debug:
    #                 print(f"PE {node._cat_comp_id}, computed: {node.computed}")
    #                 print(f"{node.namespace_name}: {ns.ns_items.keys()}")
    #                 print(f'Data id = {node.node_id}')
    # return output_weight_nodes


def write_to_input_file(data, filepath):
    with open(filepath, 'w') as f:
        for item in data:
            f.write(str(item))
            f.write('\n')


def write_pe_files(data_nodes, mem_interface_artifacts_dir, architecture, dtype, debug=False):
    """Given input data list (NW or ND), write each data point to corresponding PE file."""

    # data_values = []
    # with open(data_file, 'r') as f:
    #     for index, line in enumerate(f):
    #         data_point = int(line)
    #         data_values.append(data_point)
    # #print(data_values)

    pes = {}
    for node in data_nodes:
        pe = architecture.component_map[node.src_component]
        pe_id = pe.category_id
        data_point = node.value
        if pe_id not in pes:
            pes[pe_id] = [data_point]
        else:
            pes[pe_id].append(data_point)
    if debug:
        print(pes)

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


def get_index(data_node, architecture):
    pe = architecture.component_map[data_node.src_component]
    ns = pe.get_namespace(data_node.namespace_name)
    return ns.find_data_index(data_node.data_id)


def reorder_nodes(data_nodes, architecture):
    return sorted(data_nodes, key=lambda x: (get_index(x, architecture), find_dest_pe_id(architecture, x.src_component)))


class MetadataLoadGenerator(object):
    def __init__(self, architecture, schedule):
        self.architecture = architecture
        self.schedule = schedule

    def generate_pe_instructions(self, schedule: Schedule, arch: TablaTemplate, filename, debug="values"):
        # TODO This should be configurable
        width = 16

        if not schedule.is_dfg_loaded():
            raise RuntimeError(f"Schedule has not loaded a DFG yet.")
        pes = [pe for _, pe in arch.category_component_dict["pe"].items() if isinstance(pe, PE)]
        # config = self.compute_instr_widths(arch.config, debug=debug)
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
                bin_str = f"\t\t\t{address_width}'d{index} : rdata = 16'b{np.binary_repr(meta_node.computed, width=width)};\n"
                pe_str += bin_str
            pe_str += f"\t\t\tdefault : rdata = 16'b{np.binary_repr(0, width=width)};\n"
            pe_str += f"\t\tendcase\n"
            pe_str += f"\tend\n"
            pe_str += f"end\n"
            pe_str += f"\n"
            pe_blocks.append(pe_str)
        pe_blocks.append("endgenerate\n")
        self.write_instr_file("".join(pe_blocks), filename)

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

        prefix = """`timescale 1ns/1ps
`ifdef FPGA
	`include "config.vh"
`endif

module bufferM #(

	parameter addrLen = 10,
	parameter dataLen = 32,
	parameter peId = 0

	)(
	clk,
	reset,

	rd_addr,

	data_out
);

	//--------------------------------------------------------------------------------------

	//--------------------------------------------------------------------------------------

	//--------------------------------------------------------------------------------------
	input clk;
	input reset;

	input [ addrLen - 1 : 0 ] rd_addr;

	output reg [ dataLen - 1 : 0 ] data_out;
	//--------------------------------------------------------------------------------------
reg [dataLen-1:0] rdata;
wire [addrLen-1:0] address;
assign address = rd_addr;

always @(posedge clk or posedge reset) begin
    if(reset)
        data_out <= {dataLen{1'b0}};
    else
	   data_out <= rdata;
end
/******************************************************/
"""
        suffix = """/******************************************************/
endmodule
"""
        bufferM = prefix + pe_instrs + suffix

        with open(filename, "w") as f:
            f.write(bufferM)

    def assign_meta_to_pe(self, meta_nodes):
        component_map = self.architecture.component_map
        dest_pes = {}
        for meta_node in meta_nodes:
            if meta_node.op_name == 'mu':
                for child in meta_node.children:
                    node = self.schedule.get_schedule_node(child)
                    #print(node.op_name, node._cat_comp_id, child)
                    # Find destination PE
                    dest_pe = component_map[node.component_id]

                    inst = str(node.get_instruction())
                    import re
                    match = re.search('NM\d*', inst)
                    nm = match.group()
                    index = int(nm[2:])

                    # Check if this PE has been checked
                    if dest_pe in dest_pes:
                        # Check if NM index is already in use.
                        if index in dest_pes[dest_pe]:
                            continue
                        else:
                            dest_pe.meta_nodes.append(meta_node)
                            dest_pe[dest_pe].append(index)
                    else:
                        # Put the node in the PE
                        dest_pe.meta_nodes.append(meta_node)
                        dest_pes[dest_pe] = [index]
            else:
                dest_pe = component_map[meta_node.component_id]
                dest_pe.meta_nodes.append(meta_node)


class Dtype(Enum):
    DATA = 1
    WEIGHT = 2


def dump_output_weights_in_axi(basepath, schedule, architecture, config, debug=False):
    axi_dir = f"{basepath}/mem-inst/axi/output_weights"
    if not os.path.exists(axi_dir):
        os.makedirs(axi_dir)

    if debug:
        print('Generating output weight AXI files')

    # output_weights is a list of (computed, pe category id, data index, op name)
    output_weights = get_output_weight_nodes(schedule, architecture, debug)

    axi_list = [AXI(i) for i in range(config['num_axi'])]
    axi_controller = AXIController(axi_list, architecture, debug=debug)

    # Number of PE's per Lane should be calcualted based on the total number of PE's and Lanes
    # NOTE num_pes and num_lanes must be powers of 2
    pes_per_lane = int(config['num_pes'] / config['num_lanes'])

    lane_gen = LaneGenerator(architecture, config['num_lanes'], pes_per_lane, config['num_axi'])
    init_lanes = lane_gen.init_lanes()
    lanes_per_axi = 4
    for axi_index, axi in enumerate(axi_list):
        lanes = init_lanes[axi_index * lanes_per_axi: axi_index * lanes_per_axi + lanes_per_axi]
        axi.set_lanes(lanes)

    # Dictionary to look up output weight nodes for each PE
    # This is so we avoid making any modification to PE class
    weight_nodes_by_pe_id = {}

    # Put output weight nodes to each PE
    # Get destination PE of a ScheduleNode by looking up category component id
    for weight_node in output_weights:
        # Find destination PE
        # dest_pe_id = weight_node._cat_comp_id
        dest_pe_id = weight_node[1]
        if dest_pe_id in weight_nodes_by_pe_id:
            weight_nodes_by_pe_id[dest_pe_id].append(weight_node)
        else:
            weight_nodes_by_pe_id[dest_pe_id] = [weight_node]

    # gen_matrix_for_axi needs to use output_weight_nodes instead of weight_nodes
    def gen_matrix_for_axi(axi_id, axi_list):
        axi = axi_list[axi_id]
        lane_data = []
        for lane in axi.lanes:
            weight_data = []
            for pe_id in lane.peids:
                if pe_id in weight_nodes_by_pe_id:
                    weight_nodes = weight_nodes_by_pe_id[pe_id]
                    # values = [node.computed for node in weight_nodes]
                    values = [node[0] for node in weight_nodes]
                    weight_data.extend(values)
                else:
                    pass
            lane_data.append(weight_data)
        return lane_data

    # Put placeholder values for each AXI and write to AXI files
    if debug:
        print('After filling in placeholders')
    for axi_id in range(config['num_axi']):
        weight_data = gen_matrix_for_axi(axi_id, axi_list)
        max_num_weights = axi_controller.find_max_number_of_weights(weight_data)
        for lane in weight_data:
            if len(lane) < max_num_weights:
                num_placeholders = max_num_weights - len(lane)
                lane.extend([0 for _ in range(num_placeholders)])
        weight_data = np.array(weight_data)
        if debug:
            print(f'AXI {axi_id}')
            print(weight_data)
            print()
        axi_controller.write_weights_from_axi(weight_data, os.path.join(axi_dir, f'axi_{axi_id}.txt'))


def generate_memory_instr(basepath, schedule, architecture, config, input_data_file, weight_file, meta_file, debug=False):
    mem_dir = f"{basepath}/mem-inst"
    # weight_file = f"{basepath}/mem-inst/input_weights.txt"
    # input_data_file = f"{basepath}/mem-inst/input_data.txt"
    # meta_file = f"{basepath}/mem-inst/meta.txt"
    axi_dir = f"{basepath}/mem-inst/axi"
    axi_input_data_dir = f"{axi_dir}/input_data"
    axi_weight_dir = f"{axi_dir}/weights"

    debug = False

    if not os.path.exists(axi_dir):
        os.makedirs(axi_dir)
    if debug:
        print(f'Generating instructions for Weight Data')

    # Get a list of weights (DFG nodes)
    weights = get_input_weight_nodes(schedule)

    # Set data values to DFG nodes
    set_values_to_nodes(weight_file, weights)
    if debug:
        print_pe_assignments(weights, architecture, debug)

    weights = reorder_nodes(weights, architecture)
    if debug:
        print("After re-ordering weight nodes:")
        print_pe_assignments(weights, architecture, debug)
        print()

    # Number of PE's per Lane should be calcualted based on the total number of PE's and Lanes
    # NOTE num_pes and num_lanes must be powers of 2
    pes_per_lane = int(config['num_pes'] / config['num_lanes'])

    # Print weight-to-PE assignment for each AXI
    meminst_gen = MemoryInstructionGenerator(weights, Dtype.WEIGHT, config['num_axi'], config['num_lanes'],
                                             pes_per_lane, architecture, debug=debug)
    if debug:
        print('AXI Contents:')
        meminst_gen.axi_controller.print_axi_contents()

    # Put placeholder values for each AXI and write to AXI files
    if debug:
        print('After filling in placeholders')
    for axi_index in range(config['num_axi']):
        weight_data = meminst_gen.axi_controller.gen_matrix_for_axi(axi_index)
        max_num_weights = meminst_gen.axi_controller.find_max_number_of_weights(weight_data)
        for lane in weight_data:
            if len(lane) < max_num_weights:
                num_placeholders = max_num_weights - len(lane)
                lane.extend([0 for _ in range(num_placeholders)])
        weight_data = np.array(weight_data)
        if debug:
            print(f'AXI {axi_index}')
            print(weight_data)
            print()
        meminst_gen.axi_controller.write_weights_from_axi(weight_data,
                                                          os.path.join(axi_dir, f'axi_{axi_index}.txt'))

    write_pe_files(weights, mem_dir, architecture, Dtype.WEIGHT, debug)

    # Generate weight config file (weightInst.txt)
    wconf_gen = WeightConfigGenerator(architecture)
    wconf_gen.gen_weightconf(weights, os.path.join(mem_dir, 'weight_insts.vh'))

    if debug:
        print(f'Generating instructions for Input Data')
    # Get a list of input data (DFG nodes)
    input_data_nodes = get_input_data_nodes(schedule)

    # Set data values to DFG nodes
    set_values_to_nodes(input_data_file, input_data_nodes)
    if debug:
        print("Before re-ordering input data nodes:")
        print_pe_assignments(input_data_nodes, architecture, debug)

    # Reorder data nodes so their destination indices appear in increasing order.
    input_data_nodes = reorder_nodes(input_data_nodes, architecture)
    if debug:
        print("\nAfter re-ordering input data nodes:")
        print_pe_assignments(input_data_nodes, architecture, debug)

    meminst_gen = MemoryInstructionGenerator(input_data_nodes, Dtype.DATA, config['num_axi'], config['num_lanes'],
                                             pes_per_lane, architecture, debug=debug)
    meminst_gen.axi_controller.write_axi_data(axi_dir)

    # Generate memory instructions
    batches, lanes_by_shift = meminst_gen.gen_inst(os.path.join(mem_dir, 'meminst.json'))
    binary = meminst_gen.gen_binary(os.path.join(mem_dir, 'meminst.txt'))
    num_instr = meminst_gen.write_to_verilog(binary, os.path.join(mem_dir, 'ROM_ASIC.v'), batches, lanes_by_shift)

    # Write input data to corresponding PE files
    write_pe_files(input_data_nodes, mem_dir, architecture, Dtype.DATA, debug)

    if debug:
        print(f'Generating Verilog file for metadata')
    # Generate Verilog files for loading metadata
    meta_nodes = get_input_meta_nodes(schedule)

    # Set meta value to meta nodes
    # with open(meta_file, 'r') as f:
    #     meta_data = int(f.readlines()[0])
    # for meta in meta_nodes:
    #     meta.value = meta_data
    # if debug:
    #     print_pe_assignments(meta_nodes, architecture, debug)

    meta_gen = MetadataLoadGenerator(architecture, schedule)
    meta_gen.assign_meta_to_pe(meta_nodes)
    meta_loader = os.path.join(mem_dir, 'bufferM.v')
    meta_gen.generate_pe_instructions(schedule, architecture, meta_loader)
    return num_instr


# NOTE main is deprecated. Don't run this.
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
    weights = get_input_weight_nodes(schedule)
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

    # Number of PE's per Lane should be calcualted based on the total number of PE's and Lanes
    # NOTE num_pes and num_lanes must be powers of 2
    pes_per_lane = int(config['num_pes'] / config['num_lanes'])

    # Print weight to PE assignment for each AXI
    meminst_gen = MemoryInstructionGenerator(weights, Dtype.WEIGHT, config['num_axi'], config['num_lanes'],
                                             pes_per_lane, architecture)
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
