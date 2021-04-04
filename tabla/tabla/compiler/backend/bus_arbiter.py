import copy
from _collections import deque
from typing import List

from backend import PU, Bus


class BusArbiter(object):

    def __init__(self):
        pass


class PEGBArbiter(BusArbiter):

    def __init__(self, pu: PU):
        super(PEGBArbiter, self).__init__()
        self.pu = pu
        self.pegb = self.pu.get_bus('PEGB')
        self.cycle = 0
        self.pes_per_pu = len(self.pu.pe_ids)

    def find_pes_with_write_buffer(self, cycle) -> List[int]:
        """Go through every write buffer in PEGB and find all the non-empty ones."""
        # List of PE category IDs
        pe_ids = []
        try:
            cycle_buffer = self.pegb.get_cycle_buffer(cycle)
        except RuntimeError:
            # print(f"Bus has not been written yet. No data in bus in cycle {cycle}")
            return []
        for pe_id in cycle_buffer:
            if pe_id == 'pegb':
                continue
            pe_buffer = cycle_buffer[pe_id]
            pe_write_buffer = pe_buffer['write']
            if len(pe_write_buffer) > 0:
                pe_ids.append(pe_id)
        return pe_ids

    def set_priority(self, cycle):
        """Set priority values for each PE in the given cycle."""
        priority_list = deque(range(self.pes_per_pu))
        priority_list.rotate(cycle)
        return priority_list

    def find_highest_priority_pe_id(self, pe_ids, priority_list):
        relative_pe_ids = list(map(lambda x : x % self.pes_per_pu, pe_ids))
        rel_pe_priorities = list(map(lambda x : priority_list[x], relative_pe_ids))
        index = rel_pe_priorities.index(min(rel_pe_priorities))
        return pe_ids[index]

    def get_data_from_pe_write_buffer(self, cycle, pe_id):
        val = self.pegb.get_data_from_pegb_write_buffer(cycle, pe_id)
        # val = self.pegb.get_data_pegb_write_buffer(cycle, pe_id)
        print(val.value)
        return val

    def get_bus_data(self, cycle):
        return self.pegb.get_data_from_pegb(cycle)
        # try:
        #     cycle_buffer = self.pegb.get_cycle_buffer(cycle)
        # except RuntimeError:
        #     # print(f"Bus has not been written yet. No data in bus in cycle {cycle}")
        #     return None
        # data_copy = cycle_buffer['pegb']
        # cycle_buffer['pegb'] = None
        #
        # state_name = self.pegb.get_state_name(cycle)
        # pe_buffers_updated = copy.deepcopy(cycle_buffer)
        # metadata = {"buffer": pe_buffers_updated}
        # _ = self.pegb.update_cycle_state(cycle, state_name, metadata)
        # return data_copy

    def write_to_bus(self, cycle, data):
        """Write data from PE Write Buffer to Bus (PEGB)."""
        self.pegb.move_data_to_pegb(cycle, data)
        #self.pegb.add_buffer_data(cycle, -1, -1, 'pegb', -1, data)

    def write_to_read_buffer(self, cycle, data):
        """Read data from Bus (PEGB) and write to PE Read Buffer."""
        # data is a tuple of (value, source_pe_id, dest_pe_id)
        dest_pe_id = data.value[2]
        self.pegb.move_data_to_pegb_read_buffer(cycle, dest_pe_id, data)
        #self.pegb.add_buffer_data(cycle, 'pegb', -1, dest_pe_id, -1, data)

    def run_cycle(self, cycle):
        """Perform the following routine for simulation:
        (1) Check if bus contains data.
        (2) If so, write data to its destination PE Read Buffer.
        (3) Find all PEs whose Write buffers have data written.
        (4) Determine highest priority PE in this cycle (priority varies by cycle).
        (5) Write data from this PE to Bus.
        """
        bus_data = self.get_bus_data(cycle)
        if bus_data:
            # data = bus_data.value
            print(f"PE Global Bus data found: {bus_data.value}")
            write_delay = 1
            self.write_to_read_buffer(cycle, bus_data)
        pe_ids = self.find_pes_with_write_buffer(cycle)
        if len(pe_ids) == 0:
            return
        priority_list = self.set_priority(cycle)
        print(f"PE IDs: {pe_ids}, Priority list: {priority_list}")
        pe_id = self.find_highest_priority_pe_id(pe_ids, priority_list)
        print(f"Highest priority PE ID: {pe_id}\n")
        data = self.get_data_from_pe_write_buffer(cycle, pe_id)
        write_delay = 0
        self.write_to_bus(cycle + write_delay, data)


class PUGBArbiter(BusArbiter):

    def __init__(self, pugb: Bus, num_pus):
        super(PUGBArbiter, self).__init__()
        self.pugb = pugb
        self.pu_count = num_pus

    def find_pus_with_write_buffer(self, cycle) -> List[int]:
        """Go through every write buffer in PEGB and find all the non-empty ones."""
        # List of PE category IDs
        pu_ids = []
        try:
            cycle_buffer = self.pugb.get_cycle_buffer(cycle)
        except RuntimeError:
            # print(f"Bus has not been written yet. No data in bus in cycle {cycle}")
            return []
        for pu_id in cycle_buffer:
            if pu_id == 'bus':
                continue
            pu_buffer = cycle_buffer[pu_id]
            pu_write_buffer = pu_buffer['write']
            if len(pu_write_buffer) > 0:
                pu_ids.append(pu_id)
        return pu_ids

    def set_priority(self, cycle):
        """Set priority values for each PU in the given cycle."""
        priority_list = deque(range(self.pu_count))
        priority_list.rotate(cycle)
        return priority_list

    def find_highest_priority_pu_id(self, pu_ids, priority_list):
        relative_pe_ids = list(map(lambda x : x % self.pu_count, pu_ids))
        rel_pe_priorities = list(map(lambda x : priority_list[x], relative_pe_ids))
        index = rel_pe_priorities.index(min(rel_pe_priorities))
        return pu_ids[index]

    def get_bus_data(self, cycle):
        """Get data from Bus (PU Global Bus) and return the data value."""
        return self.pugb.get_data_from_pugb(cycle)

    def write_to_bus(self, cycle, data):
        """Write data from PU Write Buffer to Bus (PUGB)."""
        self.pugb.move_data_to_pugb(cycle, data)
        # self.pugb.add_buffer_data(cycle, -1, -1, 'bus', -1, data)

    def get_pu_write_buffer_data(self, cycle, pu_id):
        val = self.pugb.get_data_from_pugb_write_buffer(cycle, pu_id)
        # val = self.pugb.get_data_pegb_write_buffer(cycle, pe_id)
        return val

    def write_to_read_buffer(self, cycle, data):
        """Read data from Bus (PUGB) and write it to PU Read Buffer."""
        # data is a tuple of (value, source_pe_id, dest_pe_id)
        dest_pu_id = data.value[2]
        self.pugb.move_data_to_pugb_read_buffer(cycle, dest_pu_id, data)
        # self.pugb.add_buffer_data(cycle, 'bus', -1, dest_pu_id, -1, data)

    def run_cycle(self, cycle):
        """Perform the following routine for simulation:
        (1) Check if bus contains data.
        (2) If so, write data to its destination PU Read Buffer (i.e. Head PE Read Buffer).
        (3) Find all Head PEs whose Write Buffers have data written.
        (4) Determine highest priority Head PE in this cycle (priority varies by cycle).
        (5) Write data from this Head PE to Bus.
        """
        bus_data = self.get_bus_data(cycle)
        if bus_data:
            print(f"PU Global Bus data found: {bus_data.value}")
            self.write_to_read_buffer(cycle, bus_data)
        pu_ids = self.find_pus_with_write_buffer(cycle)
        if len(pu_ids) > 0:
            priority_list = self.set_priority(cycle)
            print(f"PU IDs: {pu_ids}, Priority list: {priority_list}")
            pu_id = self.find_highest_priority_pu_id(pu_ids, priority_list)
            print(f"Highest priority PU ID: {pu_id}")
            data = self.get_pu_write_buffer_data(cycle, pu_id)
            write_delay = 0
            self.write_to_bus(cycle + write_delay, data)
        else:
            print(f"PUGB Arbiter: No PUs have data written to write buffer.")
            return
