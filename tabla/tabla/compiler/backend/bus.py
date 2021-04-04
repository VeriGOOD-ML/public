from collections import deque

from .component import Component
from .state import State
from .instruction import Source, Dest, Instruction, DataLocation
from . import Component
from . import State
from . import Instruction, DataLocation
import copy
from . import CYCLE_DELAYS
from typing import List


class BusItem(object):

    def __init__(self, src_id: int, comp_id: int,  dst_id: int, data_id: int, value=-1):
        self.src_id = src_id
        self.comp_id = comp_id
        self.dst_id = dst_id
        self.data_id = data_id

        self.valid = True
        self.uses = 1
        self.value = value

    def __str__(self):
        return f"Source ID: {self.src_id}\n" \
            f"Comp ID: {self.comp_id}" \
            f"Dest ID: {self.dst_id}" \
            f"Data ID: {self.data_id}"

    def __eq__(self, other):
        equality_list = []
        equality_list.append(other.data_id == self.data_id)
        equality_list.append(other.src_id == self.src_id)
        equality_list.append(other.dst_id == self.dst_id)
        equality_list.append(other.comp_id == self.comp_id)
        return all(equality_list)

    def invalidate(self):
        if not self.valid:
            raise RuntimeError(f"Namespace item {self.data_id} is already invalid.")
        self.valid = False

    def update_data(self, src_id: int, comp_id: int,  dst_id: int, data_id: int, value=-1):
        if self.valid:
            raise RuntimeError(f"Namespace item {self.data_id} is already valid.")
        self.src_id = src_id
        self.comp_id = comp_id
        self.data_id = data_id
        self.dst_id = dst_id
        self.uses += 1
        self.valid = True
        self.value = value


class Bus(Component):
    """Bus class for all communication objects within the architecture.

    Attributes
    ----------
    component_id : int
            This represents the globally unique id given to each component upon instantiation.
    category_id : int
            This represents the unique id within components of a specific `component_type`.
    component_type : str
            The string name corresponding to the component type, e.g., `bus`.
    component_subtype : str
            The string name corresponding to a subtype, e.g., `PEGB` for `bus`.

    """
    possible_states = ['busy', 'free']
    idle_state = 'free'

    def __init__(self, bus_type: str, buffer_size=256, pes=None, pus=None):
        """

        Parameters
        ----------
        bus_type : str
            The string name of the type of bus. e.g., `PUNB` for PU neighbor bus.
        """
        # TODO not a good design
        self.buffer_size = buffer_size
        self.pes = pes
        self.pus = pus

        super(Bus, self).__init__('bus', bus_type)
        self.component_id = self.resource_id

        self.pegb_read_buffer_written = []
        self.pugb_read_buffer_written = []

        self.create_initial_state()

    def create_busy_state(self, cycle: int, bus_metadata: dict) -> State:
        """
        Creates a busy cycle state in `cycle`, using the source (`src`), destination (`dst`),
        and data id (`data_id`) in `bus_metadata`.

        Parameters
        ----------
        cycle : int
            Cycle to create the busy state for.
        bus_metadata : dict
            A dictionary of metadata for the bus, consisting of `src`, `dst`, and `data_id` keys.

        Returns
        -------
        new_state : State
            The newly created busy state.

        """
        if set(bus_metadata.keys()) != {'src', 'dst', 'data_id'}:
            raise ValueError(f"Invalid bus metadata {bus_metadata}.\n{self}")
        new_state = self.add_cycle_state(cycle, 'busy', bus_metadata)
        return new_state

    def __str__(self):
        return f"Type: {self.component_type}\n\t" \
            f"Subtype: {self.component_subtype}\n\t" \
            f"ID: {self.component_id}"

    def get_state_src(self, cycle: int) -> int:
        """
        For a given busy bus state in `cycle`, retrieves the id of the source component.

        Parameters
        ----------
        cycle : int
            Cycle to retrieve the source id for.

        Returns
        -------
        src_id : int
            The id of the source component for the current instruction.

        """
        state = self.get_cycle_state(cycle)
        src_id = state.get_metadata_by_key('src')
        return src_id

    def get_state_dst(self, cycle: int) -> int:
        """
        For a given busy bus state in `cycle`, retrieves the id of the destination component.

        Parameters
        ----------
        cycle : int
            Cycle to retrieve the destination id for.

        Returns
        -------
        dst_id : int
            The id of the destination component for the current instruction.

        """
        state = self.get_cycle_state(cycle)
        dst_id = state.get_metadata_by_key('dst')
        return dst_id

    def get_state_data_id(self, cycle: int) -> int:
        """
        For a given busy bus state in `cycle`, retrieves the id of the data or edge id within the DFG.

        Parameters
        ----------
        cycle : int
            Cycle to retrieve the data id for.

        Returns
        -------
        data_id : int
            The id of the data/edge id for the current instruction.

        """
        state = self.get_cycle_state(cycle)
        data_id = state.get_metadata_by_key('data_id')
        return data_id

    def create_initial_state(self) -> State:
        """
        Initializes the Bus to as an idle state in the first cycle

        Returns
        -------
        init_state : State
            The newly created initial state
        """
        if self.component_subtype == 'PEGB':
            pe_buffers = {pe_id: {'read': deque([], self.buffer_size),
                                  'write': deque([], self.buffer_size)}
                          for pe_id in self.pes}
            pe_buffers['pegb'] = None
            metadata = {'buffer': pe_buffers}
        elif self.component_subtype == 'PENB':
            metadata = {'buffer': {'read': deque([], self.buffer_size)}}
        elif self.component_subtype == "PUGB":
            pugb = {pu_id: {'read': deque([], self.buffer_size),
                            'write': deque([], self.buffer_size)}
                    for pu_id in self.pus}
            pugb['bus'] = None
            metadata = {'buffer': pugb}
        elif self.component_subtype == "PUNB":
            metadata = {'buffer': {'read': deque([], self.buffer_size)}}
        else:
            # TODO
            metadata = {'buffer': deque([], self.buffer_size)}
        cycle = 0
        state_name = self.idle_state
        init_state = self.add_cycle_state(cycle, state_name, metadata)
        return init_state

    def is_data_present(self, cycle: int, bus_item: BusItem) -> bool:
        """
        Determines if the data corresponding to `data_id` is in the bus in cycle `cycle`.

        Parameters
        ----------
        cycle : int
            Cycle to check if the data is present.
        data_id : int
            Id of the data/edge within the DFG to check.

        Returns
        -------
        False if the data is not present in the bus, otherwise True.

        """
        buffer = self.get_cycle_buffer(cycle)
        if self.component_subtype == "PEGB":
            return False
            #print(buffer)
            # for pe_id in buffer:
            #     pe_buffer = buffer[pe_id]
            #     # # TODO ad-hoc
            #     # if isinstance(pe_buffer, deque):
            #     #     return False
            #     #print(pe_buffer)
            #
            #     pe_read_buffer = pe_buffer["read"]
            #     #print(pe_read_buffer)
            #     for data in pe_read_buffer:
            #         if data == bus_item:
            #             return True
        elif self.component_subtype == "PENB":
            read_buffer = buffer["read"]
            for entry in read_buffer:
                if entry == bus_item:
                    return True
        else:
            for data in buffer:
                if data == bus_item:
                    return True
        return False

    def is_same_bus(self, data_loc: DataLocation) -> bool:
        """
        Checks if the `data_loc` is the same as the current bus in order to make sure no bus contention.

        Parameters
        ----------
        data_loc : DataLocation
            Location corresponding to the data source or destination.

        Returns
        -------
        True if `data_loc` is the same as the current bus, otherwise False.
        """
        if self.component_subtype != data_loc.location:
            return False
        elif self.component_id != data_loc.data_id:
            return False
        return True

    def get_cycle_buffer(self, cycle: int):
        """
        Gets the storage items in `cycle` for the namespace.

        Parameters
        ----------
        cycle : int
            The cycle in which to return the storage.

        Returns
        -------
        The storage in state `cycle`.

        """
        state = self.get_cycle_state(cycle)
        return state.get_metadata_by_key("buffer")

    def add_instruction(self, cycle: int, instruction: Instruction) -> State:
        """
        Adds `instruction` to `cycle` by updating the state of the bus in order to occupy the bus with the
        instruction data.

        Parameters
        ----------
        cycle : int
            Cycle in which to add an instruction.
        instruction : Instruction
            Instruction to determine what data to add to the bus.

        Returns
        -------
        Newly updated cycle state with `instruction` added.

        """

        for dest in instruction.get_dests():
            if dest.data_id == self.component_id:
                metadata = {'src': instruction.component_id,
                            'dst': self.component_id,
                            'data_id': dest.dest_id}
                if self.cycle_state_exists(cycle):
                    self.update_cycle_state(cycle, 'busy', metadata)
                else:
                    self.add_cycle_state(cycle, 'busy', metadata)

        return self.get_cycle_state(cycle)

    # def find_first_use(self, bus_item: BusItem) -> int:
    #     if self.find_all_data_use(bus_item) < 0:
    #         raise RuntimeError(f"Unable to find data for {bus_item.data_id} in bus {self.component_subtype}-"
    #                            f"{self.component_id}")
    #
    #     added_cycle = -1
    #     for cycle in range(self.max_cycle + 1):
    #         if self.is_data_present(cycle, bus_item):
    #             added_cycle = cycle
    #             break
    #
    #     return added_cycle

    # def find_data_index(self, cycle, bus_item: BusItem) -> int:
    #     """
    #     Finds the namespace storage index associated with `data_id`.
    #
    #     Parameters
    #     ----------
    #     cycle : int
    #         The cycle to find the data index in.
    #     data_id : int
    #         The id associated with the data.
    #
    #     Returns
    #     -------
    #     The index within the namespace storage of the data.
    #     """
    #     if not self.cycle_state_exists(cycle):
    #         self.generate_intermediate_states(cycle)
    #
    #     storage = self.get_cycle_buffer(cycle)
    #     for index, data in enumerate(storage):
    #         if data == bus_item:
    #             return index
    #     raise RuntimeError(f"Unable to find data id {bus_item.data_id} in cycle {cycle}."
    #                        f"\nFirst found in cycle: {self.find_first_use(bus_item)}")
    #
    # def find_all_data_use(self, bus_item: BusItem) -> int:
    #     index = -1
    #     for cycle in range(self.max_cycle + 1):
    #         if self.is_data_present(cycle, bus_item):
    #             index = self.find_data_index(cycle, bus_item)
    #             break
    #
    #     return index

    def generate_intermediate_states(self, cycle: int) -> State:
        for cycle_state in range(self.max_cycle + 1, cycle + 1):
            max_state = self.get_cycle_state(self.max_cycle)
            metadata = max_state.metadata.copy()
            state_name = max_state.state_name
            self.add_cycle_state(cycle_state, state_name, metadata)
        return self.get_cycle_state(cycle)

    def add_data_to_neighbor_bus(self, cycle, data_val):
        if not self.cycle_state_exists(cycle):
            buffer = self.get_cycle_buffer(self.max_cycle)
            self.generate_intermediate_states(cycle)
        else:
            buffer = self.get_cycle_buffer(cycle)
        # We need to operate on a buffer copy. Otherwise, this cycle state buffer is also affected.
        buffer = copy.deepcopy(buffer)
        bus_item = BusItem(-1, -1, -1, -1, data_val)
        if self.component_subtype in ['PENB', 'PUNB']:
            read_buffer = buffer['read']
            read_buffer.append(bus_item)
            self.update_cycle_state(cycle, self.get_state_name(cycle), {'buffer': buffer})

    def add_data_to_pegb_write_buffer(self, cycle, src_pe_id, dest, data_id, data_val):
        if self.component_subtype == 'PEGB':
            if not self.cycle_state_exists(cycle):
                self.generate_intermediate_states(cycle)
            bus_item = BusItem(-1, -1, -1, -1, data_val)
            buffer = self.get_cycle_buffer(cycle)
            pe_buffer = buffer[src_pe_id]
            pe_buffer_write = pe_buffer['write']
            pe_buffer_write.append(bus_item)
            self.update_cycle_state(cycle, self.get_state_name(cycle), {'buffer': buffer})
        else:
            pass

    def move_data_to_pegb(self, cycle, data):
        if self.component_subtype == 'PEGB':
            buffer = self.get_cycle_buffer(cycle)
            buffer['pegb'] = data
            self.update_cycle_state(cycle, self.get_state_name(cycle), {'buffer': buffer})
        else:
            pass

    def move_data_to_pegb_read_buffer(self, cycle, dest_pe_id, data):
        if self.component_subtype == 'PEGB':
            buffer = self.get_cycle_buffer(cycle)
            pe_buffer = buffer[dest_pe_id]
            pe_buffer['read'].append(data)
            self.pegb_read_buffer_written.append({'src': data.value[1], 'dst': data.value[2], 'id': data.value[3]})
            self.update_cycle_state(cycle, self.get_state_name(cycle), {'buffer': buffer})
        else:
            pass

    def add_data_to_pugb_write_buffer(self, cycle, src_pu_id, dest, data_id, data):
        if self.component_subtype == 'PUGB':
            bus_item = BusItem(-1, -1, -1, -1, data)

            if not self.cycle_state_exists(cycle):
                self.generate_intermediate_states(cycle)
                buffer = self.get_cycle_buffer(self.max_cycle)
            else:
                buffer = self.get_cycle_buffer(cycle)

            #buffer = self.get_cycle_buffer(cycle)
            pu_buffer = buffer[src_pu_id]
            pu_buffer_write = pu_buffer['write']
            pu_buffer_write.append(bus_item)
            self.update_cycle_state(cycle, self.get_state_name(cycle), {'buffer': buffer})
        else:
            pass

    def move_data_to_pugb(self, cycle, data):
        if self.component_subtype == 'PUGB':
            buffer = self.get_cycle_buffer(cycle)
            buffer['bus'] = data
            self.update_cycle_state(cycle, self.get_state_name(cycle), {'buffer': buffer})
        else:
            pass

    def move_data_to_pugb_read_buffer(self, cycle, dest_pu_id, data):
        if self.component_subtype == 'PUGB':
            buffer = self.get_cycle_buffer(cycle)
            pu_buffer = buffer[dest_pu_id]
            pu_buffer['read'].append(data)
            self.pugb_read_buffer_written.append({'src': data.value[1], 'dst': data.value[2]})
            self.update_cycle_state(cycle, self.get_state_name(cycle), {'buffer': buffer})
        else:
            pass

    def add_buffer_data(self, cycle: int, src_comp, comp_id: int, dst_comp, data_id: int, value=-1):
        """
               Inserts data into the Namespace storage using the DFG edge with id `data_id`, produced by the node
               `src_id`.

               Parameters
               ----------
               cycle : int
                   The cycle to insert the data.
               edge_id : int
                   The source DFG node id.
               data_id : int
                   The DFG edge id.

               Returns
               -------
               The updated state with the newly added data.
               """
        bus_item = BusItem(src_comp, comp_id, dst_comp, data_id, value)

        #delay_cycles = CYCLE_DELAYS["PE"][self.component_subtype] + 1
        delay_cycles = 0
        if not self.cycle_state_exists(cycle + delay_cycles):
            _ = self.generate_intermediate_states(cycle + delay_cycles)

        # if self.is_data_present(cycle, bus_item):
        #     return self.get_cycle_state(cycle)

        for future_cycle in range(cycle, cycle + delay_cycles + 1):
            # if self.is_data_present(future_cycle, bus_item):
            #     continue
            print(f"add_buffer_data: {cycle}")

            if self.component_subtype == "PEGB":
                pe_buffers = self.get_cycle_buffer(future_cycle)
                # NOTE: dst_comp is the destination PE category ID
                buffer = pe_buffers[dst_comp]
                if dst_comp == "pegb":
                    pe_buffers["pegb"] = bus_item
                else:
                    if src_comp == "pegb":
                        buffer["read"].append(bus_item)
                        for item in self.pegb_read_buffer_written:
                            #if bus_item.value[1]
                        #if dst_comp not in self.pegb_read_buffer_written:
                            self.pegb_read_buffer_written.append({'src': bus_item.value[1], 'dst': bus_item.value[2], 'id': bus_item.value[3]})
                    else:
                        buffer["write"].append(bus_item)
                state_name = self.get_state_name(future_cycle)
                pe_buffers_updated = {}
                for pe_id in self.pes:
                    if pe_id == dst_comp:
                        pe_buffers_updated[pe_id] = copy.deepcopy(buffer)
                    else:
                        pe_buffers_updated[pe_id] = copy.deepcopy(pe_buffers[pe_id])
                pe_buffers_updated['pegb'] = copy.copy(pe_buffers["pegb"])
                metadata = {"buffer": copy.deepcopy(pe_buffers_updated)}
                self.update_cycle_state(future_cycle, state_name, metadata)

                # self.pegb_read_buffer_written.append(dst_comp)

            elif self.component_subtype == "PENB":
                read_buffer = self.get_cycle_buffer(future_cycle)
                #print(read_buffer)
                buffer = read_buffer["read"]
                buffer.append(bus_item)
                state_name = self.get_state_name(future_cycle)
                metadata = {'buffer': {'read': copy.deepcopy(buffer)}}
                self.update_cycle_state(future_cycle, state_name, metadata)
            elif self.component_subtype == "PUGB":
                pugb = self.get_cycle_buffer(future_cycle)
                pu_buffer = pugb[dst_comp]
                if dst_comp == 'bus':
                    pugb['bus'] = bus_item
                else:
                    if src_comp == 'bus':
                        pu_buffer['read'].append(bus_item)
                    else:
                        pu_buffer['write'].append(bus_item)

                state_name = self.get_state_name(future_cycle)
                pugb_updated = {}
                for key in self.pus:
                    if key == dst_comp:
                        pugb_updated[key] = copy.deepcopy(pu_buffer)
                    else:
                        pugb_updated[key] = copy.deepcopy(pugb[key])
                pugb_updated['bus'] = copy.deepcopy(pugb['bus'])
                metadata = {'buffer': pugb_updated}
                self.update_cycle_state(future_cycle, state_name, metadata)
            elif self.component_subtype == "PUNB":
                buffer = self.get_cycle_buffer(future_cycle)
                read_buffer = buffer["read"]
                read_buffer.append(bus_item)
                state_name = self.get_state_name(future_cycle)
                metadata = {'buffer': {'read': copy.deepcopy(read_buffer)}}
                self.update_cycle_state(future_cycle, state_name, metadata)
            else:
                buffer = self.get_cycle_buffer(future_cycle)
                buffer.append(bus_item)

                state_name = self.get_state_name(future_cycle)
                metadata = {"buffer": copy.deepcopy(buffer)}
                self.update_cycle_state(future_cycle, state_name, metadata)

        return self.get_cycle_state(cycle)


    def check_data_exists_from_pegb_read_buffer(self, cycle, src_pe_id, dest_pe_id, id):
        if self.component_subtype == 'PEGB':
            try:
                if self.cycle_state_exists(cycle):
                    buffer = self.get_cycle_buffer(cycle)
                else:
                    buffer = self.get_cycle_buffer(self.max_cycle)
            except RuntimeError:
                # This RuntimeError is thrown because add_buffer_data hasn't been previously called yet, which is expected.
                return False
            pe_buffer = buffer[dest_pe_id]['read']
            if len(pe_buffer) == 0:
                return False

            # Traverse the read buffer and find the entry with corresponding source PE ID
            for index, bus_entry in enumerate(pe_buffer):
                # bus_entry.value is a tuple of form (val, src_pe_id, dest_pe_id)
                print(f"index {index}: {bus_entry.value}, {src_pe_id}")
                if src_pe_id == bus_entry.value[1] and id == bus_entry.value[3]:
                    return True
            return False

    def get_data_from_pegb_read_buffer(self, cycle, src_pe_id, dest_pe_id):
        if self.component_subtype == 'PEGB':
            try:
                if self.cycle_state_exists(cycle):
                    buffer = self.get_cycle_buffer(cycle)
                else:
                    buffer = self.get_cycle_buffer(self.max_cycle)
            except RuntimeError:
                # This RuntimeError is thrown because add_buffer_data hasn't been previously called yet, which is expected.
                raise BusEmptyException
            pe_buffer = buffer[dest_pe_id]['read']
            if len(pe_buffer) == 0:
                raise BusEmptyException

            # Traverse the read buffer and find the entry with corresponding source PE ID
            for index, bus_entry in enumerate(pe_buffer):
                # bus_entry.value is a tuple of form (val, src_pe_id, dest_pe_id)
                print(f"index {index}: {bus_entry.value}, {src_pe_id}")
                if src_pe_id == bus_entry.value[1]:
                    print(f"Data found: {bus_entry.value}")
                    entry_copy = bus_entry
                    del pe_buffer[index]
                    print(f"Removed data: {entry_copy.value}")
                    print(f"after removing: [", end='')
                    for item in pe_buffer:
                        print(f"{item.value}", end=', ')
                    print("] ")

                    if not self.cycle_state_exists(cycle):
                        self.generate_intermediate_states(cycle)
                    self.update_cycle_state(cycle, self.get_state_name(cycle), {'buffer': buffer})
                    return entry_copy
            raise DataNotFoundException
        else:
            pass

    def get_data_from_pegb(self, cycle):
        if self.component_subtype == 'PEGB':
            if not self.cycle_state_exists(cycle):
                self.generate_intermediate_states(cycle)
                buffer = self.get_cycle_buffer(self.max_cycle)
            else:
                buffer = self.get_cycle_buffer(cycle)
            data = buffer['pegb']
            buffer['pegb'] = None
            self.update_cycle_state(cycle, self.get_state_name(cycle), {'buffer': buffer})
            return data
        else:
            pass

    def get_data_from_pegb_write_buffer(self, cycle, pe_id):
        if self.component_subtype == 'PEGB':
            try:
                buffer = self.get_cycle_buffer(cycle)
            except RuntimeError:
                # This RuntimeError is thrown because add_buffer_data hasn't been previously called yet, which is expected.
                raise BusEmptyException
            pe_buffer = buffer[pe_id]["write"]
            data = pe_buffer.popleft()
            self.update_cycle_state(cycle, self.get_state_name(cycle), {'buffer': buffer})
            return data
        else:
            pass

    def get_data_from_pugb_read_buffer(self, cycle, src_pu_id, dest_pu_id):
        if self.component_subtype == 'PUGB':
            if not self.cycle_state_exists(cycle):
                self.generate_intermediate_states(cycle)
                buffer = self.get_cycle_buffer(self.max_cycle)
            else:
                buffer = self.get_cycle_buffer(cycle)
            pu_buffer = buffer[dest_pu_id]['read']
            if len(pu_buffer) == 0:
                raise BusEmptyException

            for index, bus_entry in enumerate(pu_buffer):
                if src_pu_id == bus_entry.value[1]:
                    entry_copy = bus_entry
                    del pu_buffer[index]
                    if not self.cycle_state_exists(cycle):
                        self.generate_intermediate_states(cycle)
                    self.update_cycle_state(cycle, self.get_state_name(cycle), {'buffer': buffer})
                    return entry_copy
            raise DataNotFoundException
        else:
            pass

    def get_data_from_pugb(self, cycle):
        if self.component_subtype == 'PUGB':
            if not self.cycle_state_exists(cycle):
                self.generate_intermediate_states(cycle)
                buffer = self.get_cycle_buffer(cycle)
            else:
                buffer = self.get_cycle_buffer(cycle)
            data = buffer['bus']
            buffer['bus'] = None
            self.update_cycle_state(cycle, self.get_state_name(cycle), {'buffer': buffer})
            return data
        else:
            pass

    def get_data_from_pugb_write_buffer(self, cycle, pu_id):
        if self.component_subtype == 'PUGB':
            if not self.cycle_state_exists(cycle):
                self.generate_intermediate_states(cycle)
                buffer = self.get_cycle_buffer(self.max_cycle)
            else:
                buffer = self.get_cycle_buffer(cycle)
            pu_buffer = buffer[pu_id]['write']
            data = pu_buffer.popleft()
            self.update_cycle_state(cycle, self.get_state_name(cycle), {'buffer': buffer})
            return data
        else:
            pass


    def get_data(self, cycle, pe_id=-1, src_pe_id=-1):
        if self.component_subtype == "PENB":
            if self.cycle_state_exists(cycle):
                cycle_buffer = self.get_cycle_buffer(cycle)
            else:
                cycle_buffer = self.get_cycle_buffer(self.max_cycle)
                self.generate_intermediate_states(cycle)


            # if pe_id.resource_id == 60:
            #     print('-' * 80 + 'FUCK' + 'max cycle: ' + str(self.max_cycle))

            # try:
            #     cycle_buffer = self.get_cycle_buffer(cycle)
            # except RuntimeError:
            #     if pe_id.resource_id == 60:
            #         print('-' * 80 + 'FUCK' + 'max cycle: ' + str(self.max_cycle))
            #     # This RuntimeError is thrown because add_buffer_data hasn't been previously called yet, which is expected.
            #     raise BusEmptyException
            # print(read_buffer)
            buffer = cycle_buffer["read"]
            if len(buffer) == 0:
                raise BusEmptyException
            value = buffer.popleft()

            state_name = self.get_state_name(cycle)
            pe_buffers_updated = copy.deepcopy(cycle_buffer)
            metadata = {"buffer": pe_buffers_updated}
            self.update_cycle_state(cycle, state_name, metadata)

            return value
        elif self.component_subtype == "PEGB":
            try:
                if self.cycle_state_exists(cycle):
                    cycle_buffer = self.get_cycle_buffer(cycle)
                else:
                    cycle_buffer = self.get_cycle_buffer(self.max_cycle)
            except RuntimeError:
                # This RuntimeError is thrown because add_buffer_data hasn't been previously called yet, which is expected.
                raise BusEmptyException
            if pe_id == "pegb":
                if cycle_buffer["pegb"] is None:
                    raise BusEmptyException
                else:
                    value_copy = copy.copy(cycle_buffer["pegb"])
                    cycle_buffer["pegb"] = None

                    state_name = self.get_state_name(cycle)
                    pe_buffers_updated = copy.deepcopy(cycle_buffer)
                    metadata = {"buffer": pe_buffers_updated}
                    _ = self.update_cycle_state(cycle, state_name, metadata)

                    return value_copy
            else:
                pe_buffer = cycle_buffer[pe_id]
                buffer = pe_buffer["read"]
                # If there's no data in read buffer, notify the caller.
                if len(buffer) == 0:
                    raise BusEmptyException
                # Traverse the read buffer and find the entry with corresponding source PE ID
                for index, bus_entry in enumerate(buffer):
                    # bus_entry.value is a tuple of form (val, src_pe_id, dest_pe_id)
                    print(f"index {index}: {bus_entry.value}, {src_pe_id}")
                    if src_pe_id == bus_entry.value[1]:
                        print(f"Data found: {bus_entry.value}")
                        entry_copy = copy.deepcopy(bus_entry)
                        del buffer[index]
                        print(f"Removed data: {entry_copy.value}")
                        print(f"after removing: [", end='')
                        for item in buffer:
                            print(f"{item.value}", end=', ')
                        print("] ")

                        self.generate_intermediate_states(cycle)

                        state_name = self.get_state_name(cycle)
                        pe_buffers_updated = copy.deepcopy(cycle_buffer)
                        self.print_global_bus(pe_buffers_updated)
                        metadata = {"buffer": pe_buffers_updated}
                        _ = self.update_cycle_state(cycle, state_name, metadata)

                        return entry_copy
                raise DataNotFoundException
                # value = buffer.popleft()
        elif self.component_subtype == "PUGB":
            try:
                pugb = self.get_cycle_buffer(cycle)
            except RuntimeError:
                raise BusEmptyException
            if pe_id == 'bus':
                if pugb['bus'] is None:
                    raise BusEmptyException
                else:
                    value_copy = copy.deepcopy(pugb['bus'])
                    pugb['bus'] = None

                    state_name = self.get_state_name(cycle)
                    pugb_updated = copy.deepcopy(pugb)
                    metadata = {'buffer': pugb_updated}
                    self.update_cycle_state(cycle, state_name, metadata)

                    return value_copy
            else:
                pu_buffer = pugb[pe_id]
                read_buffer = pu_buffer['read']
                if len(read_buffer) == 0:
                    raise BusEmptyException
                for bus_item in read_buffer:
                    if src_pe_id == bus_item.value[1]:
                        value_copy = copy.deepcopy(bus_item)
                        read_buffer.remove(bus_item)

                        state_name = self.get_state_name(cycle)
                        pugb_updated = copy.deepcopy(pugb)
                        metadata = {'buffer': pugb_updated}
                        self.update_cycle_state(cycle, state_name, metadata)

                        return value_copy
                print(f"Bus Item not found.")

        elif self.component_subtype == "PUNB":
            if not self.cycle_state_exists(cycle):
                buffer = self.get_cycle_buffer(self.max_cycle)
                self.generate_intermediate_states(cycle)
            else:
                buffer = self.get_cycle_buffer(cycle)
            buffer = copy.deepcopy(buffer)
            read_buffer = buffer["read"]
            if len(read_buffer) == 0:
                raise BusEmptyException
            self.print_neighbor_bus(buffer)
            value = read_buffer.popleft()
            self.update_cycle_state(cycle, self.get_state_name(cycle), {'buffer': buffer})
            return value
        else:
            pass

    def check_data_exists(self, cycle):
        if self.component_subtype == "PENB":
            if self.cycle_state_exists(cycle):
                cycle_buffer = self.get_cycle_buffer(cycle)
            else:
                cycle_buffer = self.get_cycle_buffer(self.max_cycle)
                self.generate_intermediate_states(cycle)
            buffer = cycle_buffer["read"]
            if len(buffer) == 0:
                return False
            else:
                return True
        elif self.component_subtype == "PUNB":
            if not self.cycle_state_exists(cycle):
                buffer = self.get_cycle_buffer(self.max_cycle)
                self.generate_intermediate_states(cycle)
            else:
                buffer = self.get_cycle_buffer(cycle)
            buffer = copy.deepcopy(buffer)
            read_buffer = buffer["read"]
            if len(read_buffer) == 0:
                return False
            else:
                return True

    def print_global_bus(self, bus):
        for key in bus:
            print(f"PE {key}: ")
            if key == "bus" or key == "pegb":
                print(f"Bus: {bus[key]}")
            else:
                read_write = bus[key]
                print(f"Read: [", end='')
                for item in read_write["read"]:
                    print(f"{item.value}", end=', ')
                print("] ", end='')
                print(f"Write: [", end='')
                for item in read_write["write"]:
                    print(f"{item.value}", end=', ')
                print("]")

    def print_neighbor_bus(self, bus):
        if self.component_subtype in ['PENB', 'PUNB']:
            buffer = bus['read']
            print(f"Read Buffer: [", end='')
            for item in buffer:
                print(item.value, end=', ')
            print(']\n')

    def is_valid_instruction(self, cycle: int, instruction: Instruction) -> bool:
        """
        Checks if `instruction` is valid in `cycle` by making sure the data is present if it is the source of the
        instruction, and makes sure it is idle if it is the destination bus.

        Parameters
        ----------
        cycle : int
            Cycle to check the state in.
        instruction : Instruction
            Instruction to check its validity


        Returns
        -------
        True if the instruction is valid in `cycle`, otherwise False.
        """
        data_id = self.get_state_data_id(cycle)
        # for source in instruction.srcs:
        #     # bus_item = BusItem(source.source.data_id
        #     if self.is_same_bus(source) and not self.is_data_present(cycle, data_id):
        #         return False

        for dest in instruction.dests:
            if self.is_same_bus(dest) and not self.is_idle(cycle + 1):
                return False
        return True


class BusEmptyException(Exception):
    pass


class DataNotFoundException(Exception):
    pass
