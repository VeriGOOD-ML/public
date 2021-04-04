from . import Component
from . import NamespaceItem
from . import Instruction
from . import State
from typing import List, Dict
import bisect
import copy
import pickle
from functools import reduce
from pytools import memoize_method

class Namespace(Component):
    """A Namespace class which stores data, an is associated with a specific PE.

    Attributes
    ----------
    possible_states : List[str]
            This represents the possible state names a Namespace can take on during a given cycle.
    idle_state : str
            This is the string name which represents an idle state. For a Namespace, the idle state is "free",
            which indicates a free slot is available.
    component_id : int
            This is the unique component id, also known as `resource_id`.
    """
    possible_states = ['full', 'free']
    idle_state = 'free'

    def __init__(self, ns_type: str, capacity: int):
        self._capacity = capacity
        super(Namespace, self).__init__('namespace', ns_type)
        self.component_id = self.resource_id
        self.ns_items: Dict[int, (int, NamespaceItem)] = {}
        self._item_count = 0
        _ = self.create_initial_state()
        if not isinstance(capacity, int):
            raise TypeError(f"Capacity must be an integer value")

    def __str__(self):
        return f"Type: {self.component_type}\n\t" \
            f"Subtype: {self.component_subtype}\n\t" \
            f"ID: {self.component_id}"

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def pe_id(self) -> int:
        for pe_id, pe in self.category_component_dict["pe"].items():
            if self.component_id == pe.ns_id(self.component_subtype):
                return pe_id
        raise RuntimeError()

    def decr_items(self):
        self._item_count -= 1

    def item_count(self) -> int:
        """
        Provides the number of items stored in the Namespace in `cycle`.

        Parameters
        ----------
        cycle : int
            Cycle in which to determine the item count.

        Returns
        -------
            The number of items in storage in state `cycle`.
        """
        return self._item_count

    def utilization(self, cycle: int) -> str:
        """
        Returns the percent utilization of the Namespace in `cycle`. The utilization is the
        number of items in the Namespace divided by the capacity of the Namespace.

        Parameters
        ----------
        cycle : int
            Cycle to check the utilization.

        Returns
        -------
        The percent utilization during state `cycle`.

        """
        return f"{100 * self.item_count()/self._capacity}%"

    def num_unique(self, cycle: int):
        """
        Returns the percent utilization of the Namespace in `cycle`. The utilization is the
        number of items in the Namespace divided by the capacity of the Namespace.

        Parameters
        ----------
        cycle : int
            Cycle to check the utilization.

        Returns
        -------
        The percent utilization during state `cycle`.

        """

        storage = self.get_cycle_storage()
        ids = set([data.data_id for data in storage if data.is_valid()])
        return len(ids), self.item_count()

    def create_initial_state(self) -> State:
        """
        Initializes the Namespace with empty storage in the first cycle.

        Returns
        -------
        init_state : State
            The newly created initial state.
        """
        storage = []
        for _ in range(self._capacity):
            new_data = NamespaceItem(-1, -1, -1)
            new_data.invalidate()
            storage.append(new_data)
        metadata = {'storage' : storage}
        cycle = 0
        state_name = self.idle_state
        init_state = self.add_cycle_state(cycle, state_name, metadata)
        return init_state

    def get_cycle_storage(self) -> List[NamespaceItem]:
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
        state = self.get_cycle_state(0)
        return state.get_metadata_by_key('storage')

    def is_full(self) -> bool:
        """
        Determines if the Namespace is full in `cycle` by comparing against the
        Namespace capacity.

        Parameters
        ----------
        cycle : int
            The cycle to check if the storage is full.

        Returns
        -------
        True if the storage is full, otherwise False.

        """
        return self.item_count() == self._capacity


    def is_index_valid(self, index: int) -> bool:
        """
        Checks if `index` is being used in `cycle` by checking its validity in the storage.

        Parameters
        ----------
        cycle : int
            The cycle to check the index validity.
        index : int
            Index in the Namespace to check if it is valid.

        Returns
        -------
        True if the index is valid and being used in `cycle`, otherwise False.

        """
        storage = self.get_cycle_storage()
        return storage[index].is_valid()

    def find_empty_index(self, cycle, checked_indices: List[int]) -> int:
        """
        Finds an empty index in the namespace storage in state `cycle`.

        Parameters
        ----------
        cycle : int
            Cycle in which to find an empty index.
        checked_indices : List[int]
            Previously checked indices which might be empty in `cycle`, but invalid for other reasons.

        Returns
        -------
        Index which is available for use in `cycle`.

        """
        # TODO: Find empty index by also checking future index uses
        if self.is_full():
            raise RuntimeError(f"Storage is full in cycle {cycle}")
        storage = self.get_cycle_storage()

        for index, data in enumerate(storage):
            if index in checked_indices:
                continue
            if not data.is_valid():
                return index
        return -1

    def check_index_uses(self, index: int) -> bool:
        """
        Checks to see if `index` has been used in any cycle, starting with `start_cycle`.

        Parameters
        ----------
        start_cycle : int
            The cycle to begin checking states with.
        index : int
            The index in the Namespace storage.

        Returns
        -------
        True if the index is used in some future cycle, otherwise False.

        """
        return not self.is_index_valid(index)


    def generate_intermediate_states(self, cycle: int) -> State:

        max_state = self.get_cycle_state(self.max_cycle)
        _ = self.add_cycle_state(cycle, max_state.state_name, max_state.metadata)
        return self.get_cycle_state(cycle)

    def find_first_use(self, data_id: int) -> int:
        if self.find_all_data_use(data_id) < 0:
            raise RuntimeError(f"Unable to find data for {data_id} in namespace {self.component_subtype}-{self.component_id} within"
                               f" PE {self.pe_id}")

        assert data_id in self.ns_items
        return self.ns_items[data_id][1].ready_cycle

    def find_all_data_use(self, data_id: int) -> int:
        index = -1
        if data_id in self.ns_items:
            return self.ns_items[data_id][0]
        return index

    def insert_cycle_state(self, cycle):
        sorted_keys = sorted(list(self.states.keys()))
        previous_cycle = sorted_keys[bisect.bisect_left(sorted_keys, cycle) - 1]
        current_state = self.get_cycle_state(previous_cycle)
        return self.add_cycle_state(cycle, current_state.state_name, current_state.metadata)

    def insert_data(self, cycle: int, edge_id: int, data_id: int, value=None) -> int:
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
        index = self.find_all_data_use(data_id)

        if not self.cycle_state_exists(cycle):
            _ = self.generate_intermediate_states(cycle)

        if cycle not in self.states.keys():
            self.insert_cycle_state(cycle)

        if self.is_data_present(cycle, data_id) and self.component_subtype == "NW":
            storage = self.get_cycle_storage()
            storage[index].set_written_cycle(cycle, value=value)
            storage[index].updated_state(edge_id)
            state_name = self.get_state_name(cycle)
            _ = self.update_cycle_state(cycle, state_name, {"storage": storage})
            # return self.get_cycle_state(cycle)
        elif self.is_full():
            raise RuntimeError(f"Namespace {self.resource_id} is full in cycle {cycle}."
                               f"\n{self}")
        else:
            checked_indices = []
            if index < 0:
                index = self.find_empty_index(cycle, checked_indices)

                while not self.check_index_uses(index):
                    checked_indices.append(index)
                    index = self.find_empty_index(cycle, checked_indices)
                    if index == -1:
                        raise RuntimeError(f"Unable to find valid storage index for "
                                           f"Namespace {self.resource_id}"
                                           f" in cycle {cycle}\n{self}")

            storage = self.get_cycle_storage()
            storage[index].set_ready_cycle(cycle)

            if not self.is_data_present(cycle, data_id):
                storage[index].update_data(edge_id, data_id, value)
                self.ns_items[data_id] = (index, storage[index])
                self._item_count += 1
                if self.item_count() + 1 == self._capacity:
                    state_name = 'full'
                else:
                    state_name = self.get_state_name(cycle)
                metadata = {"storage": storage}
                _ = self.update_cycle_state(cycle, state_name, metadata)
        assert isinstance(index, int)
        return index

    def is_data_present(self, cycle: int, data_id: int) -> bool:
        """
        Checks to see if the data associated with `data_id` is present in the namespace storage.

        Parameters
        ----------
        cycle : int
            The cycle to check if data is present.
        data_id : int
            The id associated with the data.

        Returns
        -------
        True if the data is present in `cycle`, otherwise False.
        """

        return data_id in self.ns_items and self.ns_items[data_id][1].ready_cycle <= cycle and self.ns_items[data_id][1].ready_cycle >= 0

    def is_data_valid(self, cycle, data_id) -> bool:
        """
        Checks if the data associated with `data_id` is valid in `cycle`.
        Parameters
        ----------
        cycle : int
            Cycle to check if the data is valid in.
        data_id : int
            The id associated with the data.

        Returns
        -------
        True if the data is valid in `cycle`, otherwise False.
        """
        storage = self.get_cycle_storage()
        for data in storage:
            if data.data_id == data_id and data.is_valid() and data.ready_cycle <= cycle:
                return True
        return False

    def find_data_index(self, data_id, cycle=-1) -> int:
        """
        Finds the namespace storage index associated with `data_id`.

        Parameters
        ----------
        cycle : int
            The cycle to find the data index in.
        data_id : int
            The id associated with the data.

        Returns
        -------
        The index within the namespace storage of the data.
        """
        assert data_id in self.ns_items
        assert self.ns_items[data_id][1].ready_cycle <= cycle or cycle < 0
        return self.ns_items[data_id][0]

    def get_data(self, cycle: int, data_id: int) -> NamespaceItem:
        index = self.find_data_index(data_id)
        storage = self.get_cycle_storage()
        return storage[index]

    def remove_data(self, cycle, data_id) -> State:
        """
        Removes data from the Namespace storage using the DFG edge with id `data_id`.
        This also removes this data from future states, starting with `cycle`.

        Parameters
        ----------
        cycle : int
            The cycle to remove the data.
        data_id : int
            The DFG edge id.

        Returns
        -------
        The updated state without the removed data.
        """

        storage = self.get_cycle_storage()
        invalid_index = self.find_data_index(data_id)
        storage[invalid_index].invalidate()
        self._item_count -= 1

        if self.is_full():
            state_name = self.idle_state
        else:
            state_name = self.get_state_name(cycle)

        metadata = {"storage" : storage}

        return self.update_cycle_state(cycle, state_name, metadata)

    def add_instruction(self, cycle: int, instruction: Instruction) -> State:
        """
        Adds an instruction by adding the data in the instruction to the Namespace storage, if required.

        Parameters
        ----------
        cycle :  int
            The cycle in which to add the instruction.
        instruction : Instruction
            The instruction with destinations for adding the instruction.

        Returns
        -------
        The newly updated state in `cycle` with `instruction`.

        """
        for dest in instruction.get_dests():
            if dest.data_id == self.component_id:
                self.insert_data(cycle, dest.dest_id, dest.data_id)
        return self.get_cycle_state(cycle)
