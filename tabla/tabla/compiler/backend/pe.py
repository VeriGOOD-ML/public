from . import Component
from . import State
from . import Namespace
from . import Source, Dest, DataLocation, Instruction
from . import CYCLE_DELAYS
from collections import OrderedDict
from typing import List, Dict
from pytools import memoize_method



class PE(Component):
    """A PE class which executes instructions, contained within PUs.

    Attributes
    ----------
    possible_states : List[str]
            This represents the possible state names a PE can take on during a given cycle.
    idle_state : str
            This is the string name which represents an idle state. For a PE, the idle state is "free".
    component_id : int
            This is the unique component id, also known as `resource_id`.
    """
    possible_states = ['pass', 'nop', '+', '-', '*', '/', '>', '<', 'sigmoid', 'sqrt', 'square']
    idle_state = 'nop'
    required_keys = ['instr']

    def __init__(self, pu_id: int, namespace_size: int, namespace_interim_size: int):
        super(PE, self).__init__('pe', str(f"{pu_id}"))
        self.pu_id = pu_id
        self.component_id = self.resource_id
        self._namespace_map = {'NW': Namespace('NW', namespace_size),
                               'NI': Namespace('NI', namespace_interim_size),
                               'NG': Namespace('NG', namespace_size),
                               'NM': Namespace('NM', namespace_size),
                               'ND': Namespace('ND', namespace_size)}
        self.instr_counter = 0
        self.edge_writes = []
        self.edge_reads = []
        self._neighbor_bus = -1
        self._global_bus = -1
        self.is_head_pe = False
        self.instr_calls = 0
        # For simulation
        self._cycle = 0
        self.instructions = []
        self.rbus_data = []
        self.wbus_data = []
        self._inst_position = 0
        self.alu_data = None
        # For memory interface
        self.weight_nodes = []
        self.data_nodes = []
        self.meta_nodes = []
        self.sorted_instr = []
        self.sorted_cycles = []
        self.instr_op_map = {}
        self.sorted_state_keys = []


    def __str__(self):
        return f"Type: {self.component_type}\n\t" \
            f"Containing PU Component ID: {self.component_subtype}\n\t" \
            f"ID: {self.component_id}\n\t" \
            f"Category ID: {self.category_id}"

    def get_namespace(self, namespace_name: str) -> Namespace:
        """
        Retrieves one of 'NI', 'NW', 'NG', or 'ND' for a given PE.

        Parameters
        ----------
        namespace_name : str
            The name of the desired namespace.

        Returns
        -------
        The namespace object corresponding to `namespace_name`.

        """
        if namespace_name not in self._namespace_map:
            raise KeyError(f"{namespace_name} not found in PE{self.component_id} map")
        return self._namespace_map[namespace_name]

    def ns_id(self, ns_name: str) -> int:
        return self._namespace_map[ns_name].component_id

    @property
    @memoize_method
    def subcomponent_ids(self) -> List[int]:
        return [namespace.component_id for _, namespace in self._namespace_map.items()]

    def set_is_head_pe(self):
        self.is_head_pe = True

    def get_instr(self, cycle: int) -> Instruction:
        st = self.states[cycle]
        instr = st.get_metadata_by_key('instr')
        return instr

    def all_instr_len(self):
        return len(self.instructions)

    def all_instructions(self) -> List[Instruction]:
        if len(self.sorted_instr) > 0:
            return self.sorted_instr
        instr_list = []
        keys = sorted(list(self.states.keys()))
        for cycle in keys:
            if not self.is_idle(cycle):
                instr_list.append(self.get_instr(cycle))
        if len(instr_list) != len(self.instructions):
            instr_list_str = "\n".join([str(i) for i in instr_list])
            attr_instr_list_str = "\n".join([str(i) for i in self.instructions])
            raise RuntimeError(f"Unmatched instructions:\n"
                               f"Computed (length={len(instr_list)}): {instr_list_str}\n"
                               f"Attribute (length={len(self.instructions)}): {attr_instr_list_str}")
        return instr_list

    def all_instructions_with_cycles(self):
        if len(self.sorted_instr) > 0:
            return self.sorted_instr, self.sorted_cycles

        instr_list = []
        cycle_list = []
        keys = sorted(list(self.states.keys()))
        for cycle in keys:
            if not self.is_idle(cycle):
                cycle_list.append(cycle)
                instr_list.append(self.get_instr(cycle))
        return instr_list, cycle_list

    def all_instruction_map(self) -> Dict[int, Instruction]:
        instr_map = {}
        keys = sorted(list(self.states.keys()))
        for cycle in keys:
            if not self.is_idle(cycle):
                instr_map[cycle] = self.get_instr(cycle)
        return instr_map

    def add_data(self, cycle: int, namespace: str):
        """
        Adds data to `namespace` storage in `cycle`.

        Parameters
        ----------
        cycle : int
            Cycle to add storage for.
        namespace : str
            Namespace key name to add storage for.

        Returns
        -------

        """

    @property
    def utilization(self):
        busy_cycles = 0
        if self.max_cycle == -1:
            return 0

        keys = sorted(list(self.states.keys()))
        for cycle in keys:
            if not self.is_idle(cycle):
                busy_cycles += 1
        return (busy_cycles/(self.max_cycle + 1))

    def create_initial_state(self) -> State:
        """
        Initializes the PE to an idle state in the first cycle.

        Returns
        -------
        init_state : State
            The newly created initial state.
        """
        cycle = 0
        metadata = {'instr': None}
        state_name = self.idle_state
        init_state = self.add_cycle_state(cycle, state_name, metadata)
        return init_state

    def is_local_namespace(self, namespace_name: str) -> bool:
        """
        Checks to see if the `namespace_name` is a namespace belonging to the PE.

        Parameters
        ----------
        namespace_name : str
            Namespace name to check if it is local.

        Returns
        -------
            True if the `namespace_name` is a local namespace, otherwise False.

        """
        return namespace_name in self._namespace_map

    def are_sources_ready(self, cycle: int, sources: List[Source]) -> bool:
        """
        Checks if the namespaces corresponding to the source data locations for an
        instruction have the correct data in `cycle`.

        Parameters
        ----------
        cycle : int
            Cycle to check the state of data.
        sources : List[Source]
            List of data locations which may be included in the PE namespaces.

        Returns
        -------
        True if the instruction sources are ready in `cycle`, otherwise False.
        """

        for source in sources:
            if source.is_local():
                namespace = self.get_namespace(source.location)
                if not namespace.is_data_present(cycle, source.data_id):
                    return False
        return True

    def are_dests_full(self, cycle: int, dests: List[Dest]) -> bool:
        """
        Checks to see if the destination data locations are full in `cycle`.

        Parameters
        ----------
        cycle : int
            Cycle to check the state of the destinations.
        dests : List[Dest]
            List of destination data locations which may be included in the PE namespaces.

        Returns
        -------
        True if any destination data locations are either full or the data already exists in the
        destination, otherwise False.
        """

        for dest in dests:
            if dest.is_local():
                namespace = self.get_namespace(dest.location)
                if namespace.is_full():
                    return False
                elif namespace.is_data_valid(cycle, dest.dest_id):
                    return False
        return True

    def is_valid_instruction(self, cycle: int, instruction: Instruction) -> bool:
        """
        Checks if scheduling `instruction` is valid in `cycle`.

        Parameters
        ----------
        cycle : int
            Cycle to check if the instruction scheduling is valid.
        instruction : Instruction
            Instruction to check if valid.

        Returns
        -------
        True if scheduling `instruction` is valid in `cycle`, otherwise False.

        """

        if instruction.op_name not in self.possible_states:
            print(f"instruction op name not in states")
            return False

        if not self.is_idle(cycle):
            print(f"Not idle in cycle")
            return False

        sources = instruction.get_sources()
        if not self.are_sources_ready(cycle, sources):
            print(f"Sources not ready")
            return False

        dests = instruction.get_dests()
        if not self.are_dests_full(cycle, dests):
            print(f"Dests full")
            return False

        if not is_unique_namespaces(sources):
            print(f"Not unique src namespaces ")
            return False

        return True

    def first_idle_cycle(self):
        idle_cycles = list(filter(lambda x: self.is_idle(x) and x > 0, range(self.max_cycle + 1)))

        return idle_cycles

    def get_instr_by_operands(self, key):
        assert key in self.instr_op_map
        return self.instr_op_map[key]

    def busy_cycles(self):
        busy_cycles = []
        keys = sorted(list(self.states.keys()))
        for cycle in keys:
            if not self.is_idle(cycle) and cycle > 0:
                busy_cycles.append(cycle)
        return busy_cycles

    def add_instruction(self, cycle: int, instruction: Instruction) -> bool:
        """
        Adds `instruction` in `cycle` to the PE. This method assumes that `instruction` has already
        been checked to be valid.

        Parameters
        ----------
        cycle :  int
            Cycle to add the instruction for.
        instruction: Instruction
            Instruction to add.

        Returns
        -------
        True if the instruction was successfully added, otherwise False.
        """
        state_name = instruction.op_name
        if cycle == float('inf'):
            raise RuntimeError(f"Cannot add cycle to infinite cycle for {instruction.node_id}")
        instruction.cycle_insert = (cycle, len(self.instructions))
        metadata = {'instr': instruction}
        self.instructions.append(instruction)

        for s in instruction.srcs:

            self.edge_reads.append(s.source_id)
            if s.location in DataLocation.bus_types:
                self.rbus_data.append(s.data_id)

        for d in instruction.dests:

            self.edge_writes.append(d.dest_id)
            if d.location in DataLocation.bus_types:
                self.wbus_data.append(d.data_id)

        if self.is_idle(cycle):

            _ = self.add_cycle_state(cycle, state_name, metadata)
        else:

            raise RuntimeError(f"Unable to add instruction for pe {self.category_id}"
                               f" in cycle {cycle}."
                               f"\nMax cycle: {self.max_cycle}"
                               f"\nOccupying instr: {self.get_instr(cycle).node_id} --- {self.get_instr(cycle).op_name}"
                               f"\nNew instr: {instruction.node_id} --- {instruction.op_name}")

        return True

    def update_sorted(self, cycle, instruction):

        idx = self.sorted_cycles.index(cycle)
        self.sorted_instr.insert(idx, instruction)
        prev = cycle - 1
        for i, c in enumerate(self.sorted_cycles):
            if i >= idx:
                if (c - prev) == 1:
                    self.sorted_cycles[i] += 1
                    prev = c
                else:
                    self.sorted_cycles.insert(idx, cycle)
                    break

    def insert_instruction(self, cycle: int, instruction: Instruction) -> bool:
        """
        Adds `instruction` in `cycle` to the PE. This method assumes that `instruction` has already
        been checked to be valid.

        Parameters
        ----------
        cycle :  int
            Cycle to add the instruction for.
        instruction: Instruction
            Instruction to add.

        Returns
        -------
        True if the instruction was successfully added, otherwise False.
        """
        if len(self.sorted_instr) > 0:
            self.update_sorted(cycle, instruction)

        self.shift_states(cycle)
        state_name = instruction.op_name
        if cycle == float('inf'):
            raise RuntimeError(f"Cannot add cycle to infinite cycle for {instruction.node_id}")
        instruction.cycle_insert = (cycle, len(self.instructions))
        metadata = {'instr': instruction}
        self.instructions.append(instruction)

        if self.is_idle(cycle):
            _ = self.add_cycle_state(cycle, state_name, metadata)
        else:

            raise RuntimeError(f"Unable to add instruction for pe {self.category_id}"
                               f" in cycle {cycle}."
                               f"\nMax cycle: {self.max_cycle}"
                               f"\nOccupying instr: {self.get_instr(cycle).node_id} --- {self.get_instr(cycle).op_name}"
                               f"\nNew instr: {instruction.node_id} --- {instruction.op_name}")

        return True

    def get_instruction_by_id(self, data_id: int, start_cycle = 0):
        instrs = []
        for cycle in self.busy_cycles():
            if cycle < start_cycle:
                continue
            inst = self.get_instr(cycle)
            if inst.node_id == data_id:
                instrs.append((cycle, inst))
        return instrs

    def remove_instruction(self, cycle: int) -> bool:
        """
        Removes `instruction` in `cycle` from the PE. This method assumes that `instruction` has already
        been checked to be valid.

        Parameters
        ----------
        cycle :  int
            Cycle to add the instruction for.

        Returns
        -------
        True if the instruction was successfully added, otherwise False.
        """
        assert self.cycle_state_exists(cycle) and not self.is_idle(cycle)
        prev_instr = self.get_instr(cycle)
        self.instructions.pop(self.instructions.index(prev_instr))
        self.states.pop(cycle)

        if not self.is_idle(cycle):
            raise RuntimeError(f"Cycle is not idle after trying to remove instruction {prev_instr}\n"
                               f"Cycle: {cycle}")
        return True

    def set_neighbor_bus(self, bus_id: int):
        """
        Sets the neighbor bus component id for the PE.
        Parameters
        ----------
        bus_id : int
            Component id for the bus.

        Returns
        -------

        """
        self._neighbor_bus = bus_id

    def create_temp_pass_instr(self, src):
        instr = Instruction(src.data_id, "pass")
        instr.add_source(src.source_id, src.location, src.data_id, src.comp_id, src.index)
        instr.add_dest(src.source_id, src.location, src.data_id, src.comp_id, src.index)
        return instr

    @property
    def neighbor_bus(self) -> int:
        return self._neighbor_bus


    def set_global_bus(self, bus_id: int):
        """
        Sets the neighbor bus component id for the PE.
        Parameters
        ----------
        bus_id : int
            Component id for the bus.

        Returns
        -------

        """
        self._global_bus = bus_id

    @property
    def global_bus(self) -> int:
        return self._global_bus

    def run(self, cycle):
        instr = self.get_instr(cycle)
        if instr is not None:
            print(instr)
            print(instr.srcs[0].source_type, instr.srcs[0].index, instr.srcs[1])
            print(instr.op_name)

        init_nd = self.get_namespace('ND')
        print(type(init_nd))
        nd_storage = init_nd.get_cycle_storage(0)
        print(nd_storage[0])

        if instr.op_name == "*":
            id = instr.srcs[0].source_id
            print(id)

    def print_instructions(self):
        for inst in self.instructions:
            print(inst)

    def get_current_inst(self):
        if self.inst_position >= len(self.instructions):
            return None
        elif len(self.instructions) > 0:
            return self.instructions[self.inst_position]
        else:
            return None

    @property
    def cycle(self):
        return self._cycle

    @property
    def inst_position(self):
        return self._inst_position

    @cycle.setter
    def cycle(self, cycle):
        self._cycle = cycle

    @inst_position.setter
    def inst_position(self, pos):
        self._inst_position = pos

    def inst_all_complete(self):
        """Returns true if this PE completed executing all of its instructions."""
        if len(self.instructions) == 0:
            return True
        elif self.inst_position >= len(self.instructions):
            return True
        else:
            return False

    def __repr__(self):
        return f"PE{self.category_id}"


def is_unique_namespaces(locations: List[DataLocation]) -> bool:
    """
    Function for checking if the namespaces or buses in an instruction are all unqiue.
    This is done because hardware limits reading or writing from the same datalocation in the same cycle.

    Parameters
    ----------
    locations : List[DataLocation]
        List of source or destination datalocations which could be a namespace or bus.

    Returns
    -------
    True if the data locations are all unique, otherwise False.

    """
    namespace_names = [loc.location for loc in locations if loc.location != 'ZERO']
    return len(set(namespace_names)) == len(namespace_names)

