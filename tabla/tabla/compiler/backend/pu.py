from . import Component
from . import Bus
from . import PE
from . import State
from . import Source, Dest, DataLocation, Instruction
from pytools import memoize_method
from typing import List

class PU(Component):

    possible_states = ['busy', 'free']
    idle_state = 'free'

    def __init__(self, num_pes: int, pugb_id: int, *pe_args):
        super(PU, self).__init__('pu', '')
        self.component_id = self.resource_id
        # TODO: Change all ids to component id
        self._pe_map = {
            f"PE{Component.category_resource_counter['pe']-1}" : PE(self.component_id, *pe_args)
            for _ in range(num_pes)
        }

        self._bus_map = {
            f"PENB{i + num_pes*self.category_id}" : Bus('PENB')
            for i in range(num_pes)
        }

        pes = [self._pe_map[pe].category_id for pe in self._pe_map]

        self._bus_map['PEGB'] = Bus('PEGB', pes=pes)
        self._pugb = pugb_id
        self._punb = -1
        min_pe_id = float('inf')
        min_pe_key = ''
        for pe_id, pe in self._pe_map.items():
            if pe.category_id < min_pe_id:
                min_pe_id = pe.category_id
                min_pe_key = pe_id
            pe.set_global_bus(self._bus_map['PEGB'].component_id)
            pe.set_neighbor_bus(self._bus_map[f"PENB{pe.category_id}"].component_id)
        self._pe_map[min_pe_key].set_is_head_pe()

    def __str__(self):
        return f"Type: {self.component_type}\n\t" \
            f"PE Category IDs: {self.pe_ids}\n\t" \
            f"Component ID: {self.component_id}" \
            f"Category ID: {self.category_id}"

    def get_pe(self, pe_id: int) -> PE:
        return self._pe_map[f'PE{pe_id}']

    @property
    @memoize_method
    def neighbor_bus(self):
        return self._punb

    @property
    @memoize_method
    def global_bus(self):
        return self._pugb

    def set_neighbor_bus(self, bus_id: int):
        self._punb = bus_id

    @property
    def utilization(self):
        util = 0
        for _, pe in self._pe_map.items():
            util += pe.utilization
        return util*100.0/(len(self._pe_map.items()))

    @property
    @memoize_method
    def pe_ids(self) -> List[int]:
        return [pe.category_id for _, pe in self._pe_map.items()]

    def get_bus(self, bus_name: str) -> Bus:
        if bus_name not in self._bus_map:
            raise KeyError(f"{bus_name} not found in PE{self.component_id} map")
        return self._bus_map[bus_name]

    def create_initial_state(self) -> State:
        cycle = 1
        metadata = {}
        state_name = self.idle_state
        init_state = self.add_cycle_state(cycle, state_name, metadata)
        return init_state

    def are_sources_ready(self, cycle: int, sources: List[Source]) -> bool:
        for source in sources:
            if not source.is_local():
                bus = self.get_bus(source.location)
                if not bus.is_data_present(cycle, source.source_id):
                    return False
        return True

    def is_valid_instruction(self, cycle: int, instruction: Instruction) -> bool:
        # Question: Is this supposed to always return True?
        return True

    def find_avail_pe(self, cycle: int, instruction: Instruction) -> int:

        for pe_id in self.pe_ids:
            pe = self.get_component_category('pe', pe_id)
            if pe.is_valid_instruction(cycle, instruction):
                return pe_id
        return -1
