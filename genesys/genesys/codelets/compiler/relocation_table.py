from typing import Dict, Union
from dataclasses import dataclass, field
import polymath as pm
import numpy as np
from codelets.codelet_impl.codelet import Codelet

@dataclass
class Fragment:
    offset_id: Union[int, str]
    start: int
    end: int

# TODO: Add datatype
@dataclass
class Relocation:
    offset_type: str
    bases: Dict[Union[int, str], Fragment] = field(default_factory=dict)
    total_length: int = field(default=0)

    def __getitem__(self, item):
        return self.bases[item]

    def get_fragment_str(self, item):
        return f"$({self.offset_type}[{self.bases[item]}])"

    def get_absolute_address(self, item):
        return self.bases[item]



class RelocationTable(object):

    def __init__(self):
        self._instr_mem = Relocation('INSTR_MEM')
        self._input = Relocation('INPUT')
        self._state = Relocation('STATE')
        self._intermediate = Relocation('INTERMEDIATE')
        self._scratch = Relocation('SCRATCH')

        self._relocatables = {'INSTR_MEM': self.instr_mem,
                              'INPUT': self.input,
                              'STATE': self.state,
                              'INTERMEDIATE': self.intermediate,
                              'SCRATCH': self.scratch}

    @property
    def instr_mem(self):
        return self._instr_mem

    @property
    def input(self):
        return self._input

    @property
    def state(self):
        return self._state

    @property
    def intermediate(self):
        return self._intermediate

    @property
    def scratch(self):
        return self._scratch

    @property
    def relocatables(self):
        return self._relocatables

    def update_relocation_offset(self, offset_type, offset_id, size):
        current_offset = self.relocatables[offset_type].total_length
        relocatable = self.relocatables[offset_type]
        if offset_id not in self.relocatables[offset_type].bases:
            relocatable.bases[offset_id] = Fragment(offset_id, current_offset, current_offset + size)
            relocatable.total_length += size
        else:
            stored_size = relocatable.bases[offset_id].end - relocatable.bases[offset_id].start
            assert stored_size == size

    def add_instr_relocation(self, cdlt: Codelet):
        instr_len = cdlt.num_instr
        self.update_relocation_offset('INSTR_MEM', cdlt.instance_id, instr_len)

    def add_data_relocation(self, node: pm.Node):
        # instr_len = cdlt.num_instr
        # self.update_relocation_offset('INSTR_MEM', cdlt.instance_id, instr_len)
        for i in node.inputs:
            data_size = np.prod(i.shape)
            # TODO: Figure out if input storage type is needed
            # TODO: Add datatype to datasize
            # if isinstance(i, pm.input):
            #     offset_type = 'INPUT'
            if isinstance(i, pm.state):
                offset_type = 'STATE'
            else:
                offset_type = 'INTERMEDIATE'
            self.update_relocation_offset(offset_type, i.name, data_size)

        for o in node.outputs:
            data_size = np.prod(o.shape)
            offset_type = 'INTERMEDIATE'
            self.update_relocation_offset(offset_type, o.name, data_size)