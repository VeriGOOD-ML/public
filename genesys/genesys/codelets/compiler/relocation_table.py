from typing import Dict, Union
from dataclasses import dataclass, field
import polymath as pm
import numpy as np
from codelets.adl.graph import StorageNode
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

    def item_names(self):
        return list(self.bases.keys())



class RelocationTable(object):
    MEM_LAYOUT = ['INSTR_MEM', 'STATE', 'INTERMEDIATE']
    def __init__(self, storage_node: StorageNode, mem_layout=None, offsets=None):
        self._top_level_node = storage_node.name
        self._mem_layout = mem_layout or RelocationTable.MEM_LAYOUT
        if offsets:
            self._offset_type = "static"
            assert list(offsets.keys()) == self.mem_layout
            self._namespace_offsets = {ml: off for ml, off in offsets.items()}
        else:
            self._offset_type = "dynamic"
            self._namespace_offsets = {ml: 0 for ml in self.mem_layout}


        assert isinstance(self.mem_layout, list)
        self._relocatables = {}
        self._storage_node = storage_node
        for ml in self.mem_layout:
            self._relocatables[ml] = Relocation(ml)


    def __repr__(self):
        return str([(k, list(v.item_names())) for k, v in self.relocatables.items()])

    @property
    def mem_layout(self):
        return self._mem_layout

    @property
    def relocatables(self):
        return self._relocatables

    @property
    def namespace_offsets(self):
        return self._namespace_offsets

    @property
    def storage_node(self):
        return self._storage_node

    @property
    def offset_type(self):
        return self._offset_type

    @property
    def is_empty(self):
        return all([self.relocatables[ml].total_length == 0 for ml in self.mem_layout])

    def get_relocation_by_name(self, name: str):
        for k, v in self.relocatables.items():
            if name in v.item_names():
                return v[name]
        raise KeyError(f"Unable to find relocation for {name}")

    def get_namespace_by_name(self, name: str):
        for k, v in self.relocatables.items():
            if name in v.item_names():
                return k
        raise KeyError(f"Unable to find relocation for {name}")

    def get_base_by_name(self, name: str):
        namespace = self.get_namespace_by_name(name)
        return self.get_relocation_base(namespace, name)

    def get_relocation(self, namespace: str, item_id: Union[str, int]):
        return self.relocatables[namespace][item_id]

    def update_namespace_offsets(self):
        offset = 0

        if self.offset_type == "dynamic":
            for ns in self.mem_layout:
                self.namespace_offsets[ns] = offset
                offset += self.relocatables[ns].total_length
        else:
            for i, ns in enumerate(self.mem_layout):
                if offset > self.namespace_offsets[ns]:
                    assert i > 0, f"Something is broken"
                    raise RuntimeError(f"Storage in {self.mem_layout[i - 1]} overruns {ns} memory.")
                offset += self.relocatables[ns].total_length


    def get_namespace_offset(self, namespace: str):
        assert namespace in self.mem_layout and namespace in self.namespace_offsets
        return self.namespace_offsets[namespace]

    def get_relocation_base(self, namespace: str, item_id: Union[str, int]):
        namespace_offset = self.get_namespace_offset(namespace)
        object_offset = self.get_relocation(namespace, item_id).start
        total_bit_offset = (namespace_offset + object_offset)

        assert total_bit_offset % self.storage_node.mem_width == 0, f"Invalid offset for address retrieval:\n" \
                                                           f"Storage width: {self.storage_node.width}\n" \
                                                           f"Namespace {namespace} offset: {namespace_offset}\n" \
                                                           f"Object {item_id} offset: {object_offset}"
        return self.storage_node.address_from_bits(namespace_offset + object_offset)


    def update_relocation_offset(self, offset_type, offset_id, size):
        current_offset = self.relocatables[offset_type].total_length
        relocatable = self.relocatables[offset_type]
        if offset_id not in self.relocatables[offset_type].bases:
            relocatable.bases[offset_id] = Fragment(offset_id, current_offset, current_offset + size)
            relocatable.total_length += size
        else:
            # TODO: Need to add back in error handling here for the same data with different datatypes
            stored_size = relocatable.bases[offset_id].end - relocatable.bases[offset_id].start
            if stored_size < size:
                prev_fragment = relocatable.bases[offset_id]
                relocatable.bases[offset_id] = Fragment(offset_id, prev_fragment.start, prev_fragment.start + size)
                relocatable.total_length += (size - stored_size)
        self.update_namespace_offsets()

    def add_data_relocation(self, node: pm.Node, cdlt: Codelet):
        for idx, operand in enumerate(cdlt.inputs):
            i = node.inputs[idx]
            data_size = np.prod(operand.shape)*operand.dtype.bits()
            if isinstance(i, pm.state):
                offset_type = 'STATE'
            else:
                offset_type = 'INTERMEDIATE'
            self.update_relocation_offset(offset_type, i.name, data_size)

        for idx, operand in enumerate(cdlt.outputs):
            o = node.outputs[idx]
            data_size = np.prod(operand.shape)*operand.dtype.bits()
            offset_type = 'INTERMEDIATE'
            self.update_relocation_offset(offset_type, o.name, data_size)

        self.update_namespace_offsets()
