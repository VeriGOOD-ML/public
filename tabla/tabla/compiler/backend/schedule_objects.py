from typing import List, Optional, Dict, Tuple
from pytools import memoize_method
from . import Instruction
import numpy as np
from . import CYCLE_DELAYS

class ScheduleNode(object):

    def __init__(self, node_id):
        self.node_id = node_id
        self._children = []
        self._parents = []
        self._in_edges = []
        self._out_edges = []
        self._op_name = None
        self._instruction = None
        self._comm_instructions = []
        self._component_id = -1
        self._cat_comp_id = -1
        self._exec_cycle = -1
        self._depth = 0
        self._has_data_edge = False
        self._is_training = True
        self.computed = None

    def __repr__(self):
        return f"Node{self.node_id}"

    def set_computed(self, value):
        self.computed = value

    def add_op_name(self, op_name: str):
        self._op_name = op_name

    def set_training_mode(self, is_training):
        self._is_training = is_training

    def is_scheduled(self):
        return self.component_id >= 0

    def set_cat_comp(self, comp_id):
        self._cat_comp_id = comp_id

    @property
    def is_training(self):
        return self._is_training

    @property
    def comm_instructions(self):
        return self._comm_instructions

    def add_comm_instruction(self, instr):
        self._comm_instructions.append(instr)

    def to_json(self) -> dict:
        objects = {
            "node_id": self.node_id,
            "op_name": self.op_name,
            "num_comm_instr": len(self.comm_instructions),
            "exec_cycle": self.exec_cycle,
            "ns": self.namespace_name,
            "instr": str(self.get_instruction()),
            "is_data_node": self.is_data_node(),
            "has_data_edge": self.has_data_edge,
            "dtype": self.dtype,
            "component_id": self.component_id,
            "cat_component_id": self._cat_comp_id,
            "depth": self.depth,
            "children": self.children,
            "parents":  self.parents,
            "in_edges": self.in_edges,
            "out_edges": self.out_edges,
            "computed": self.computed
        }
        for k, v in objects.items():
            if v is not None and not isinstance(v, (int, str, list)):
                raise RuntimeError(f"Invalid type for key {k} with type {type(v)}")
        return objects

    def set_has_data_edge(self):
        self._has_data_edge = True

    @property
    def has_data_edge(self):
        return self._has_data_edge

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def out_edges(self) -> List[int]:
        return self._out_edges

    def add_out_edges(self, edge_ids: List[int]):
        self._out_edges = edge_ids

    def set_depth(self, depth: int):
        self._depth = depth

    def set_component_id(self, comp_id: int):
        if comp_id < 0:
            raise AttributeError(f"Error, cannot assign component id {comp_id}"
                                 f"to node {self.node_id}. Comp ids must be positive")
        elif self.is_data_node and self.component_id >= 0 and comp_id != self.component_id and self.namespace_name != "NM":
            raise AttributeError(f"Error, cannot assign component id {comp_id}"
                                 f"to data node {self.node_id} more than once.\n"
                                 f"Previous Component id: {self.component_id}\n"
                                 f"New component id: {comp_id}")
        self._component_id = comp_id

    def get_instruction(self) -> Instruction:
        return self._instruction

    @property
    def component_id(self) -> int:
        return self._component_id

    @property
    @memoize_method
    def in_edges(self) -> List[int]:
        return self._in_edges

    def add_in_edge(self, edge_id: int):
        self._in_edges.append(edge_id)

    def add_out_edge(self, edge_id: int):
        self._out_edges.append(edge_id)

    def add_in_edges(self, edge_ids: List[int]):
        self._in_edges = edge_ids

    @memoize_method
    def is_data_node(self) -> bool:
        if len(self._parents) == 1 and self._parents[0] == 0:
            return True
        else:
            return False

    @memoize_method
    def is_sink_node(self) -> bool:
        if len(self._children) == 1 and self._children[0] == 1:
            return True
        else:
            return False

    @memoize_method
    def is_source_sink(self) -> bool:
        if self.op_name in ["source", "sink"]:
            return True
        else:
            return False

    def set_exec_cycle(self, cycle: int):
        if self.exec_cycle > cycle:
            raise ValueError(f"Cannot reduce exec cycle for node {self.node_id}")
        self._exec_cycle = cycle

    def set_instruction(self, instr: Instruction):
        self._instruction = instr

    @property
    def exec_cycle(self) -> int:
        return self._exec_cycle

    @property
    def op_name(self) -> Optional[str]:
        return self._op_name

    def add_children(self, children: List[int]):
        self._children = children

    @property
    def children(self) -> List[int]:
        return self._children

    def add_parents(self, parents: List[int]):
        self._parents = parents

    @property
    def parents(self) -> List[int]:
        return self._parents

    def add_dtype(self, dtype: str):
        if dtype in ["model_input", "input", "model_output"]:
            self._dtype = "input"
        elif dtype in ["param", "constant"]:
            self._dtype = "param"
        elif dtype in ["model", "state"]:
            self._dtype = "state"
        elif dtype == "output":
            self._dtype = "output"
        else:
            self._dtype = "interim"

    @property
    @memoize_method
    def namespace_name(self):
        if self.dtype == "param":
            return "NM"
        elif self.dtype == "input":
            return "ND"
        elif self.is_output_dtype:
            return "NW"
        else:
            return "NI"

    @property
    def is_output_dtype(self):
        return self.dtype in ["state", "output"]

    @property
    @memoize_method
    def dtype(self) -> str:
        return self._dtype

class ScheduleEdge(object):

    def __init__(self, edge_id: int, is_src_edge: bool):
        self.edge_id = edge_id
        self._edge_name = ""
        self._src_id = -1
        self._dest_id = -1
        self._source_component = -1
        self._src_cat_comp = -1
        self._dst_cat_comp = -1
        self._path = []
        self._text_path = []
        self._dest_component = -1
        self._ready_cycle = -1
        self._comm_instructions = []
        self._sink_components = []
        self.alu_edge = False
        self.is_src_edge = is_src_edge
        self.is_sink_edge = False
        self._sink_node = -1
        self._path_cycles = []
        self._default_cycles = []


        self._sink_default_cycles = []
        self._sink_paths = []
        self._sink_path_cycles = []
        self._sink_path_text = []
        self.write_ns = False
        self._is_training = True

        self.value = None

    @property
    def is_training(self):
        return self._is_training

    def set_training_mode(self, is_training):
        self._is_training = is_training

    def to_json(self) -> dict:
        objects = {
            "edge_id": self.edge_id,
            "edge_name": self._edge_name,
            "ready_cycle": self._ready_cycle,
            "ns": self.namespace_name,
            "sink_node": self.sink_node,
            "dtype": self.dtype,
            "src_id": self._src_id,
            "dest_id": self._dest_id,
            "source_comp": self._source_component,
            "dest_comp": self._dest_component,
            "source_cat_comp": self._src_cat_comp,
            "dest_cat_comp": self._dst_cat_comp,
            "path_cycles": self._path_cycles,
            "path": self._path,
            "text_path": self._text_path,
            "src_edge":  self.is_src_edge,
            "sink_edge":  self.is_sink_edge,
            "sink_components":  self._sink_components,
        }
        for k, v in objects.items():
            if not isinstance(v, (int, str, list)):
                raise RuntimeError(f"Invalid type for key {k} with type {type(v)}")
        return objects

    @property
    def sink_paths(self) -> List[List[int]]:
        return self._sink_paths

    @property
    def path_cycles(self):
        if self.sink_node >= 0:
            return self._sink_path_cycles[-1]
        else:
            return self._path_cycles

    @property
    def num_pes(self):
        added_pes = []
        if self.alu_edge:
            return 1
        for idx, comp_name in enumerate(self.text_path):
            if comp_name == "PE" and self.path[idx] not in added_pes:
                added_pes.append(self.path[idx])
        return len(added_pes)

    @property
    def edge_name(self):
        return self._edge_name

    @property
    def data_id(self):
        if self.sink_node >= 0:
            return self.sink_node
        else:
            return self._src_id

    @property
    def overhead_cycles(self):
        cum_overhead = 0
        overhead = []
        for idx, cycle in enumerate(self.path_cycles):
            if cycle < 0:
                overhead.append(cum_overhead)
            else:
                cum_overhead += cycle - self._default_cycles[idx]
                overhead.append(cum_overhead)

        return overhead


    @property
    def ready_cycle(self) -> int:
        return self._ready_cycle

    @property
    def sub_paths(self):
        if self.sink_node >= 0:
            path = self.sink_paths[-1]
        else:
            path = self.path
        index = 0
        e_list = []
        while index < len(path) - 1:
            e_list.append(path[index:index + 3])
            index += 2
        return e_list

    @property
    def sub_path_cycles(self):
        index = 0
        e_list = []
        while index < len(self.path_cycles) - 1:
            e_list.append(self.path_cycles[index:index + 3])
            index += 2
        return e_list

    @property
    def dest_id(self) -> int:
        return self._dest_id

    @property
    def source_id(self) -> int:
        return self._src_id

    @property
    def sink_node(self):
        return self._sink_node

    @property
    def src_component(self):
        return self._source_component

    @property
    def dst_component(self):
        return self._dest_component

    @property
    def path(self):
        if self.sink_node >= 0:
            return self.sink_paths[-1]
        else:
            return self._path

    @property
    def text_path(self):
        if self.sink_node >= 0:
            return self._sink_path_text[-1]
        else:
            return self._text_path

    def set_ns_write(self):
        self.write_ns = True

    def unset_ns_write(self):
        self.write_ns = False

    @property
    @memoize_method
    def namespace_name(self):
        if self.dtype == "param":
            return "NM"
        elif self.dtype == "input":
            return "ND"
        elif self.is_output_dtype:
            return "NW"
        else:
            return "NI"

    @property
    def sink_components(self) -> List[int]:
        return list(set(self._sink_components))

    @property
    @memoize_method
    def dtype(self) -> str:
        return self._dtype

    def add_sink_component(self, comp_id: int):
        if not self.is_sink_edge or not self.is_output_dtype:
            raise RuntimeError(f"Cannot add sink node for non-sink edge"
                               f"\n\tEdge id: {self.edge_id}"
                               f"\n\tNode id: {comp_id}")

        self._sink_components.append(comp_id)

    def set_default_cycles(self, path: List[int]):
        if self.sink_node >= 0:
            self._sink_default_cycles.append(path)
        else:
            self._default_cycles = path


    def add_comm_instruction(self, instr: Instruction):
        self._comm_instructions.append(instr)

    def set_alu_edge(self):
        self.alu_edge = True

    def set_sink_edge(self):
        self.is_sink_edge = True

    def add_ns_path(self, ns_id: int):
        if self.sink_node < 0:
            raise TypeError(f"Cannot set ns path for non-sink edge {self.edge_id} with ns id: {ns_id}")
        self._sink_paths[-1].append(ns_id)
        self._sink_paths[-1].append(self._sink_paths[-1][-2])
        self._sink_path_text[-1].append("NS")
        self._sink_path_text[-1].append("PE")
        self._sink_path_cycles[-1].append(-1)
        self._sink_path_cycles[-1].append(-1)

    def set_text_path(self, text_path: List[str]):
        if self.sink_node >= 0:
            self._sink_path_text.append(text_path)
            self._sink_path_cycles.append([-1 for _ in range(len(self.text_path))])
        else:
            self._text_path = text_path
            self._path_cycles = [-1 for _ in range(len(self._text_path))]

    def set_path(self, path: List[int]):
        if self.sink_node >= 0 and self.alu_edge:
            raise ValueError(f"Unable to set path for alu and sink node")
        if self.sink_node >= 0:
            self._sink_paths.append([])
            for path_id in path:
                if path_id == self._source_component:
                    continue
                self._sink_paths[-1].append(path_id)
            if path[-1] == self._source_component:
                self._sink_paths[-1] = [self._source_component] + self._sink_paths[-1] + [path[-1]]
            else:
                self._sink_paths[-1] = [self._source_component] + self._sink_paths[-1]

        else:
            for path_id in path:
                if path_id == self._source_component or path_id == self._dest_component:
                    continue
                self._path.append(path_id)

            self._path = [self._source_component] + self._path + [self._dest_component]

    def set_edge_name(self, edge_name: str):
        self._edge_name = edge_name

    def add_source_node(self, src_id: int):
        self._src_id = src_id

    def add_dest_node(self, dest_id: int):
        self._dest_id = dest_id

    def add_source_component(self, component_id: int):
        if self.src_component >= 0 and not self.is_src_edge and component_id != self.src_component:
            raise RuntimeError(f"Error: Source component for edge {self.edge_id}"
                                 f" has already been set to {self.src_component}:\n"
                               f"\tNew: {component_id}")
        self._source_component = component_id

    def add_dest_component(self, component_id: int):

        if self.dst_component >= 0 and not self.is_src_edge:
            raise RuntimeError(f"Error: Source component for edge {self.edge_id}"
                                 f" has already been set to {self.dst_component}. Cannot set to {component_id}")
        self._dest_component = component_id


    def update_paths(self, index, text_id: str, comp_id: int, cycle: int):

        self.path.insert(index, comp_id)
        self.text_path.insert(index, text_id)

        self.path_cycles.insert(index, cycle)


    def set_ready_cycle(self, cycle: int):
        if self._ready_cycle == -1 or self._ready_cycle < cycle:
            self._ready_cycle = cycle

    def is_scheduled(self) -> bool:
        return self._source_component >= 0

    def get_component_index(self, comp_id: int) -> int:
        for path_index in range(len(self.path)):
            if self.path[path_index] == comp_id:
                return path_index
        raise ValueError(f"Error, could not find component index for {comp_id} in edge {self.edge_id} in path:"
                         f"\n{self.path}")


    def add_dtype(self, dtype: str):
        self._dtype = dtype

    def get_instr_src(self) -> Tuple[int, int]:
        rev_path = [comp for comp in reversed(self.path)]
        rev_text_path = [comp for comp in reversed(self.text_path)]

        if len(rev_path) != len(self.text_path):
            raise RuntimeError(f"The path of component is not equal to the text paths for "
                               f"edge {self.edge_id} with paths: {self.text_path} and "
                               f"{self.path}")

        for idx, comp in enumerate(rev_path[:-1]):
            if rev_text_path[idx] == "PE":
                return rev_path[idx + 1], rev_path[idx]

        raise RuntimeError(f"Unable to find PE id in edge {self.edge_id}:"
                           f"\n{self.text_path}\t{self.path}")

    def get_instr_dst(self) -> Tuple[int, int]:
        full_path = [comp for comp in self.path]

        if len(full_path) != len(self.text_path):
            raise RuntimeError(f"The path of component is not equal to the text paths for "
                               f"edge {self.edge_id} with paths: {self.text_path} and "
                               f"{self.path}")

        for idx, comp in enumerate(full_path):
            if idx == 0:
                continue
            if self.text_path[idx] == "PE":
                return full_path[idx - 1], comp

        raise RuntimeError(f"Unable to find PE id in edge {self.edge_id}")

    @property
    def is_output_dtype(self):
        return self.dtype in ["state", "output"]

    def set_sink_node(self, node_id: int):
        if not self.is_sink_edge or not self.is_output_dtype:
            raise RuntimeError(f"Cannot add sink node for non-sink edge"
                               f"\n\tEdge id: {self.edge_id}"
                               f"\n\tNode id: {node_id}")
        self._sink_node = node_id



    def set_path_cycle(self, index: int, cycle: int):
        if index > len(self.path_cycles):
            raise ValueError(f"Index {index} for updating path cycles in edge {self.edge_id}"
                             f" is too large with length {len(self.path_cycles)}.")
        for path_index, cost in enumerate(self.path_cycles):
            if cost > cycle and path_index < index:
                raise ValueError(f"Cannot set path cycle to {cycle} for edge {self.edge_id} at index {index}"
                                 f" because a previous cycle is greater than it:"
                                 f"\n\t:Path: {self.path}\n\tText Path: {self.text_path}\n\tCycles: {self.path_cycles}")
            elif cost < cycle and path_index > index and cost >= 0:
                raise ValueError(f"Cannot set path cycle to {cycle} for edge {self.edge_id} at index {index}"
                                 f" because a future cycle is greater than it:"
                                 f"\n\tPath: {self.path}\n\tText Path: {self.text_path}\n\tCycles: {self.path_cycles}")

        if self.path_cycles[index] > cycle:
            raise ValueError(f"Cannot set path cycle to {cycle} for edge {self.edge_id} at index {index}"
                             f" because the previous cycle is greater than it:"
                             f"\n\tPath: {self.path}\n\tText Path: {self.text_path}\n\tCycles: {self.path_cycles}")
        self.path_cycles[index] = cycle

    def __repr__(self):
        return f"Edge{self.edge_id}: {self.source_id} -> {self.dest_id}"

    def add_path_overhead(self, start_index: int, overhead: int):
        for cost_index in range(len(self.path_cycles)):
            if cost_index < start_index:
                continue
            if self.path_cycles[cost_index] >= 0:
                self.path_cycles[cost_index] += overhead
