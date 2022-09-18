from types import FunctionType
from codelets.graph import Node, Graph
from .graph_algorithms import compute_node_levels, get_shortest_paths
from . import ArchitectureGraph
from typing import List, Dict, Union, TYPE_CHECKING, Any
# from pygraphviz import AGraph
from collections import namedtuple, deque
from dataclasses import dataclass, field
import itertools
import dill
from codelets.adl.flex_template import Instruction, FlexTemplate

if TYPE_CHECKING:
    from codelets.templates.codelet_template import CodeletTemplate
    from .compute_node import ComputeNode
    from .storage_node import StorageNode
    from .communication_node import CommunicationNode
    from codelets.codelet_impl import Codelet

# Edge = namedtuple('Edge', ['src', 'dst', 'attributes', 'transfer_fn_map'])
OpTemplate = namedtuple('OpTemplate', ['instructions', 'functions'])


@dataclass
class Edge:
    src: str
    src_id: int
    dst: str
    dst_id: int
    bandwidth: int = field(default=0)
    attributes: Dict[str, int] = field(default_factory=dict)

    @property
    def attribute_names(self):
        return ["src", "src_id", "dst", "dst_id", "bandwidth", "attributes"]

    @property
    def bandwidth_bytes(self):
        return self.bandwidth // 8


class UtilFuncs(object):

    def __init__(self):
        self._funcs = {}
        self._func_def_names = {}

    @property
    def funcs(self):
        return self._funcs

    @property
    def func_def_names(self):
        return self._func_def_names

    def __getattr__(self, item):
        try:
            return self.get_util_fnc(item)
        except KeyError:
            raise AttributeError(item)

    def __getstate__(self):
        state = {"_funcs": self._funcs, "_func_def_names": self._func_def_names}
        return state

    def __setstate__(self, state):
        self._funcs = state["_funcs"]
        self._func_def_names = state["_func_def_names"]

    def add_fn(self, name, arg_vars: List[str], body):
        arg_str = ", ".join(arg_vars)
        self._func_def_names[name] = f"util_fn{len(list(self._funcs.keys()))}"
        self._funcs[name] = f"def {self._func_def_names[name]}({arg_str}):\n\t" \
                            f"return {body}"

    def get_util_fnc(self, name):
        util_fnc_code = compile(self._funcs[name], "<string>", "exec")
        util_fnc = FunctionType(util_fnc_code.co_consts[0], globals(), self._func_def_names[name])
        return util_fnc

    def run_param_fnc(self, fn_name, *args, **kwargs):
        util_fnc = self.get_util_fnc(fn_name)
        return util_fnc(*args, **kwargs)


class ArchitectureNode(Node):
    """
    Base class for Architecture Node
    Inherited from Node
    """
    graph_stack = deque([None])
    sub_graph_ctx_nodes = deque([[]])
    sub_graph_ctx_edges = deque([{}])

    def __init__(self, name, meta_cfg=None, index=None):
        super(ArchitectureNode, self).__init__(index=index)
        self._meta_cfg = meta_cfg
        self._has_parent = None
        self._subgraph = ArchitectureGraph()
        self._instr_length = None
        self._name = name
        if self._name:
            self.set_attr("field_name", self.name)
        # type
        self._anode_type = type(self).__name__
        self._all_subgraph_nodes = {}
        self._all_subgraph_edges = {}
        self._subgraph_nodes = {}
        self._subgraph_edges = []
        self._edge_map = {}
        self._in_edges = []
        self._out_edges = []

        # primitives
        self._primitives = {}

        # capability_sequence
        self._codelets = {}

        # occupied: [(op_node, primitive, begin_cycle, end_cycle)]
        # later consider changing to custom Heap because this needs to be accessed very frequently
        self._occupied = []  # NOTE state in TABLA compiler...
        self._operation_mappings = {"program": {"start": None, "end": None},
                                    "codelet": {"start": None, "end": None},
                                    "config": {},
                                    "transfer": {},
                                    "loop": None,
                                    "loop_end": None,
                                    "compute": {}}
        self._util_fns = UtilFuncs()
        if self.parent_graph is not None:
            ArchitectureNode.sub_graph_ctx_nodes[-1].append(self)

        self._node_levels = {}

    @property
    def attribute_names(self):
        raise NotImplementedError

    def __enter__(self):
        if len(ArchitectureNode.graph_stack) == 1 and ArchitectureNode.graph_stack[0] is None:
            ArchitectureNode.reset()
        ArchitectureNode.graph_stack.append(self)
        ArchitectureNode.sub_graph_ctx_nodes.append([])
        ArchitectureNode.sub_graph_ctx_edges.append({})
        return self

    def __exit__(self, exec_type, exec_value, exec_traceback):
        top_graph = ArchitectureNode.graph_stack.pop()
        ctx_nodes = ArchitectureNode.sub_graph_ctx_nodes.pop()
        ctx_edges = ArchitectureNode.sub_graph_ctx_edges.pop()
        assert top_graph == self
        for n in ctx_nodes:
            self.add_subgraph_node(n)

        for key, attr in ctx_edges.items():
            self.add_subgraph_edge(*key, **attr)

        if len(ArchitectureNode.graph_stack) == 1 and ArchitectureNode.graph_stack[0] is None:
            self.set_node_depths()

    def __str__(self):
        return f'op {self.index} ({self.get_type()}): \
                 preds={self.get_preds_indices()} ({self._attrs["in_degree"]}), \
                 succs={self.get_succs_indices()} ({self._attrs["out_degree"]})'

    # two categories of operand
    # modifying anode type arbitrarily should not be permitted
    @staticmethod
    def reset():
        ArchitectureNode.graph_stack = deque([None])
        ArchitectureNode.sub_graph_ctx_nodes = deque([[]])
        ArchitectureNode.sub_graph_ctx_edges = deque([{}])

    @property
    def parent_graph(self) -> Union[None, 'ArchitectureNode']:
        return ArchitectureNode.graph_stack[-1]

    @property
    def meta_cfg(self):
        return self._meta_cfg

    @property
    def parent_ctx_nodes(self) -> List:
        return ArchitectureNode.sub_graph_ctx_nodes[-1]

    @property
    def parent_ctx_edges(self) -> Dict:
        return ArchitectureNode.sub_graph_ctx_edges[-1]

    @property
    def name(self):
        return self._name

    @property
    def subgraph(self):
        return self._subgraph

    @property
    def edge_map(self):
        return self._edge_map

    @property
    def in_edges(self):
        return self._in_edges

    @property
    def out_edges(self):
        return self._out_edges

    @property
    def primitives(self) -> Dict[str, 'Instruction']:
        return self._primitives

    @property
    def codelets(self) -> Dict[str, 'Codelet']:
        return self._codelets

    @property
    def util_fns(self):
        return self._util_fns

    @util_fns.setter
    def util_fns(self, util_fns: UtilFuncs):
        self._util_fns = util_fns

    @property
    def instr_length(self):
        if not self._instr_length:
            raise RuntimeError(f"Instruction length is not set for {self.name}")
        return self._instr_length

    @instr_length.setter
    def instr_length(self, length):
        self._instr_length = length

    @property
    def node_levels(self):
        return self._node_levels

    @property
    def operation_mappings(self) -> Dict[str, Any]:
        return self._operation_mappings

    @operation_mappings.setter
    def operation_mappings(self, operation_mappings):
        self._operation_mappings = operation_mappings

    @property
    def all_codelet_names(self) -> List[str]:
        names = [] + list(self.codelets.keys())
        for n in self.get_subgraph_nodes():
            names += n.all_codelet_names
        return names

    @property
    def node_type(self):
        raise NotImplementedError

    @property
    def instr_length_set(self):
        return self._instr_length is not None and self._instr_length != -1

    def get_node_level(self, node_name: str):
        for lev, names in self.node_levels.items():
            if node_name in names:
                return lev
        raise KeyError(f"Unable to find node level for {node_name}")


    def add_util_fn(self, name, arg_vars: List[str], body):
        self.util_fns.add_fn(name, arg_vars, body)

    def run_util_fn(self, fn_name, *args):
        return self.util_fns.run_param_fnc(fn_name, *args)

    def set_parent(self, node_id):
        self._has_parent = node_id

    def get_viz_attr(self):
        raise NotImplementedError

    def has_primitive(self, name):
        if name in self.primitives:
            return True
        else:
            for n in self.get_subgraph_nodes():
                if n.has_primitive(name):
                    return True
        return False

    def has_codelet(self, name):
        return name in self.all_codelet_names

    def add_subgraph_edge(self, src, dst, bandwidth=0, attributes=None, transfer_fn_map=None):
        if self.parent_graph is not None:
            attributes = attributes or {}
            kwargs = {"bandwidth": bandwidth, "attributes": attributes}
            ArchitectureNode.sub_graph_ctx_edges[-1][(src, dst)] = kwargs
        else:
            if self._has_parent:
                raise RuntimeError("Already added node to graph, cannot continue to add subgraph edges")
            attr = attributes or {}

            if isinstance(src, (int, str)):
                src = self.get_subgraph_node(src)

            if isinstance(dst, (int, str)):
                dst = self.get_subgraph_node(dst)
            if bandwidth is None:
                raise TypeError(f"Invalid value for bandwidth: {bandwidth}")
            edge = Edge(src.index, src.name, dst.index, dst.name, bandwidth=bandwidth, attributes=attr)
            if src.name not in dst._in_edges:
                dst._in_edges.append(src.name)

            if dst.name not in src._out_edges:
                src._out_edges.append(dst.name)

            self._subgraph_edges.append(edge)
            self._edge_map[(src.name, dst.name)] = edge
            self.subgraph.add_edge(src, dst)

    def add_subgraph_node(self, node: 'ArchitectureNode'):
        if self._has_parent:
            raise RuntimeError("Already added node to graph, cannot continue to add subgraph nodes")

        self.merge_subgraph_nodes(node)
        node.set_parent(self.index)
        self.subgraph._add_node(node)
        self._subgraph_nodes[node.name] = node
        self._all_subgraph_nodes[node.name] = node

    def add_composite_node(self, node: 'ArchitectureNode', sub_nodes):
        for s in sub_nodes:
            s_node = self.get_subgraph_node(s)
            s_node.set_parent(None)
            node.add_subgraph_node(s_node)
            self.subgraph._nodes.pop(s_node.index)
            s_node.set_parent(node)
        self.add_subgraph_node(node)

    def has_op_template(self, op_type, op_subtype):
        assert op_type in self.operation_mappings
        return op_subtype in self.operation_mappings[op_type] and self.operation_mappings[op_type][op_subtype] is not None

    def get_program_template(self, subtype: str):
        return self.operation_mappings['program'][subtype]

    def get_cdlt_op_template(self, subtype: str):
        return self.operation_mappings['codelet'][subtype]

    def get_program_template_copy(self, subtype: str):
        template = self.operation_mappings['program'][subtype]
        temp_copy = [ft.template_copy() for ft in template.instructions]

        return temp_copy

    def get_cdlt_op_template_copy(self, subtype: str):
        template = self.operation_mappings['codelet'][subtype]
        temp_copy = [ft.template_copy() for ft in template.instructions]
        return temp_copy

    def get_operation_template(self, op):

        if op.op_type == 'transfer':
            template = []
            a, b = itertools.tee(op.path)
            next(b, None)
            for key in zip(a, b):
                template += self.operation_mappings['transfer'][key].instructions
        elif op.op_type == 'config':
            template = self.operation_mappings['config'][op.target_name][op.start_or_finish].instructions
        elif op.op_type == 'compute':
            template = self.operation_mappings['compute'][op.target][op.op_name].instructions

            if not isinstance(template, list):
                raise RuntimeError(f"Unable to find template for {op.op_str}, target: {op.target}, Op: {op.op_name}")
        elif op.op_type == 'loop':
            # TODO: Check why this is showing up as a warning
            if isinstance(self.operation_mappings['loop'].instructions, list):
                template = self.operation_mappings['loop'].instructions
            else:
                template = [self.operation_mappings['loop'].instructions]
        elif op.op_type == 'loop_end':
            # TODO: Check why this is showing up as a warning
            if isinstance(self.operation_mappings['loop_end'].instructions, list):
                template = self.operation_mappings['loop_end'].instructions
            else:
                template = [self.operation_mappings['loop_end'].instructions]
        else:
            raise TypeError(f"Invalid type for getting operation template: {type(op)}")

        return template

    def add_template(self, template, template_type, template_subtype=None, target=None, template_fns=None):
        if target is None:
            target = self.name

        if template_type == "config":
            if template_subtype == "start":
                self.add_start_template(target, template, template_fns)
            else:
                assert template_subtype == "end"
                self.add_end_template(target, template, template_fns)
        elif template_type == "transfer":
            self.add_transfer_template(*template_subtype, template, template_fns)
        elif template_type == "compute":
            self.add_compute_template(target, template_subtype, template, template_fns)
        elif template_type == "loop":
            self.add_loop_template(target, template, template_fns)
        else:
            raise RuntimeError(f"Invalid template type: {template_type}")

    def has_node(self, name: str) -> bool:
        if name in self._all_subgraph_nodes:
            return True
        else:
            for k, n in self._all_subgraph_nodes.items():
                if n.has_node(name):
                    return True
        return False

    def has_edge(self, src: str, dst: str) -> bool:
        if self.parent_graph == self:
            return (src, dst) in self.edge_map or (src, dst) in self.parent_ctx_edges
        else:
            return (src, dst) in self.edge_map

    # TODO: Need to validate that each parameter is correctly mapped
    def add_start_template(self, target, template, template_fns=None):
        if target not in self.operation_mappings['config']:
            self.operation_mappings['config'][target] = {}
        self.operation_mappings['config'][target]['start'] = OpTemplate(instructions=template, functions=template_fns)

    def add_end_template(self, target, template, template_fns=None):
        if target not in self.operation_mappings['config']:
            self.operation_mappings['config'][target] = {}
        self.operation_mappings['config'][target]['end'] = OpTemplate(instructions=template, functions=template_fns)

    def add_transfer_template(self, src, dst, template, template_fns=None):
        self.operation_mappings['transfer'][(src, dst)] = OpTemplate(instructions=template, functions=template_fns)

    def add_compute_template(self, target, op_name, template, template_fns=None):
        if target not in self.operation_mappings['compute']:
            self.operation_mappings['compute'][target] = {}
        self.operation_mappings['compute'][target][op_name] = OpTemplate(instructions=template, functions=template_fns)

    def add_loop_template(self, target, template, template_fns=None):
        self.operation_mappings['loop'] = OpTemplate(instructions=template, functions=template_fns)

    def add_loop_end_template(self, target, template, template_fns=None):

        self.operation_mappings['loop_end'] = OpTemplate(instructions=template, functions=template_fns)

    def add_program_start_template(self, target, template, template_fns=None):
        assert isinstance(template, list)
        for t in template:
            t.update_template_type("program")
        self.operation_mappings['program']['start'] = OpTemplate(instructions=template, functions=template_fns)

    def add_program_end_template(self, target, template, template_fns=None):
        assert isinstance(template, list)
        for t in template:
            t.update_template_type("program")
        self.operation_mappings['program']['end'] = OpTemplate(instructions=template, functions=template_fns)

    def add_codelet_start_template(self, target, template, template_fns=None):
        assert isinstance(template, list)
        for t in template:
            t.update_template_type("codelet")
        self.operation_mappings['codelet']['start'] = OpTemplate(instructions=template, functions=template_fns)

    def add_codelet_end_template(self, target, template, template_fns=None):
        assert isinstance(template, list)
        for t in template:
            t.update_template_type("codelet")
        self.operation_mappings['codelet']['end'] = OpTemplate(instructions=template, functions=template_fns)

    def get_subgraph_node(self, name: str) -> Union['ComputeNode', 'StorageNode', 'CommunicationNode']:

        assert isinstance(name, str)
        if self.has_node(name):
            return self._all_subgraph_nodes[name]
        elif self.parent_graph == self:
            for n in self.parent_ctx_nodes:
                if n.name == name:
                    return n

            for n in self.parent_ctx_nodes:
                if n.has_node(name):
                    return n.get_subgraph_node(name)
        else:
            for n, v in self._all_subgraph_nodes.items():
                if v.has_node(name):
                    return v.get_subgraph_node(name)

        raise KeyError(f"{name} not found in subgraph or input_components")

    def get_subgraph_edge(self, src: str, dst: str) -> Union['ComputeNode', 'StorageNode', 'CommunicationNode']:
        key = (src, dst)
        if key in self.edge_map:
            return self.edge_map[key]
        else:
            for n, v in self._all_subgraph_nodes.items():
                if v.has_edge(src, dst):
                    return v.edge_map[key]

        raise KeyError(f"{key} not found in subgraph or edges:"
                       f"Edges: {self.edge_map.keys()}")

    def set_node_depths(self):
        assert self.parent_graph != self
        self._node_levels = compute_node_levels(self.all_subgraph_nodes)

    def get_paths(self, src, dst):
        return get_shortest_paths(self.all_subgraph_nodes, src, dst)

    def get_off_chip_storage(self):
        min_level = min(list(self.node_levels.keys()))
        min_level_nodes = []
        for n in self.node_levels[min_level]:
            node = self.get_subgraph_node(n)
            if node.node_type == 'storage':
                min_level_nodes.append(node)

        if len(min_level_nodes) > 1:
            raise RuntimeError(f"Found more than one off-chip storage node:\n"
                               f"{[n.name for n in min_level_nodes]}")
        elif len(min_level_nodes) == 0:
            raise RuntimeError(f"Unable to find any off-chip storage nodes")

        return min_level_nodes[0]

    def get_node_depth(self, node_name: str):
        for depth, nodes in self.node_levels.items():
            if node_name in nodes:
                return depth
        raise RuntimeError(f"Unable to find node {node_name} in graph")

    def get_type(self):
        return self._anode_type

    def merge_subgraph_nodes(self, node):
        intersection = node.subgraph._nodes.keys() & self.subgraph._nodes.keys()
        if len(intersection) > 0:
            raise RuntimeError(f"Overlapping keys when merging nodes for {self.name} and {node.field_name}")
        for name, n in node._all_subgraph_nodes.items():
            self._all_subgraph_nodes[n.name] = n
        self.subgraph._nodes.update(node.subgraph._nodes)

    def set_all_instr_lengths(self, length):
        # NOTE: This assumes that a context manager has been constructed
        if len(ArchitectureNode.graph_stack) <= 1:
            raise RuntimeError(f"Unable to find top level graph because graph stack is empty")
        top_level_graph = ArchitectureNode.graph_stack[1]
        assert isinstance(top_level_graph, ArchitectureNode)
        top_level_graph._instr_length = length

        for _, n in top_level_graph.all_subgraph_nodes.items():
            if not n.instr_length_set:
                n._instr_length = length
            else:
                assert n.instr_length == length

    def add_primitive(self, primitive: 'Instruction', set_all_instr_lengths = True):
        if primitive.target is None:
            primitive.target = self.name
        if self.instr_length_set and primitive.instr_length != self.instr_length:
            raise RuntimeError(f"Invalid instruction length for architecture:\n"
                               f"Instruction: {primitive}\n"
                               f"Instruction Length: {primitive.instr_length}\n"
                               f"Required Instr length: {self.instr_length}")
        elif set_all_instr_lengths and not self.instr_length_set:
            self.set_all_instr_lengths(primitive.instr_length)

        self._primitives[primitive.name] = primitive

    def get_primitive_template(self, name, template_type="instruction") -> 'FlexTemplate':

        if name in self.primitives:
            assert template_type in ["program", "codelet", "instruction"]
            return FlexTemplate(self.primitives[name].instruction_copy(), template_type=template_type)
        elif self.parent_graph == self:
            for n in self.parent_ctx_nodes:
                if n.has_primitive(name):
                    return n.get_primitive_template(name, template_type=template_type)
        else:
            for n in self.get_subgraph_nodes():
                if n.has_primitive(name):
                    return n.get_primitive_template(name, template_type=template_type)
        raise KeyError(f"Primitive {name} not found!")

    def get_primitives(self) -> List['Instruction']:
        return list(self._primitives.keys())

    def add_codelet(self, codelet: 'Codelet'):
        # TODO: Validate memory paths
        if codelet.op_name in self._codelets:
            raise KeyError(f"Duplicate codelets for {codelet.op_name}")
        self._codelets[codelet.op_name] = codelet

    def get_codelet_template(self, name) -> Union['Codelet', 'CodeletTemplate']:
        if name in self.codelets:
            return self.codelets[name]
        else:
            for n in self.get_subgraph_nodes():
                if n.has_codelet(name):
                    return n.get_codelet_template(name)
        raise KeyError(f"Codelet {name} not found!")

    def get_codelets(self):
        return self._codelets.keys()

    def is_compatible(self, op_name):
        return op_name in self._primitives.keys()

    def set_occupied(self, op_code, primitive, begin_cycle, end_cycle):

        # check for overlaps, "expr" is occupied and "n" is new
        n = (begin_cycle, end_cycle)
        overlaps = [o for o in self._occupied if o[2] > n[0] and o[2] < n[1] or o[3] > n[0] and o[3] < n[1]]
        assert len(overlaps) == 0, 'this op_node cannot be mapped here, check before using set_occupied'

        # append to _occupied
        self._occupied.append((op_code, primitive, begin_cycle, end_cycle))

    def get_occupied(self):
        return self._occupied

    def is_available(self, begin_cycle, end_cycle):

        # check for overlaps, "expr" is occupied and "n" is new
        n = (begin_cycle, end_cycle)
        overlaps = [o for o in self._occupied if o[2] > n[0] and o[2] < n[1] or o[3] > n[0] and o[3] < n[1]]
        return len(overlaps) == 0

    def edge_exists(self, src_node: str, dst_node: str):
        if dst_node not in self.get_subgraph_node(src_node).out_edges:
            return False
        elif src_node not in self.get_subgraph_node(dst_node).in_edges:
            return False
        return True

    def read_capacity(self):
        raise NotImplementedError

    @property
    def viz_color(self):
        raise NotImplementedError

    def get_subgraph_nodes(self) -> List['ComputeNode']:
        return list(self._subgraph_nodes.values())

    @property
    def subgraph_edges(self) -> List[Edge]:
        return self._subgraph_edges

    @property
    def all_subgraph_nodes(self):
        return self._all_subgraph_nodes

    def get_graph_node_count(self):
        count = 0
        for n in self.get_subgraph_nodes():
            count += (1 + n.get_graph_node_count())
        return count

    def get_graph_edge_count(self):
        count = len(self.subgraph_edges)
        for n in self.get_subgraph_nodes():
            count += n.get_graph_edge_count()
        return count

    def print_isa(self):
        for name, node in self.all_subgraph_nodes.items():
            for op, instr in node.primitives.items():
                print(f"{name}{op}: {instr}")

    def is_adjacent(self, n1, n2):
        res = False
        for e in self.subgraph_edges:
            src = e.src_id
            dst = e.dst_id
            if (src, dst) == (n1, n2):
                return True
            if (dst, src) == (n1, n2):
                return True
        return False

    def adjacent_nodes(self, node_name):
        nodes = []
        for e in self.subgraph_edges:
            if e.src_id == node_name:
                nodes.append(e.dst_id)
            if e.dst_id == node_name:
                nodes.append(e.src_id)
        return list(set(nodes))

    def print_subgraph_edges(self, tabs=""):
        edge_pairs = [f"SRC: {self.subgraph.get_node_by_index(e.src).name}\t" \
                      f"DST:{self.subgraph.get_node_by_index(e.dst).name}" for e in self.subgraph_edges]
        print(f"Total edges: {len(edge_pairs)}\n"
              f"Unique: {len(set(edge_pairs))}")
        print("\n".join(edge_pairs))
        tabs = tabs + "\t"
        for n in self.get_subgraph_nodes():
            n.print_subgraph_edges(tabs=tabs)

    def to_json(self):
        blob = self.initialize_json()
        return self.finalize_json(blob)

    def from_json(self):
        pass

    def initialize_json(self):
        blob = {}
        blob['node_id'] = self.index
        blob['field_name'] = self.name
        blob['node_type'] = self.get_type()
        blob['node_color'] = self.viz_color
        if self.instr_length_set:
            blob['instruction_length'] = self.instr_length
        else:
            blob['instruction_length'] = -1
        blob['attributes'] = {}
        return blob

    def initialize_from_json(self, blob):
        pass

    def finalize_json(self, blob):
        blob['subgraph'] = {}
        blob['subgraph']['nodes'] = [sg.to_json() for sg in self.get_subgraph_nodes()]
        blob['subgraph']['edges'] = []
        for e in self.subgraph_edges:
            e_attr = {k: v for k, v in e.attributes.items()}
            sub_edge = {'src': e.src, 'dest': e.dst, 'bandwidth': int(e.bandwidth), 'attributes': e_attr}
            blob['subgraph']['edges'].append(sub_edge)

        if len(self.codelets) > 0:
            codelets_dill_fname = "node-" + str(blob['node_id']) + "-codelets.dill"
            with open(codelets_dill_fname, "wb") as f:
                dill.dump(self.codelets, f)
            blob['codelets'] = codelets_dill_fname

        if len(self.primitives) > 0:
            primitives_dill_fname = "node-" + str(blob['node_id']) + "-primitives.dill"
            with open(primitives_dill_fname, "wb") as f:
                dill.dump(self.primitives, f)
            blob['primitives'] = primitives_dill_fname

        utility_funcs_dill_fname = "node-" + str(blob['node_id']) + "-utility-functions.dill"
        with open(utility_funcs_dill_fname, "wb") as f:
            dill.dump(self.util_fns, f)
        blob['utility_funcs'] = utility_funcs_dill_fname

        operation_mappings_dill_fname = "node-" + str(blob['node_id']) + "-operation-mappings.dill"
        with open(operation_mappings_dill_fname, "wb") as f:
            dill.dump(self.operation_mappings, f)
        blob['operation_mappings'] = operation_mappings_dill_fname
        return blob

    # TODO: Finish filling this out
    def data_transfer_constraints(self, src: str, dst: str):
        src_node = self.get_subgraph_node(src)
        dst_node = self.get_subgraph_node(dst)
        edge = self.edge_map[(src, dst)]
