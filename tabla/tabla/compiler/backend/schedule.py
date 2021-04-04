
from typing import List, Dict, Tuple, TYPE_CHECKING
from time import time
if TYPE_CHECKING:
    from . import TablaTemplate
from . import determine_edge_path, get_cost_type, get_edge_dest,\
    cycle_cost, compute_default_cost, find_exec_cycle, optimize_instructions, \
    create_temp_instr, create_inter_pe_instr, create_comm_instruction
from . import Instruction
from . import ScheduleNode, ScheduleEdge
from . import CYCLE_DELAYS, NAMESPACES

from pytools import memoize_method
from tqdm import tqdm
from collections import defaultdict, deque, Counter
from pathlib import Path, PurePosixPath
import sys
import rapidjson
import numpy as np
import networkx as nx
# from line_profiler import LineProfiler
# import atexit
# profile = LineProfiler()
# atexit.register(profile.print_stats)



# from memory_profiler import profile
# mfp = open(f"memrep.log", "w+")

def check_graph_order(nodes, inp_depth):
    works = True
    depth_map = defaultdict(list)
    for n in nodes:
        if len(n["parents"]) == 0:
            continue
        min_parent = min(n["parents"])
        depth_map[inp_depth[n["id"]]].append(min_parent)
        if max(depth_map[inp_depth[n["id"]]]) != min_parent:
            print(f"Node id {n['id']} is not minimum for depth {depth_map[inp_depth[n['id']]]}\n\t"
                  f"Min: {max(depth_map[inp_depth[n['id']]])}\n\t"
                  f"Node min: {min_parent}")
            works = False
    return works

class Schedule(object):

    def __init__(self, tabla_template,
                 debug=False,
                 optimize=True,
                 progress_bar=True,
                 is_training_algorithm=True,
                 ct=None):
        self._dfg_nodes: List[ScheduleNode] = []
        self._dfg_edges: List[ScheduleEdge] = []
        self.is_training_algorithm = is_training_algorithm
        self.dfg_node_map = {}
        self.dfg_edge_map = {}
        self.dfg_path_name = None
        self.graph_name = None
        self.debug = debug
        self.optimize = optimize
        self.times = {}
        self.progress_bar = not progress_bar
        self.pes_per_pu = tabla_template.pes_per_pu
        self.num_pus = tabla_template.num_pus




    def print_schedule_graph(self, fpath: str):
        obj_list = {"nodes": [], "edges": []}

        for node in self._dfg_nodes:
            obj_list["nodes"].append(node.to_json())

        for edge in self._dfg_edges:
            obj_list["edges"].append(edge.to_json())

        with open(fpath, 'w') as outfile:
            rapidjson.dump(obj_list, outfile, indent=2)

    def find_valid_cycle(self, src_nodes: List[int]) -> int:
        max_cycle = -1
        for src_id in src_nodes:
            node = self.get_schedule_node(src_id)

            if not node.is_scheduled() and not node.is_data_node():
                raise RuntimeError(f"Node with id {node.node_id} is not scheduled, "
                                   f"and has no valid cycles.")
            instr = node.get_instruction()
            instr_cycle = -1 if not instr else instr.cycle_insert[0]
            tgt_cycle = max(node.exec_cycle, instr_cycle)
            if tgt_cycle > max_cycle:
                max_cycle = tgt_cycle
            if node.exec_cycle > max_cycle:
                max_cycle = node.exec_cycle

        return max_cycle + 1

    def is_dfg_loaded(self) -> bool:
        return len(self._dfg_nodes) > 0

    @memoize_method
    def get_schedule_node(self, node_id: int) -> ScheduleNode:
        if node_id not in self.dfg_node_map.keys():
            raise KeyError(f"Node with id {node_id} not found in schedule nodes.")
        else:
            return self.dfg_node_map[node_id]

    @memoize_method
    def get_schedule_edge(self, edge_id: int) -> ScheduleEdge:
        if edge_id not in self.dfg_edge_map.keys():
            raise KeyError(f"Edge with id {edge_id} not found in schedule nodes.")
        else:
            return self.dfg_edge_map[edge_id]

    @memoize_method
    def get_schedule_edge_by_nodes(self, src_node: int, dst_node: int) -> ScheduleEdge:
        for edge in self._dfg_edges:
            if edge.source_id == src_node and edge.dest_id == dst_node:
                return edge
        raise RuntimeError(f"Could not find edge with source id {src_node} and dest id {dst_node}")

    @memoize_method
    def get_parent_edge_id(self, child_node_id: int, parent_node_id: int) -> int:
        parent_node = self.get_schedule_node(parent_node_id)
        for edge_id in parent_node.out_edges:
            edge = self.get_schedule_edge(edge_id)
            if edge.dest_id == child_node_id:
                return edge_id
        raise RuntimeError(f"Could not find edge id for edge between "
                           f"node {parent_node.node_id} and {child_node_id}")

    def get_dfg_attrs(self, filepath: str, sort_type=None):
        self.dfg_path_name = Path(filepath)
        self.graph_name = PurePosixPath(filepath).stem
        if self.is_dfg_loaded():
            raise RuntimeError(f"DFG has already been loaded.")

        with open(filepath, 'r') as f:
            unsorted_graph = rapidjson.loads(f.read())
            graph = create_nx_graph(unsorted_graph)
            # graph, order = topological_sort(unsorted_graph, sort_type=sort_type)
        levels = find_graph_levels(graph)
        level_sizes = sorted([len(l) for l in levels])
        max_width = max(level_sizes)
        print(f"Max Level: {max_width}")

        print(f"Total: {level_sizes}")

    def load_dfg(self, filepath: str, sort_type=None):
        self.dfg_path_name = Path(filepath)
        self.graph_name = PurePosixPath(filepath).stem
        if self.is_dfg_loaded():
            raise RuntimeError(f"DFG has already been loaded.")
        with open(filepath, 'r') as f:
            unsorted_graph = rapidjson.loads(f.read())
            graph, order = topological_sort(unsorted_graph, sort_type=sort_type)

            for n in tqdm(graph, file=sys.stdout, dynamic_ncols=True,desc="Loading JSON DFG", disable=self.progress_bar):
                sched_node = ScheduleNode(n['id'])
                sched_node.add_op_name(n['operation'])
                sched_node.add_children(sorted(n['children']))
                sched_node.add_parents((n['parents']))
                sched_node.add_dtype(n['dataType'])
                sched_node.set_training_mode(self.is_training_algorithm)
                self.dfg_node_map[n['id']] = sched_node
                if 'computed' in n:
                    sched_node.set_computed(n['computed'])
                else:
                    sched_node.set_computed(int(np.random.randint(-3,3,1)[0]))
                self._dfg_nodes.append(sched_node)

        append_edge = self._dfg_edges.append
        edge_id = 0
        sink_id = -1
        max_depth = -1
        for i, dfg_node in enumerate(tqdm(self._dfg_nodes,
                                          file=sys.stdout,
                                          dynamic_ncols=True,
                                          desc="Creating Schedule nodes from DFG.",
                                          disable=self.progress_bar)):
            is_src_edge = dfg_node.is_data_node()
            if dfg_node.op_name == "sink":
                sink_id = dfg_node.node_id
            assert sorted(dfg_node.children) == dfg_node.children
            for c in dfg_node.children:

                sched_edge = ScheduleEdge(edge_id, is_src_edge)
                sched_edge.set_training_mode(self.is_training_algorithm)
                sched_edge.add_source_node(dfg_node.node_id)
                sched_edge.add_dest_node(c)
                dfg_node.add_out_edge(edge_id)
                sched_edge.add_dtype(dfg_node.dtype)
                append_edge(sched_edge)
                self.dfg_edge_map[sched_edge.edge_id] = sched_edge
                child_node = self.dfg_node_map[c]
                if is_src_edge:
                    child_node.set_has_data_edge()

                edge_id += 1

            for p in dfg_node.parents:
                dfg_node.add_in_edge(self.get_parent_edge_id(dfg_node.node_id, p))

            node_depth = self.get_node_depth(dfg_node)


            assert node_depth >= max_depth
            if max_depth < node_depth:
                max_depth = node_depth
            dfg_node.set_depth(node_depth)

        if sink_id < 0:
            raise RuntimeError(f"Graph does not include sink node.")

        sink_node = self.get_schedule_node(sink_id)

        for in_edge_id in tqdm(sink_node.in_edges,
                               file=sys.stdout,
                               dynamic_ncols=True,
                               desc="Creating schedule edges from JSON DFG.",
                               disable=self.progress_bar):
            in_edge = self.get_schedule_edge(in_edge_id)
            in_edge.set_sink_edge()
            parent_node = self.get_schedule_node(in_edge.source_id)
            if self.is_training_algorithm:
                for parent_input_id in parent_node.in_edges:
                    parent_input = self.get_schedule_edge(parent_input_id)
                    if parent_input.is_src_edge:
                        if parent_input.dtype == "state":
                            in_edge.set_sink_node(parent_input.source_id)
                        elif parent_input.dtype == "output":
                            pass
                        break

    @memoize_method
    def is_src_child(self, node: ScheduleNode) -> bool:
        for p_edge_id in node.in_edges:
            p_edge = self.get_schedule_edge(p_edge_id)
            if not p_edge.is_src_edge:
                return False
        return True

    @memoize_method
    def is_sink_parent(self, node: ScheduleNode) -> bool:
        if len(node.children) == 1:
            child = self.get_schedule_node(node.children[0])
            if child.op_name == "sink":
                return True
        return False

    def compute_cycle_cost(self, cycle: int, edge1: ScheduleEdge, edge2: ScheduleEdge, dst_comp: int, arch: 'TablaTemplate') -> int:

        instr1 = self.get_schedule_node(edge1.source_id).get_instruction()
        instr2 = self.get_schedule_node(edge2.source_id).get_instruction()

        if edge1.src_component < 0:
            edge1_dest = get_edge_dest(arch, edge2, dst_comp)
            src1_comp = arch.component_map[edge2.src_component]
            comp1_cost = arch.pe_costs[edge2.src_component][dst_comp]
        else:
            src1_comp = arch.component_map[edge1.src_component]
            edge1_dest = get_edge_dest(arch, edge1, dst_comp)
            comp1_cost = arch.pe_costs[edge1.src_component][dst_comp]

        if instr1 and instr1.check_dest(edge1_dest):
            overhead = 1
            while not src1_comp.is_idle(cycle + overhead):
                overhead += 1
            comp1_cost += overhead

        if edge2.src_component < 0:
            src2_comp = arch.component_map[edge1.src_component]
            edge2_dest = get_edge_dest(arch, edge1, dst_comp)
            comp2_cost = arch.pe_costs[edge1.src_component][dst_comp]
        else:
            src2_comp = arch.component_map[edge2.src_component]
            edge2_dest = get_edge_dest(arch, edge2, dst_comp)
            comp2_cost = arch.pe_costs[edge2.src_component][dst_comp]

        if instr2 and instr2.check_dest(edge2_dest):
            overhead = 1
            while not src2_comp.is_idle(cycle + overhead):
                overhead += 1
            comp2_cost += overhead

        if comp1_cost > comp2_cost:
            cost = comp1_cost
        else:
            cost = comp2_cost

        return cycle + cost

    def greedy_heuristic(self, curr_cycle, min_cycle, min_instr_len, cand_cycle, cand_pe, ns_full, total_instr):

        if cand_pe.is_head_pe:
            return False
        elif not cand_pe.is_idle(cand_cycle + curr_cycle):
            return False
        elif ns_full:
            return False
        elif min_cycle < cand_cycle:
            return False
        elif cand_cycle == min_cycle:
        # elif cand_cycle == min_cycle and min_instr_len < total_instr:
            return False
        else:
            return True

    def brute_force_test(self, curr_cycle,
                         edge_costs,
                         edges,
                         src_instrs,
                         arch,
                         instr_size_map,
                         min_pe):
        min_instr_len = float('inf')
        min_cycle = float('inf')
        if min_pe >= 0:
            pe_id = -1

            cycles_ready = []
            ns_full = False
            for i, edge in enumerate(edges):
                # source_instruction = self.get_schedule_node(e.source_id).get_instruction()
                # cost = cycle_cost(e, source_instruction, min_pe, arch)
                cost = edge_costs[i]

                if edge.namespace_name in ["NW", "ND"]:
                    namespace = arch.component_map[min_pe].get_namespace(edge.namespace_name)
                    if namespace.is_data_present(curr_cycle + 1, edge.data_id):
                        cost = 1
                    else:
                        ns_full = namespace.is_full()

                cycles_ready.append(cost)
            cost_cycle = max(cycles_ready)
            if arch.component_map[min_pe].is_idle(cost_cycle + curr_cycle) and min_cycle > cost_cycle and not ns_full:
                pe_id = min_pe
                min_cycle = cost_cycle
            return pe_id, min_cycle + curr_cycle
        else:

            # for pe_id in arch.ordered_pes:
            for pe_id in arch.perm_pes:
                cycles_ready = []
                ns_full = False
                instr_sizes = 0
                for i, e in enumerate(edges):
                    cost = cycle_cost(e, src_instrs[e], pe_id, arch, curr_cycle)
                    # instr_sizes += arch.path_instr_len(e.src_component, pe_id)
                    instr_sizes += instr_size_map[(e.src_component, pe_id)]
                    if e.is_src_edge:
                        namespace = arch.component_map[pe_id].get_namespace(e.namespace_name)
                        if namespace.is_data_present(curr_cycle + 1, e.data_id):
                            cost = 1
                        else:
                            ns_full = namespace.is_full()

                    cycles_ready.append(cost)

                cost_cycle = max(cycles_ready)
                # instr_sizes += len(arch.component_map[pe_id].all_instructions())
                instr_sizes += instr_size_map[pe_id]
                if self.greedy_heuristic(curr_cycle, min_cycle, min_instr_len, cost_cycle, arch.component_map[pe_id],
                                         ns_full, instr_sizes):
                    min_pe = pe_id
                    min_cycle = cost_cycle
                    min_instr_len = instr_sizes
            return min_pe, min_cycle + curr_cycle

    def brute_force_pe(self, curr_cycle: int, node: ScheduleNode, arch: 'TablaTemplate') -> Tuple[int,int]:
        input_edges = []
        min_instr_len = float('inf')
        min_pe = -1
        min_cycle = float('inf')

        # First, if either of the input nodes are source nodes, the instruction needs to be added to
        # the PE where the data is located
        for in_edge_id in node.in_edges:
            e = self.get_schedule_edge(in_edge_id)
            if e.namespace_name in ["ND", "NW"] and self.get_schedule_node(e.source_id).component_id >= 0:
                min_pe = self.get_schedule_node(e.source_id).component_id

            input_edges.append(e)

        if all([in_edge.src_component < 0 for in_edge in input_edges]):
            raise RuntimeError(f"All input nodes have an unassigned PE: "
                               f"\n\tNode: {node.node_id}\n\tParents: {node.in_edges}")


        if min_pe >= 0:
            pe_id = -1

            cycles_ready = []
            ns_full = False
            for e in input_edges:
                source_instruction = self.get_schedule_node(e.source_id).get_instruction()
                cost = cycle_cost(e, source_instruction, min_pe, arch)

                if e.namespace_name in ["NW", "ND"]:
                    namespace = arch.component_map[min_pe].get_namespace(e.namespace_name)
                    if namespace.is_data_present(curr_cycle + 1, e.data_id):
                        cost = 1
                    else:
                        ns_full = namespace.is_full()

                cycles_ready.append(cost)
            cost_cycle = max(cycles_ready)
            if arch.component_map[min_pe].is_idle(cost_cycle + curr_cycle) and min_cycle > cost_cycle and not ns_full:
                pe_id = min_pe
                min_cycle = cost_cycle
            return pe_id, min_cycle + curr_cycle

        src_instrs = {e: self.get_schedule_node(e.source_id).get_instruction() for e in input_edges}

        for pe_id in arch.ordered_pes:
            cycles_ready = []
            ns_full = False
            instr_sizes = 0
            for e in input_edges:
                cost = cycle_cost(e, src_instrs[e], pe_id, arch, curr_cycle)
                instr_sizes += arch.path_instr_len(e.src_component, pe_id)
                if e.is_src_edge:
                    namespace = arch.component_map[pe_id].get_namespace(e.namespace_name)
                    if namespace.is_data_present(curr_cycle + 1, e.data_id):
                        cost = 1
                    else:
                        ns_full = namespace.is_full()

                cycles_ready.append(cost)

            cost_cycle = max(cycles_ready)
            instr_sizes += arch.component_map[pe_id].all_instr_len()
            if self.greedy_heuristic(curr_cycle, min_cycle, min_instr_len, cost_cycle, arch.component_map[pe_id], ns_full, instr_sizes):
                min_pe = pe_id
                min_cycle = cost_cycle
                min_instr_len = instr_sizes
        return min_pe, min_cycle + curr_cycle

    def set_edge_path(self, edge: ScheduleEdge, arch: 'TablaTemplate') -> List[int]:
        path = determine_edge_path(arch, edge)

        edge.set_path(path)
        text_path = [get_cost_type(arch.component_map[pe_id]) for pe_id in path]
        edge.set_text_path(text_path)
        start_cycle = self.get_schedule_node(edge.source_id).exec_cycle
        default_costs = compute_default_cost(start_cycle, text_path)
        edge.set_default_cycles(default_costs)
        return path

    @property
    def dfg_nodes(self) -> List[ScheduleNode]:
        return self._dfg_nodes

    @property
    def dfg_edges(self) -> List[ScheduleEdge]:
        return self._dfg_edges


    def create_instruction(self, arch: 'TablaTemplate', node: ScheduleNode) -> Instruction:
        instr = Instruction(node.node_id, node.op_name)
        instr.set_component_id(node.component_id)

        in_edge_cycles = []
        max_cycle = node.exec_cycle
        for i, in_edge_id in enumerate(node.in_edges):


            in_edge = self.get_schedule_edge(in_edge_id)
            src_node = self.get_schedule_node(in_edge.source_id)
            pe_id = in_edge.sub_paths[0][0]
            src_instr = src_node.get_instruction()

            # TODO: Update exec and ready cycles according to overheads for comm instr
            # TODO: Update along each path with comm instr
            if src_instr:
                location = arch.get_data_location(in_edge.sub_paths[0][1], in_edge)
                default_cycle = 1 if in_edge.alu_edge else arch.pe_costs[in_edge.src_component][in_edge.dst_component]
                default_cycle += src_node.exec_cycle
                start_num = in_edge.num_pes
                if src_instr.check_dest(location) and location in ["PEGB", "PUGB", "PENB", "PUNB"]:
                    comm_instr = create_comm_instruction(self, arch, in_edge, src_instr, src_node)
                    src_node.add_comm_instruction(comm_instr)
                else:
                    src_loc, src_index = arch.compute_data_source(src_node.exec_cycle, in_edge.sub_paths[0], in_edge)
                    if not src_instr.check_dest(location):
                        arch.add_instruction_dest(src_node.exec_cycle, src_instr, in_edge, src_loc, in_edge.data_id, pe_id, index=src_index)

                    if len(in_edge.path) > 3:
                        dst_pe_index = in_edge.get_component_index(in_edge.sub_paths[0][2])
                        dst_cost = CYCLE_DELAYS[src_loc]["PE"]
                        in_edge.set_path_cycle(dst_pe_index, in_edge.path_cycles[dst_pe_index - 1] + dst_cost)

                if in_edge_id not in src_instr.get_dest(location).all_dests:
                    src_instr.get_dest(location).add_edge(in_edge_id)

                in_edge.set_path_cycle(0, src_node.exec_cycle)

                if in_edge.num_pes > 2:
                    create_inter_pe_instr(self, arch, in_edge, src_node)
            pe_exec_cycle = find_exec_cycle(arch, src_node, in_edge, len(in_edge.path) - 1)

            if pe_exec_cycle > max_cycle:
                max_cycle = pe_exec_cycle

            in_edge.set_path_cycle(len(in_edge.path) - 1, max_cycle)
            dst_loc, dst_index = arch.compute_data_dest(src_node.exec_cycle, in_edge.sub_paths[-1][1], in_edge.sub_paths[-1][0], in_edge)
            instr.add_source(in_edge.edge_id, dst_loc, in_edge.data_id, arch.component_map[pe_id].category_id, index=dst_index)
            in_edge_cycles.append((in_edge, pe_exec_cycle))


        dest_pe = arch.component_map[node.component_id]

        if len(instr.srcs) == 2 and instr.srcs[0].location not in NAMESPACES and str(instr.srcs[0].location) == str(instr.srcs[1].location):

            max_cycle = create_temp_instr(self, arch, dest_pe, in_edge_cycles, instr, max_cycle)

        max_cycle = arch.add_pe_instruction(max_cycle, self, instr, dest_pe.component_id, instruction_fn="create_instr")
        node.set_exec_cycle(max_cycle)

        return instr

    def initialize_brute_force_args(self, node, arch):
        input_edges = []
        edge_costs = []
        src_instrs = {}
        instr_size_map = {}
        min_pe = -1
        for in_edge_id in node.in_edges:
            e = self.get_schedule_edge(in_edge_id)
            if e.namespace_name in ["ND", "NW"] and self.get_schedule_node(e.source_id).component_id >= 0:
                min_pe = self.get_schedule_node(e.source_id).component_id
            source_instruction = self.get_schedule_node(e.source_id).get_instruction()
            src_instrs[e] = source_instruction
            input_edges.append(e)
            for pe_id in arch.ordered_pes:
                instr_size_map[(e.src_component, pe_id)] = arch.path_instr_len(e.src_component, pe_id)
                instr_size_map[pe_id] = len(arch.component_map[pe_id].all_instructions())

        if min_pe >= 0:
            for i, e in enumerate(input_edges):
                cost = cycle_cost(e, src_instrs[e], min_pe, arch)
                edge_costs.append(cost)

        if all([in_edge.src_component < 0 for in_edge in input_edges]):
            raise RuntimeError(f"All input nodes have an unassigned PE: "
                               f"\n\tNode: {node.node_id}\n\tParents: {node.in_edges}")

        return edge_costs, input_edges, src_instrs, instr_size_map, min_pe

    # TODO: Change everything to operate on edges instead of nodes, and update src input nodes
    # @profile
    def schedule_node(self, node: ScheduleNode, arch: 'TablaTemplate') -> 'TablaTemplate':
        # Scheduling:
        # 1. Determine source operand availability
        # 2. Assign PE based on ordered preferences (same pe as sources, etc)
        # 3. Assign dests based on ordered preferences
        # TODO: need to consider sink parents for storage
        # TODO: need to add comm instrucitons here
        # TODO : Another reminder to change to edges...

        cycle_ready = self.find_valid_cycle(node.parents)
        dest_pe_id = -1

        start_cycle = cycle_ready
        min_cost = -1

        while dest_pe_id < 0:
            # TODO: Need to validate that the paths and components returned here are the samme as below
            dest_pe_id, min_cost = self.brute_force_pe(start_cycle, node, arch)
            start_cycle += 1

        start_cycle -= 1
        if min_cost - start_cycle < 0:
            print(f"Min cost: {min_cost}\n"
                  f"Dest pe: {dest_pe_id}\n"
                  f"Start cycle: {start_cycle}")
        assert min_cost - start_cycle >= 0

        node.set_component_id(dest_pe_id)
        node.set_cat_comp(arch.component_map[dest_pe_id].category_id)

        if node.has_data_edge:
            arch = self.set_data_edge_comp(node, arch)

        for i, in_edge_id in enumerate(node.in_edges):
            #     break
            in_edge = self.get_schedule_edge(in_edge_id)
            # if in_edge_id != node.in_edges[0] or i == 0:
            in_edge.add_dest_component(dest_pe_id)
            path = self.set_edge_path(in_edge, arch)

            src_node = self.get_schedule_node(in_edge.source_id)
            source_instruction = self.get_schedule_node(in_edge.source_id).get_instruction()

            if source_instruction and src_node.exec_cycle != source_instruction.cycle_insert[0]:
                print(f"Doesnt work for {src_node.node_id} - {source_instruction}")
            cost = cycle_cost(in_edge, source_instruction, dest_pe_id, arch)
            cost = cost + src_node.exec_cycle
            in_edge.set_ready_cycle(cost)

        node.set_exec_cycle(min_cost)
        self.initialize_output_edges(node, arch)
        instr = self.create_instruction(arch, node)

        if node.exec_cycle != instr.cycle_insert[0]:
            print(f"Node id: {node.node_id} - {node.exec_cycle}\n"
                  f"Instr cycle: {instr.cycle_insert}\n")
        node.set_instruction(instr)

        return arch

    def initialize_output_edges(self, node: ScheduleNode, arch: 'TablaTemplate'):
        for child_edge_id in node.out_edges:
            child_edge = self.get_schedule_edge(child_edge_id)
            child_edge.add_source_component(node.component_id)
            if child_edge.is_sink_edge and node.dtype == "state":
                data_node_id = child_edge.sink_node
                if data_node_id < 0:
                    raise RuntimeError(f"Sink edge has unset sink node id:"
                                       f"\n\tEdge Id: {child_edge.edge_id} for {node.node_id}\n"
                                       f"Info: {child_edge.source_id} -> {child_edge.dest_id}")
                data_node = self.get_schedule_node(data_node_id)

                for out_edge_id in data_node.out_edges:
                    out_edge = self.get_schedule_edge(out_edge_id)
                    child_edge.add_sink_component(out_edge.src_component)


    def set_edge_names(self, node: ScheduleNode):
        if node.is_data_node():
            for out_edge_id in node.out_edges:
                out_edge = self.get_schedule_edge(out_edge_id)
                out_edge.set_edge_name(node.op_name)
            node.set_exec_cycle(0)
        else:
            in_edge1_name = self.get_schedule_edge(node.in_edges[0])._edge_name
            in_edge2_name = self.get_schedule_edge(node.in_edges[1])._edge_name
            for out_edge_id in node.out_edges:
                out_edge = self.get_schedule_edge(out_edge_id)
                out_edge.set_edge_name(f"{in_edge1_name}{node.op_name}{in_edge2_name}")

    def check_existing_ns(self, arch, parents):
        nodes = [self.get_schedule_node(p).component_id for p in parents if self.get_schedule_node(p).component_id >= 0]
        if len(nodes) >= 2:
            raise RuntimeError(f"Multiple conflicting PEs!!")
        elif len(nodes) == 1:
            tgt_pe_id = nodes[0]
            tgt_pe = arch.component_map[tgt_pe_id]
            start_cycle = 1
            while not tgt_pe.is_idle(start_cycle):
                start_cycle += 1
            start_cycle += 1
        else:
            tgt_pe_id = -1
            start_cycle = 1

        return tgt_pe_id, start_cycle


    def schedule_src_child(self, node: ScheduleNode, arch: 'TablaTemplate') -> 'TablaTemplate':
        self.set_edge_names(node)
        namespaces = list(set([self.get_schedule_edge(in_id).namespace_name for in_id in node.in_edges]))
        empty_pe_id, start_cycle = self.check_existing_ns(arch, node.parents)

        while empty_pe_id < 0:
            empty_pe_id = arch.find_optimal_exec_pe(start_cycle, namespaces[0])
            start_cycle += 1
        start_cycle -= 1

        node.set_exec_cycle(start_cycle)
        node.set_component_id(empty_pe_id)
        node.set_cat_comp(arch.component_map[empty_pe_id].category_id)

        for in_edge_id in node.in_edges:
            in_edge = self.get_schedule_edge(in_edge_id)
            _ = arch.add_namespace_data(0, empty_pe_id, in_edge.namespace_name, in_edge)
            in_edge.add_source_component(empty_pe_id)
            in_edge.add_dest_component(empty_pe_id)
            in_edge.set_ready_cycle(start_cycle)
            self.set_edge_path(in_edge, arch)

        for p_node_id in node.parents:
            p_node = self.get_schedule_node(p_node_id)
            if p_node.is_data_node():
                p_node.set_exec_cycle(0)
                p_node.set_component_id(empty_pe_id)
                p_node.set_cat_comp(arch.component_map[empty_pe_id].category_id)

        self.initialize_output_edges(node, arch)
        instr = self.create_instruction(arch, node)
        node.set_instruction(instr)


        return arch

    def schedule_single_input(self, node: ScheduleNode, arch: 'TablaTemplate') -> 'TablaTemplate':
        edge = self.get_schedule_edge(node.in_edges[0])
        parent_node = self.get_schedule_node(edge.source_id)
        start_cycle = parent_node.exec_cycle + 1
        pe_id = -1
        if arch.component_map[edge.src_component].is_idle(start_cycle):

            edge_name = edge._edge_name
            for out_edge_id in node.out_edges:
                out_edge = self.get_schedule_edge(out_edge_id)
                out_edge.set_edge_name(f"{node.op_name}({edge_name})")
            edge.set_alu_edge()
            pe_id = edge.src_component
            node.set_exec_cycle(start_cycle)

            node.set_component_id(pe_id)
            node.set_cat_comp(arch.component_map[pe_id].category_id)

            if node.has_data_edge:
                arch = self.set_data_edge_comp(node, arch)
            edge.add_dest_component(pe_id)

            path = self.set_edge_path(edge, arch)
            source_instruction = self.get_schedule_node(edge.source_id).get_instruction()

            edge.set_ready_cycle(start_cycle)
            edge.set_path_cycle(-1, start_cycle)
        else:
            while pe_id < 0:
                pe_id, min_cost = self.brute_force_pe(start_cycle, node, arch)
                start_cycle += 1
            start_cycle -= 1
            assert min_cost - start_cycle >= 0
            node.set_exec_cycle(min_cost)


            node.set_component_id(pe_id)
            node.set_cat_comp(arch.component_map[pe_id].category_id)

            if node.has_data_edge:
                arch = self.set_data_edge_comp(node, arch)
            edge.add_dest_component(pe_id)

            path = self.set_edge_path(edge, arch)
            source_instruction = self.get_schedule_node(edge.source_id).get_instruction()

            cost = cycle_cost(edge, source_instruction, pe_id, arch)
            cost = cost + parent_node.exec_cycle
            edge.set_ready_cycle(cost)
            edge.set_path_cycle(-1, cost)


        self.initialize_output_edges(node, arch)
        instr = self.create_instruction(arch, node)
        node.set_instruction(instr)
        return arch

    def schedule_sink_edges(self, arch: 'TablaTemplate') -> 'TablaTemplate':
        sink_node = self.get_schedule_node(1)
        if sink_node.op_name != "sink":
            raise RuntimeError(f"Sink node does not have the appropriate id: {sink_node.op_name}")
        total_edges = len(sink_node.in_edges)
        start_num = 0
        pbar = tqdm(sink_node.in_edges,
                    file=sys.stdout,
                    dynamic_ncols=True, total=total_edges,
                    desc="Scheduling Sink edges.",
                    disable=self.progress_bar)

        for edge_id in pbar:
            start_num += 1
            edge = self.get_schedule_edge(edge_id)
            if not edge.is_output_dtype:
                raise RuntimeError(f"Input edge {edge_id} to sink node does not have state datatype: {edge.dtype}")
            #
            # if self.is_training_algorithm:
            #     if edge.dtype != "state":
            # else:
            #     if edge.dtype != "output":
            #         raise RuntimeError(f"Input edge {edge_id} to sink node does not have output datatype: {edge.dtype}")

            if edge.sink_node < 0 and edge.dtype != "output":
                raise RuntimeError(f"Input edge {edge_id} {edge.dtype} to sink node does not have storage node set.")

            src_node = self.get_schedule_node(edge.source_id)
            src_node_instr = src_node.get_instruction()

            if src_node_instr is None:
                raise RuntimeError(f"No instruction set for node {src_node.node_id}")
            assert len(edge.sink_components) == len(set(edge.sink_components))
            if src_node.dtype == "output":
                assert len(edge.sink_components) == 0 and not src_node_instr.check_dest("NW")
                loc, index = arch.get_output_idx(src_node.exec_cycle, src_node.component_id, edge)

                arch.add_instruction_dest(src_node.exec_cycle, src_node_instr, edge, loc,
                                          src_node_instr.node_id, src_node.component_id, index=index)

            for i, sink_comp in enumerate(edge.sink_components):

                _ = self.set_edge_path(edge, arch)

                location = arch.get_data_location(edge.sub_paths[0][1], edge)

                if src_node_instr.check_dest(location) and location not in NAMESPACES:
                    comm_instr = create_comm_instruction(self, arch, edge, src_node_instr, src_node)
                    src_node.add_comm_instruction(comm_instr)
                else:
                    dst_loc, dst_index = arch.compute_data_source(src_node.exec_cycle, edge.sub_paths[0], edge)
                    if not src_node_instr.check_dest(location):
                        arch.add_instruction_dest(src_node.exec_cycle, src_node_instr, edge, dst_loc,
                                                  edge.sink_node, sink_comp, index=dst_index)

                    if len(edge.path) > 3:
                        dst_pe_index = edge.get_component_index(edge.sub_paths[0][2])
                        dst_cost = CYCLE_DELAYS[dst_loc]["PE"]
                        edge.set_path_cycle(dst_pe_index, edge.path_cycles[dst_pe_index - 1] + dst_cost)

                for d in src_node_instr.dests:
                    if edge.edge_id not in d.all_dests:
                        d.add_edge(edge.edge_id)
                edge.set_path_cycle(0, src_node.exec_cycle)

                if edge.num_pes > 2:

                    create_inter_pe_instr(self, arch, edge, src_node)

                if len(edge.path) == 3:
                    end_idx = len(edge.path) - 1
                    end_loc = edge.text_path[end_idx-1]
                    edge.set_path_cycle(end_idx,
                                        edge.path_cycles[end_idx-1] + CYCLE_DELAYS[end_loc]["PE"])


                if sink_comp != src_node.component_id:
                    dst_pe = arch.component_map[sink_comp]
                    ns_id = dst_pe.get_namespace("NW").component_id

                    edge.add_ns_path(ns_id)
                    src_pe = edge.sub_paths[-2][0]
                    comp_id = edge.sub_paths[-2][1]
                    dst_loc, dst_index = arch.compute_data_dest(edge.path_cycles[-4], comp_id,
                                                                src_pe, edge)

                    comm_instr = Instruction(src_node.node_id, "pass")
                    comm_instr.add_source(edge.edge_id, dst_loc, edge.data_id, src_pe, index=dst_index)

                    pe_index = len(edge.path) - 3
                    start_cycle = edge.path_cycles[-3]

                    while not dst_pe.is_idle(start_cycle):
                        start_cycle += 1
                        edge.add_path_overhead(pe_index, 1)

                    dst_loc, dst_index = arch.compute_data_source(start_cycle, edge.sub_paths[-1], edge)
                    comm_instr.add_dest(edge.edge_id, dst_loc, edge.data_id, sink_comp, index=dst_index)
                    edge.set_path_cycle(len(edge.path) - 1,
                                        edge.sub_path_cycles[-1][1] + CYCLE_DELAYS["NAMESPACE"]["PE"])

                    arch.add_pe_instruction(start_cycle, self, comm_instr, dst_pe.component_id, instruction_fn="schedule_sink")


        return arch
    # TODO: Add case for data node
    # TODO: assign the same id to multiple output edges
    # TODO: write graph traversal to validate the graph after generation
    # TODO: If single input, use ALU instead of namespace
    # TODO: Write to namespace if that is the target output
    # TODO: validate instructions along the way
    # TODO: Add data to bus in the appropriate cycle to verify bus contention issues
    # @profile
    def schedule_graph(self, arch: 'TablaTemplate', **optimizations):

        total_n = len(self._dfg_nodes)
        start = 0
        max_depth = -1
        pbar = tqdm(iterable=self._dfg_nodes,
                    total=total_n,
                    file=sys.stdout,
                    dynamic_ncols=True,
                    desc="Scheduling graph.",
                    disable=self.progress_bar)
        num_inputs = 0
        for node in pbar:
            pbar.set_description(f"Processing node {node.node_id}.")
            start += 1
            assert node.depth >= max_depth

            if max_depth < node.depth:
                max_depth = node.depth

            # Data node means no need to schedule
            if node.is_data_node():
                self.set_edge_names(node)
                if node.namespace_name == "ND":
                    num_inputs += 1
            elif node.is_source_sink():
            # Dont need to schedule source or sink nodes
                continue
            elif self.is_src_child(node):
                # if it is the child node of the source, it needs to be assigned a namespace
                arch = self.schedule_src_child(node, arch)
            elif len(node.in_edges) == 1:
                # Only 1 operand needs to be schedule for single input nodes
                arch = self.schedule_single_input(node, arch)

            else:
                # Standard case: schedule operation using both inputs
                arch = self.schedule_node(node, arch)
        arch = self.schedule_sink_edges(arch)
        print(f"\n\n----------------Num inputs: {num_inputs}--------\n\n")
        if self.optimize:
            arch = optimize_instructions(arch, **optimizations)

        return arch, num_inputs

    @memoize_method
    def get_widths(self):
        width_dict = defaultdict(int)
        for n in self._dfg_nodes:
            if n.is_data_node() or n.is_source_sink():
                continue
            width_dict[n.depth] += 1
        return width_dict

    @memoize_method
    def get_max_width(self):
        width_dict = self.get_widths()
        return max([width for _, width in width_dict.items()])

    def get_node_depth(self, node: ScheduleNode) -> int:
        depth = -1
        if node.node_id == 0:
            return depth + 1

        for parent_id in node.parents:
            parent_node = self.get_schedule_node(parent_id)
            if parent_node.depth > depth:
                depth = parent_node.depth
        return depth + 1

    def set_data_edge_comp(self, node: ScheduleNode, arch: 'TablaTemplate'):

        scheduled_edges = [self.get_schedule_edge(in_edge_id).is_scheduled() for in_edge_id in node.in_edges]

        if all(scheduled_edges):
            raise RuntimeError(f"All edges for node {node.node_id} are scheduled.")
        empty_pe_id = node.component_id

        if empty_pe_id < 0:
            raise RuntimeError(f"Unable to find empty PE for node {node.node_id}")

        for in_edge_id in node.in_edges:
            in_edge = self.get_schedule_edge(in_edge_id)
            if in_edge.is_src_edge:
                _ = arch.add_namespace_data(0, empty_pe_id, in_edge.namespace_name, in_edge)
                in_edge.add_source_component(empty_pe_id)
                in_edge.add_dest_component(empty_pe_id)
                in_edge.set_ready_cycle(0)

        for p_node_id in node.parents:
            p_node = self.get_schedule_node(p_node_id)
            if p_node.is_data_node():
                p_node.set_exec_cycle(0)
                pe_comp_id = empty_pe_id
                p_node.set_component_id(pe_comp_id)
                p_node.set_cat_comp(arch.component_map[pe_comp_id].category_id)

        return arch

    # TODO: Need to fix this to account for no PEs available for larger input data width
    # TODO: Make sure cost is correct
    # TODO: Traverse graph after generating instructioons and path
#

def create_nx_graph(graph_dict):
    import networkx as nx
    tabla_map = {i['id']: i for i in graph_dict}
    g = nx.DiGraph()

    for n in graph_dict:
        mparent = len(g.nodes) if len(n["parents"]) == 0 else (len(g.nodes) - min(n["parents"]))
        if len(n["children"]) == 0:
            g.add_node(n["id"], mparent=mparent)
        else:
            for c in n["children"]:
                g.add_edge(n["id"], c)
            g.nodes[n["id"]]["mparent"] = mparent
    return g

def find_graph_levels(g):
    levels = []
    while g.nodes():
        no_in_nodes = [n for (n, d) in g.in_degree(g.nodes()) if d == 0]
        levels.append(no_in_nodes)
        for n in no_in_nodes:
            g.remove_node(n)
    return levels

def topological_sort(tabla_graph, sort_type="test"):
    def test_topo(graph_dict, tabla_map):

        g = nx.DiGraph()

        for n in graph_dict:
            mparent = len(g.nodes) if len(n["parents"]) == 0 else (len(g.nodes) - min(n["parents"]))
            if len(n["children"]) == 0:
                g.add_node(n["id"], mparent=mparent)
            else:
                for c in n["children"]:
                    g.add_edge(n["id"], c)
                g.nodes[n["id"]]["mparent"] = mparent

        dist = {}  # stores [node, distance] pair
        try:
            cycles = nx.find_cycle(g)
            print(f"Cycles found: {cycles}")
        except:
            pass

        for node in nx.topological_sort(g):
            # pairs of dist,node for all incoming edges
            pairs = [(dist[v][0] + 1, v) for v in g.pred[node]]
            if pairs:
                dist[node] = max(pairs)
            else:
                dist[node] = (0, node)
        node, (length, _) = max(dist.items(), key=lambda x: x[1])

        depths = defaultdict(list)
        for k, v in dist.items():
            depths[v[0]].append(k)
        depths = {k: sorted(v) for k,v in depths.items()}
        shortest_path_lengths = nx.single_source_shortest_path_length(g, 0)
        temp = {i: tabla_map[i] for i in nx.topological_sort(g)}
        ret = []
        L = []

        def sort_func(x):
            if x == 1:
                return (0)
            else:
                k1 = min([dist[cd][0] for cd in temp[x]['children']])
                k2 = min([shortest_path_lengths[cd] for cd in temp[x]['children']])
                return (k1, k2)

        for d in sorted(depths.keys()):
            if d == 0:
                L.append(0)
                ret.append(temp[d])
                continue
            L_ = sorted(depths[d], key=sort_func)
            ret += list(map(lambda y: temp[y], L_))
            L += L_

        return ret, L

    def topo_sort(graph_dict):
        graph = {n["id"]: n["children"] for n in graph_dict}

        in_degree = {u: 0 for u in graph}  # determine in-degree
        for u in graph:  # of each node
            for v in graph[u]:
                in_degree[v] += 1

        Q = deque()  # collect nodes with zero in-degree
        for u in in_degree:
            if in_degree[u] == 0:
                Q.appendleft(u)

        L = []  # list for order of nodes

        while Q:
            u = Q.pop()  # choose node of zero in-degree
            L.append(u)  # and 'remove' it from graph
            for v in graph[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    Q.appendleft(v)

        if len(L) == len(graph):
            ret = [get_tabla_node(i, graph_dict) for i in L]
            return ret, L
        else:  # if there is a cycle,
            return []  # then return an empty list


    if not sort_type:
        return topo_sort(tabla_graph)
    elif sort_type == "custom":
        gmap = {i['id']: i for i in tabla_graph}
        return test_topo(tabla_graph, gmap)
    else:
        raise ValueError(f"Invalid sort type {sort_type}")

def get_tabla_node(nid, graph):
    for v in graph:
        if v["id"] == nid:
            return v
    raise KeyError
