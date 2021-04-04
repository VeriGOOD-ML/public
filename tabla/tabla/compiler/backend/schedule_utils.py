from typing import List, TYPE_CHECKING
if TYPE_CHECKING:
    from . import ScheduleEdge, ScheduleNode
    from . import TablaTemplate
    from . import Component
    from . import Instruction
from . import CYCLE_DELAYS
from .instruction import Instruction
from .tabla_utils import get_instr_cat_pe
from pytools import memoize_method
import bisect
import pickle
from collections import defaultdict


def get_cost_type(component: 'Component') -> str:
    if component.component_type in ['pe', 'pu', 'namespace']:
        return component.component_type.upper()
    else:
        return component.component_subtype

def compute_default_cost(cycle: int, path: List[str]) -> List[int]:
    cost_path = [0]
    src_pe_type = path[0]

    for idx, dst_pe_type in enumerate(path):
        if idx == 0:
            continue
        if path[idx - 1] == "PE":
            cost_path.append(cycle + CYCLE_DELAYS[src_pe_type][dst_pe_type] + 1)
        else:
            cost_path.append(cycle + CYCLE_DELAYS[src_pe_type][dst_pe_type])
        src_pe_type = dst_pe_type
    return cost_path

@memoize_method
def get_edge_dest(arch: 'TablaTemplate', edge: 'ScheduleEdge', dst_pe_id: int) -> str:

    src_pe = arch.component_map[edge.src_component]
    dst_pe = arch.component_map[dst_pe_id]
    # TODO: Need to determine if same pe means through namespace or ALU
    # TODO: Determine if destination is a PE or namespace.

    if edge.alu_edge:
        dst = "ALU"
    elif edge.src_component == dst_pe_id:
        dst = edge.namespace_name
    elif arch.get_pe_neighbor(edge.src_component) == dst_pe_id:
        dst = "PENB"
    elif src_pe.pu_id == dst_pe.pu_id:
        dst = "PEGB"
    elif arch.get_pu_neighbor(src_pe.pu_id) == dst_pe.pu_id:
        src_pu_head_id = arch.get_head_pe(src_pe.pu_id)
        if edge.src_component != src_pu_head_id:
            if arch.get_pe_neighbor(edge.src_component) == src_pu_head_id:
                dst = "PENB"
            else:
                dst = "PEGB"
        else:
            dst = "PUNB"
    else:
        src_pu_head_id = arch.get_head_pe(src_pe.pu_id)
        if edge.src_component != src_pu_head_id:
            if arch.get_pe_neighbor(edge.src_component) == src_pu_head_id:
                dst = "PENB"
            else:
                dst = "PEGB"
        else:
            dst = "PUGB"
    return dst

def determine_edge_path(arch: 'TablaTemplate', edge: 'ScheduleEdge') -> List[int]:
    path = []
    if edge.sink_node < 0 and (edge.src_component < 0 or edge.dst_component < 0):
        raise RuntimeError(f"Unable to determine path for edge {edge.edge_id} "
                           f"because not all source and destination component"
                           f"Ids have been set\n\tSource: {edge.src_component}"
                           f"\n\tDest: {edge.dst_component}")
    if edge.sink_node >= 0:
        dst_comp = edge.sink_components[len(edge.sink_paths)]
    else:
        dst_comp = edge.dst_component
    src_pe = arch.component_map[edge.src_component]
    dst_pe = arch.component_map[dst_comp]
    # TODO: Need to determine if same pe means through namespace or ALU
    # TODO: Determine if destination is a PE or namespace.
    if edge.alu_edge:
        path.append(dst_comp)
    elif edge.src_component == dst_comp:
        if edge.is_sink_edge:
            edge.set_ns_write()
        ns_id = src_pe.ns_id(edge.namespace_name)
        path.append(ns_id)
        if edge.is_sink_edge:
            edge.unset_ns_write()
    elif arch.get_pe_neighbor(edge.src_component) == dst_comp:
        path.append(src_pe.neighbor_bus)
    elif src_pe.pu_id == dst_pe.pu_id:
        path.append(src_pe.global_bus)
    elif arch.get_pu_neighbor(src_pe.pu_id) == dst_pe.pu_id:
        src_pu_head_id = arch.get_head_pe(src_pe.pu_id)
        dst_pu_head_id = arch.get_head_pe(dst_pe.pu_id)
        if edge.src_component != src_pu_head_id:
            if arch.get_pe_neighbor(edge.src_component) == src_pu_head_id:
                path.append(src_pe.neighbor_bus)
            else:
                path.append(src_pe.global_bus)

            path.append(src_pu_head_id)

        pu = arch.component_map[src_pe.pu_id]
        path.append(pu.neighbor_bus)
        path.append(dst_pu_head_id)
        if dst_comp != dst_pu_head_id:
            dst_pu_head = arch.component_map[dst_pu_head_id]
            if arch.get_pe_neighbor(dst_pu_head_id) == dst_comp:
                path.append(dst_pu_head.neighbor_bus)
            else:
                path.append(dst_pu_head.global_bus)

    else:
        src_pu_head_id = arch.get_head_pe(src_pe.pu_id)
        dst_pu_head_id = arch.get_head_pe(dst_pe.pu_id)
        if edge.src_component != src_pu_head_id:
            if arch.get_pe_neighbor(edge.src_component) == src_pu_head_id:
                path.append(src_pe.neighbor_bus)
            else:
                path.append(src_pe.global_bus)

            path.append(src_pu_head_id)

        pu = arch.component_map[src_pe.pu_id]
        path.append(pu.global_bus)
        path.append(dst_pu_head_id)
        if dst_comp != dst_pu_head_id:
            dst_pu_head = arch.component_map[dst_pu_head_id]
            if arch.get_pe_neighbor(dst_pu_head_id) == dst_comp:
                path.append(dst_pu_head.neighbor_bus)
            else:
                path.append(dst_pu_head.global_bus)
    if path[-1] != dst_comp or edge.alu_edge:
        path.append(dst_comp)

    return [edge.src_component] + path

def cycle_cost(edge: 'ScheduleEdge', instr: 'Instruction', dst_comp: int, arch: 'TablaTemplate', start_cycle=-1) -> int:

    if edge.src_component < 0:
        return -1
    edge_dest = "ALU" if edge.alu_edge else get_edge_dest(arch, edge, dst_comp)
    component_cost = 1 if edge.alu_edge else arch.pe_costs[edge.src_component][dst_comp]
    path = arch.pe_paths[edge.src_component][dst_comp]
    if instr and instr.check_dest(edge_dest):
        component_cost += 1
        start = max(instr.cycle_insert[0], start_cycle)
    elif instr:
        start = max(instr.cycle_insert[0], start_cycle)
    else:
        assert edge.is_src_edge
        start = 0
    if component_cost + start == float('inf'):
        raise RuntimeError(f"Instruction {instr.node_id} has infinite start cycle: {instr.cycle_insert[0]}")

    assert start >= 0
    return component_cost


def get_ordered_cycle(arch, instr, pe, edge1, edge2):
    src1_id = get_instr_cat_pe(instr.srcs[0], pe, arch.pes_per_pu, arch.num_pus)
    src2_id = get_instr_cat_pe(instr.srcs[1], pe, arch.pes_per_pu, arch.num_pus)
    dst_id = arch.component_map[pe.component_id].category_id

    k1 = (src1_id, dst_id)
    k2 = (src2_id, dst_id)

    wr_idx1 = arch.pe_wr_map[k1].index(instr.srcs[0].data_id)
    wr_idx2 = arch.pe_wr_map[k2].index(instr.srcs[1].data_id)

    if arch.wcycle_data_map[k1][wr_idx1] > arch.wcycle_data_map[k2][wr_idx2]:
        pos = 0


        if wr_idx1 - 1 >= 0 and wr_idx1 - 1 < len(arch.pe_r_map[k1]):
            lower_bound = arch.rcycle_data_map[k1][wr_idx1 - 1][0]
        else:
            lower_bound = edge1[1]

        if wr_idx1 + 1 >= len(arch.pe_r_map[k1]):
            upper_bound = max(pe.max_cycle + 2, lower_bound)
        else:
            upper_bound = arch.rcycle_data_map[k1][wr_idx1 + 1][0]

    else:
        pos = 1
        if wr_idx2 - 1 >= 0 and wr_idx2 - 1 < len(arch.pe_r_map[k1]):
            lower_bound = arch.rcycle_data_map[k2][wr_idx2 - 1][0]
        else:
            lower_bound = edge2[1]

        if wr_idx2 + 1 >= len(arch.pe_r_map[k2]):
            upper_bound = max(pe.max_cycle + 2, lower_bound)
        else:
            upper_bound = arch.rcycle_data_map[k2][wr_idx2 + 1][0]

    return lower_bound, upper_bound, pos

def get_inter_pe_cycle(arch, src_pe, dest_pe, start_cycle):
    src_id = arch.component_map[src_pe].category_id
    dst_id = arch.component_map[dest_pe].category_id
    key = (src_id, dst_id)
    if len(arch.pe_r_map[key]) > 0:
        cycle = max(arch.rcycle_data_map[key][-1][0], start_cycle)
    else:
        cycle = start_cycle

    return cycle



def get_cycle_bounds(arch, prev_pe, src_pe, dest_pe, edge_id, start_cycle):
    prev_id = arch.component_map[prev_pe].category_id
    src_id = arch.component_map[src_pe].category_id
    dst_id = arch.component_map[dest_pe].category_id
    pe = arch.component_map[dest_pe]
    key0 = (prev_id, src_id)
    key1 = (src_id, dst_id)

    wr_ids = [w.eid for w in arch.wcycle_edge_map[key0]]
    wr_idx = wr_ids.index(edge_id)

    if len(arch.pe_r_map[key1]) > 0:
        lower_bound = max(start_cycle, arch.wcycle_data_map[key1][-1][0])
    elif wr_idx - 1 >= 0 and wr_idx - 1 < len(arch.pe_r_map[key0]):
        lower_bound = max(arch.rcycle_data_map[key0][wr_idx - 1][0], start_cycle)
    else:
        lower_bound = start_cycle

    return lower_bound

def get_cycle_bounds_old(arch, prev_pe, src_pe, dest_pe, edge_id, start_cycle):
    src_id = arch.component_map[src_pe].category_id
    dst_id = arch.component_map[dest_pe].category_id
    key1 = (src_id, dst_id)
    wcycles = list(map(arch.cycle_from_map, arch.wcycle_edge_map[key1]))
    wr_idx = bisect.bisect_right(wcycles, start_cycle) - 1

    if wr_idx >= 0:
        lower_bound = max(start_cycle, wcycles[wr_idx])
    else:
        lower_bound = start_cycle

    return lower_bound


def ordered_cycle_val(arch, start_cycle, send_pe, rec_pe, tgt_pe, data_id):
    read_key = (arch.component_map[send_pe].category_id, arch.component_map[rec_pe].category_id)
    write_key = (read_key[1], arch.component_map[tgt_pe].category_id)


    if len(arch.rcycle_data_map[write_key]) > 0:
        write_cycle = max(start_cycle, arch.rcycle_data_map[write_key][-1][0])
    else:
        write_cycle = start_cycle

    if data_id not in arch.pe_wr_map[read_key]:
        return start_cycle

    idx = arch.pe_wr_map[read_key].index(data_id)
    if idx - 1 >= 0:
        lower_read_idx = arch.pe_r_map[read_key].index(arch.pe_wr_map[read_key][idx - 1])
        lower_read_cycle = arch.rcycle_data_map[read_key][lower_read_idx][0]
    else:
        lower_read_cycle = start_cycle

    if idx + 1 < len(arch.wcycle_data_map[read_key]):
        upper_read_idx = arch.pe_r_map[read_key].index(arch.pe_wr_map[read_key][idx + 1])
        upper_read_cycle = arch.rcycle_data_map[read_key][upper_read_idx][0]
    else:
        upper_read_cycle = max(write_cycle, lower_read_cycle)

    if idx + 1 == len(arch.wcycle_data_map[read_key]):
        ret_cycle = max(write_cycle, upper_read_cycle)
    elif upper_read_cycle > write_cycle:
        ret_cycle = max(upper_read_cycle, start_cycle)
    else:
        ret_cycle = max(lower_read_cycle, start_cycle)

    return ret_cycle



def get_read_cycle(arch, start_cycle, src_id, dst_id, data_id):
    read_key = (arch.component_map[src_id].category_id, arch.component_map[dst_id].category_id)
    idx = arch.pe_wr_map[read_key].index(data_id)
    if idx + 1 == len(arch.pe_wr_map[read_key]) and len(arch.pe_r_map[read_key]) > 0:
        return max(arch.rcycle_data_map[read_key][-1][0], start_cycle)
    else:
        return start_cycle

def find_exec_cycle(arch: 'TablaTemplate', src_node: 'ScheduleNode', edge: 'ScheduleEdge', path_index: int) -> int:

    target_pe = arch.component_map[edge.path[path_index]]

    if target_pe.__class__.__name__ != 'PE':
        raise ValueError(f"Cycle for edge {edge.edge_id} with path index {path_index}\n"
                         f"cannot be determined because it is not a PE: {edge.text_path[path_index]}")
    if path_index > 0:
        loc = edge.text_path[path_index - 1]
        constraint_cycle = edge.path_cycles[path_index - 1] + CYCLE_DELAYS[loc]["PE"]
    else:
        constraint_cycle = src_node.exec_cycle + 1

    #
    if edge.path[path_index-2] != edge.path[path_index] and len(edge.path) == 3:

        constraint_cycle = get_read_cycle(arch, constraint_cycle,
                                     edge.path[path_index-2],
                                     edge.path[path_index],
                                     src_node.node_id)

    while not target_pe.is_idle(constraint_cycle):
        constraint_cycle += 1

    return constraint_cycle

def create_inter_pe_instr(sched, arch: 'TablaTemplate', edge: 'ScheduleEdge', src_node):

    start_len = len(edge.sub_paths)
    idx = 0
    prev_instr = 1
    while idx < len(edge.sub_paths):
        sub_path = edge.sub_paths[idx]
        if idx == 0 or sub_path[0] == edge.src_component or start_len < len(edge.sub_paths):
            incr = len(edge.sub_paths) - start_len if start_len < len(edge.sub_paths) else 1
            idx += incr
            start_len = len(edge.sub_paths)
            continue

        pe_id = sub_path[0]
        dst_loc, dst_index = arch.compute_data_dest(edge.sub_path_cycles[idx][0], edge.sub_paths[idx - 1][1], edge.sub_paths[idx - 1][0], edge)
        pe = arch.component_map[pe_id]

        start_cycle = get_cycle_bounds(arch, edge.sub_paths[idx - 1][0], pe_id, sub_path[2], edge.edge_id, edge.sub_path_cycles[idx][0])
        # start_cycle = edge.sub_path_cycles[idx][0]
        # oh = start_cycle - edge.sub_path_cycles[idx][0]

        pe_index = edge.get_component_index(sub_path[1]) - 1
        edge.add_path_overhead(pe_index, start_cycle - edge.sub_path_cycles[idx][0])

        while not pe.is_idle(start_cycle):
            start_cycle += 1
            edge.add_path_overhead(pe_index, 1)

        comm_instr = Instruction(src_node.node_id, "pass")
        comm_instr.add_source(edge.edge_id, dst_loc, edge.data_id, sub_path[0], index=dst_index)


        src_loc, src_index = arch.compute_data_source(start_cycle, sub_path, edge)
        comm_instr.add_dest(edge.edge_id, src_loc, edge.data_id, sub_path[2], index=src_index)

        prev_instr = comm_instr

        edge.set_path_cycle(edge.get_component_index(sub_path[2]), edge.sub_path_cycles[idx][1] + CYCLE_DELAYS[src_loc]["PE"])

        new_cycle = arch.add_pe_instruction(start_cycle, sched, comm_instr, pe.component_id, instruction_fn="create_inter_pe")
        comm_instr.set_component_id(pe.component_id)
        idx += 1

def create_comm_instruction(sched, arch: 'TablaTemplate', edge: 'ScheduleEdge', instr: Instruction, src_node) -> Instruction:

    src_pe = arch.component_map[edge.src_component]
    namespace = src_pe.get_namespace("NI")
    if not instr.check_dest("NI"):
        delay = src_node.exec_cycle + CYCLE_DELAYS["PE"]["NAMESPACE"] + 1
        arch.add_namespace_data(delay, src_node.component_id, "NI", edge)
        idx = namespace.find_data_index(edge.data_id, cycle=delay)
        arch.add_instruction_dest(delay, instr, edge, "NI", edge.data_id, src_node.component_id, index=idx)

    if edge.edge_id not in instr.get_dest("NI").all_dests:
        instr.get_dest("NI").add_edge(edge.edge_id)
    # Find the first available cycle for the PE
    start_cycle = src_node.exec_cycle + 1 + CYCLE_DELAYS["PE"]["NAMESPACE"]
    while not src_pe.is_idle(start_cycle):
        start_cycle += 1


    # Add the overhead resulting from the additional communication instruction to the rest of the path
    overhead = (start_cycle + CYCLE_DELAYS["PE"]["NAMESPACE"] + 1) - src_node.exec_cycle
    edge.add_path_overhead(edge.get_component_index(edge.sub_paths[0][1]), overhead)

    # Get the target PE ID
    pe_id = edge.sub_paths[0][2]
    loc, index = arch.compute_data_source(src_node.exec_cycle + overhead, edge.sub_paths[0], edge)
    comm_instr = Instruction(edge.source_id, "pass")

    edge.update_paths(1, "NAMESPACE", namespace.component_id, src_node.exec_cycle + 1 + CYCLE_DELAYS["PE"]["NAMESPACE"])
    edge.update_paths(2, "PE", src_pe.component_id, start_cycle)

    loc = arch.get_data_location(edge.sub_paths[1][1], edge)
    delay = find_exec_cycle(arch, src_node, edge, 4)
    edge.set_path_cycle(4, delay)

    # dest_index = namespace.find_data_index(edge.data_id, cycle=src_node.exec_cycle + CYCLE_DELAYS["PE"]["NAMESPACE"] + 1)
    dest_index = namespace.find_data_index(edge.data_id)

    comm_instr.add_source(edge.edge_id, "NI", edge.data_id, src_pe.category_id, index=dest_index)
    comm_instr.add_dest(edge.edge_id, loc, edge.data_id, arch.component_map[pe_id].category_id, index=index)
    arch.add_pe_instruction(start_cycle, sched, comm_instr, src_pe.component_id, instruction_fn="create_comm")

    edge.set_ready_cycle(edge.path_cycles[-1])

    return comm_instr

def create_temp_instr(sched, arch: 'TablaTemplate', dest_pe, edge_info, instr: 'Instruction', max_cycle):


    lower_bound, upper_bound, pos = get_ordered_cycle(arch, instr, dest_pe, edge_info[0], edge_info[1])
    other = 1 if pos == 0 else 0
    src = instr.srcs[pos]
    comm_edge = edge_info[pos][0]
    start_cycle = edge_info[pos][1]

    cycle = start_cycle
    while not dest_pe.is_idle(cycle):
        cycle += 1

    ###### test ##########
    other_src_pe = get_instr_cat_pe(instr.srcs[other], dest_pe, arch.pes_per_pu, arch.num_pus)

    key = (other_src_pe, dest_pe.category_id)

    wr_index = [i for i, wi in enumerate(arch.wcycle_edge_map[key]) if wi.eid == edge_info[other][0].edge_id][0]

    _ = arch.wcycle_data_map[key].pop(wr_index)
    wr_edge_info = arch.wcycle_edge_map[key].pop(wr_index)
    t = arch.pe_wr_map[key].pop(wr_index)

    ##################




    src_info = (comm_edge.edge_id, src.location, src.data_id, src.comp_id, src.index)
    comm_instr = Instruction(comm_edge.source_id, "pass")
    comm_instr.add_source(*src_info)
    delay = cycle + CYCLE_DELAYS["PE"]["NAMESPACE"] + 1
    namespace = dest_pe.get_namespace("NI")
    arch.add_namespace_data(delay, dest_pe.component_id, "NI", comm_edge)
    overhead = delay - start_cycle

    comm_edge.add_path_overhead(comm_edge.get_component_index(comm_edge.sub_paths[0][1]), overhead)

    comm_edge.update_paths(1, "NAMESPACE", dest_pe.get_namespace("NI").component_id,
                      start_cycle + CYCLE_DELAYS["PE"]["NAMESPACE"] + 1)
    comm_edge.update_paths(2, "PE", dest_pe.component_id, cycle + CYCLE_DELAYS["PE"]["NAMESPACE"] + 1)
    comm_edge.set_ready_cycle(comm_edge.path_cycles[-1])

    idx = dest_pe.get_namespace("NI").find_data_index(comm_edge.data_id, cycle=delay)
    # comm_instr.add_dest(comm_edge.edge_id, "NI", comm_edge.data_id, dest_pe.component_id, index=idx)
    comm_instr.set_component_id(dest_pe.component_id)

    arch.replace_instruction_source(instr, pos, comm_edge.edge_id, "NI", comm_edge.data_id, dest_pe.component_id, idx)

    arch.add_instruction_dest(cycle, comm_instr, comm_edge, "NI", comm_edge.data_id, dest_pe.component_id,
                              index=idx)
    comm_cycle = arch.add_pe_instruction(cycle, sched, comm_instr, dest_pe.component_id, check_maps=False, instruction_fn="create_temp_instr")


    new_exec_cycle = max(comm_edge.ready_cycle, comm_cycle)
    # new_exec_cycle = comm_edge.ready_cycle

    arch.update_write_map(key, wr_edge_info.cycle, instr.srcs[other].data_id, edge_info[other][0], index=-1)


    if new_exec_cycle > max_cycle:
        comm_edge.set_ready_cycle(new_exec_cycle)
        while not dest_pe.is_idle(new_exec_cycle):
            new_exec_cycle += 1

        return new_exec_cycle
    elif max_cycle == comm_cycle:
        while not dest_pe.is_idle(max_cycle):
            max_cycle += 1
        return max_cycle
    else:
        return max_cycle




