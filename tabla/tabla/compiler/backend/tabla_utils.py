from . import CYCLE_DELAYS
from typing import List
from collections import namedtuple

ReadWriteInfo = namedtuple('ReadWriteInfo', ['start', 'finish', 'meta', 'dataid', 'eid'])
EdgeReadWriteInfo = namedtuple('EdgeReadWriteInfo', ['start', 'finish', 'meta', 'dataid', 'index'])

def get_dst_penb(pe, index, pes_per_pu, pus):
    dst_pe = pe.category_id - (pes_per_pu - 1) if (pe.category_id + 1) % pes_per_pu == 0 else pe.category_id + 1
    if dst_pe > pus*pes_per_pu:
        raise RuntimeError(f"Invalid PE id: {dst_pe}")
    return dst_pe

def get_src_penb(pe, index, pes_per_pu, pus):
    src_pe = pe.category_id + (pes_per_pu - 1) if (pe.is_head_pe) else pe.category_id - 1
    if src_pe > pus*pes_per_pu:
        raise RuntimeError(f"Invalid PE id: {src_pe}")
    return src_pe

def get_dst_pegb(pe, index, pes_per_pu, pus):

    dst_pe = (pe.category_id // pes_per_pu) * pes_per_pu + index

    if dst_pe > pus*pes_per_pu:
        raise RuntimeError(f"Invalid PE id: {dst_pe}")
    return dst_pe

def get_src_pegb(pe, index, pes_per_pu, pus):
    src_pe = (pe.category_id // pes_per_pu) * pes_per_pu + index
    if src_pe > pus*pes_per_pu:
        raise RuntimeError(f"Invalid PE id: {src_pe}")
    return src_pe

def get_dst_punb(pe, index, pes_per_pu, pus):
    curr_pu = (pe.category_id // pes_per_pu)
    dst_pe = 0 if curr_pu == (pus - 1) else (curr_pu + 1) * pes_per_pu
    if dst_pe > pus*pes_per_pu:
        raise RuntimeError(f"Invalid PE id: {dst_pe}")
    return dst_pe

def get_src_punb(pe, index, pes_per_pu, pus):
    curr_pu = (pe.category_id // pes_per_pu)
    src_pe = (pes_per_pu * (pus - 1)) if curr_pu == 0 else (curr_pu - 1) * pes_per_pu
    if src_pe > pus*pes_per_pu:
        raise RuntimeError(f"Invalid PE id: {src_pe}")
    return src_pe

def get_dst_pugb(pe, index, pes_per_pu, pus):
    dst_pe = pes_per_pu * index
    if dst_pe > pus*pes_per_pu:
        raise RuntimeError(f"Invalid PE id: {dst_pe}")
    return dst_pe

def get_src_pugb(pe, index, pes_per_pu, pus):
    src_pe = pes_per_pu * index
    if src_pe > pus*pes_per_pu:
        raise RuntimeError(f"Invalid PE id: {src_pe}")
    return src_pe

def get_dst_ns(pe, index, pes_per_pu, pus):
    return pe.category_id

def get_src_ns(pe, index, pes_per_pu, pus):
    return pe.category_id

def get_instr_cat_pe(src, pe, pes_per_pu, pus, loc_idx=(None, None)):

    if loc_idx != (None, None):
        location = loc_idx[0]
        index = loc_idx[1]
    else:
        location = src.location
        index = src.index

    src_pe = SRC_PE_ID_FN[location](pe, index, pes_per_pu, pus)

    # if location == "PENB":
    #     # src_pe = pe.category_id + 7 if (pe.is_head_pe) else pe.category_id - 1
    #     src_pe = pe.category_id + (pes_per_pu - 1) if (pe.is_head_pe) else pe.category_id - 1
    # elif location == "PEGB":
    #     # src_pe = (pe.category_id // 8) * 8 + index
    #     src_pe = (pe.category_id // pus) * pus + index
    #
    # elif location == "PUNB":
    #     # curr_pu = (pe.category_id // 8)
    #     # src_pe = 56 if curr_pu == 0 else (curr_pu - 1) * 8
    #     curr_pu = (pe.category_id // pus)
    #     src_pe = (pes_per_pu*(pus - 1)) if curr_pu == 0 else (curr_pu - 1) * pes_per_pu
    # elif location == "PUGB":
    #     src_pe = pes_per_pu * index
    # else:
    #     src_pe = pe.category_id

    return src_pe

def find_dest_loc(source_pe, dest_pe, instr):
    assert source_pe != dest_pe

    if source_pe.category_id + 1 == dest_pe.category_id:
        assert instr.check_dest("PENB")
        return "PENB"
    elif source_pe.component_subtype == dest_pe.component_subtype:
        assert instr.check_dest("PEGB")
        return "PEGB"
    elif int(source_pe.component_subtype) + 1 == int(dest_pe.component_subtype):
        assert source_pe.is_head_pe and dest_pe.is_head_pe
        assert instr.check_dest("PUNB")
        return "PUNB"
    else:
        assert source_pe.is_head_pe and dest_pe.is_head_pe
        assert instr.check_dest("PUGB")
        return "PUGB"

def compute_cost(path: List[str]) -> int:
    cost = 0
    src_pe_type = path[0]

    for dst_pe_type in path[1:]:
        if src_pe_type == "PE":
            cost += CYCLE_DELAYS[src_pe_type][dst_pe_type] + 1
        else:
            cost += CYCLE_DELAYS[src_pe_type][dst_pe_type]
        src_pe_type = dst_pe_type

    return cost

def update_temp_edge_paths(edge, start_cycle, exec_cycle, original_cycle, pe):

    # path_index = list(filter(lambda x: edge.path_cycles[x] == original_cycle and edge.path[x] == pe.component_id, range(len(edge.path))))[0]
    path_index = edge.path.index(pe.component_id)

    overhead = exec_cycle - edge.path_cycles[path_index]
    edge.path_cycles[path_index] = start_cycle

    edge.add_path_overhead(path_index + 1, overhead)

    edge.update_paths(path_index + 1, "NAMESPACE", pe.get_namespace("NI").component_id,
                           start_cycle + CYCLE_DELAYS["PE"]["NAMESPACE"] + 1)
    edge.update_paths(path_index + 2, "PE", pe.component_id, exec_cycle)

    edge.set_ready_cycle(edge.path_cycles[-1])

def get_map_indices(src_id, mapped):
    result = []
    offset = -1
    while True:
        try:
            offset = mapped.index(src_id, offset + 1)
        except ValueError:
            return result
        result.append(offset)

def map_equality(key, src_id, r_map, w_map):

    if key[0] == key[1]:
        return False

    if len(r_map) == len(w_map):
        r_indices = get_map_indices(src_id, r_map)
        w_indices = get_map_indices(src_id, w_map)
        return r_indices != w_indices
    else:
        return False


def map_rwr_equality(key, edge, r_map, w_map):
    if key[0] == key[1]:
        return False

    if len(r_map) == len(w_map):
        ridx = [i for i, ri in enumerate(r_map) if ri.eid == edge.edge_id][0]
        widx = [i for i, wi in enumerate(w_map) if wi.eid == edge.edge_id][0]
        return ridx != widx
    else:
        return False


DEST_PE_ID_FN = {
    'PEGB': get_dst_pegb,
    'PENB': get_dst_penb,
    'PUGB': get_dst_pugb,
    'PUNB': get_dst_punb,
    'NI': get_dst_ns,
    'NS': get_dst_ns,
    'NW': get_dst_ns,
    'ND': get_dst_ns,
    'NM': get_dst_ns,
    'ALU': get_dst_ns,
}

SRC_PE_ID_FN = {
    'PEGB': get_src_pegb,
    'PENB': get_src_penb,
    'PUGB': get_src_pugb,
    'PUNB': get_src_punb,
    'NI': get_src_ns,
    'NS': get_src_ns,
    'NW': get_src_ns,
    'ND': get_src_ns,
    'NM': get_src_ns,
    'ALU': get_src_ns,
}