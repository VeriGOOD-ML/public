from . import PE, BUS_NAMES, NAMESPACES, CYCLE_DELAYS
from .tabla_utils import SRC_PE_ID_FN, DEST_PE_ID_FN
from .tabla_utils import ReadWriteInfo, EdgeReadWriteInfo
from collections import namedtuple, defaultdict
import bisect
import sys
import tqdm
import pickle

NamespaceRW = namedtuple('NamespaceRW', ['start', 'finish', 'did', 'eid'])

def reuse_ns(avail_indices, instr, instr_info, translation_map, instr_num, pe=None):

    if instr.check_dest("NI"):
        ni_dest = instr.get_dest("NI")
        unchanged = True
        if ni_dest.index in translation_map and instr_info[ni_dest.index].finish > instr_num:
            ni_dest.index = translation_map[ni_dest.index]
        else:
            for index in list(sorted(avail_indices.keys())):
                if index == ni_dest.index:
                    break
                if avail_indices[index] < instr_num:
                    translation_map[ni_dest.index] = index
                    avail_indices[index] = instr_info[ni_dest.index].finish
                    avail_indices[ni_dest.index] = -1
                    ni_dest.index = index
                    unchanged = False
                    break

            if unchanged:
                avail_indices[ni_dest.index] = instr_info[ni_dest.index].finish
                translation_map[ni_dest.index] = ni_dest.index

    src0 = instr.srcs[0]
    if src0.location == "NI" and src0.index in translation_map:

        if instr_num == instr_info[src0.index].finish:
            new_idx = translation_map.pop(src0.index)
        else:
            new_idx = translation_map[src0.index]
        src0.index = new_idx



    if len(instr.srcs) > 1:
        src1 = instr.srcs[1]

        if src1.location == "NI" and src1.index in translation_map:
            if instr_num == instr_info[src1.index].finish:
                new_idx = translation_map.pop(src1.index)
            else:
                new_idx = translation_map[src1.index]
            src1.index = new_idx

    return translation_map, avail_indices, instr_info

def reuse_ni_optimization(arch, debug=True):
    instr_info = arch.ni_read_writes()
    pes = [pe for _, pe in arch.category_component_dict["pe"].items() if isinstance(pe, PE)]
    pbar = tqdm.tqdm(total=arch.total_instructions, file=sys.stdout, dynamic_ncols=True, desc="Reusing NI Writes to ALU", disable=not debug)

    for pe_id, pe in enumerate(pes):
        pe_ni = pe.get_namespace("NI")
        pe_instr = pe.all_instructions()
        translation_map = {}
        ni_info = instr_info[pe.category_id]
        avail_indices = {idx: i.finish for idx, i in ni_info.items()}

        for instr_num, instr in enumerate(pe_instr):
            pbar.update(1)
            translation_map, avail_indices, ni_info = reuse_ns(avail_indices, instr, ni_info, translation_map, instr_num, pe)
        working_indices = set([i.get_dest("NI").index for i in pe_instr if i.check_dest("NI")])
        original_indices = [i for i in range(pe_ni.item_count())]
        if set(original_indices) != working_indices:
            storage = pe_ni.get_cycle_storage()
            new_storage = pickle.loads(pickle.dumps(storage))
            state_name = pe_ni.get_state_name(0)

            for i in original_indices:
                if i not in working_indices:
                    new_storage[i].invalidate()
                    pe_ni.decr_items()
            _ = pe_ni.update_cycle_state(0, state_name, {"storage": new_storage})


    return arch


def reorder_instr_index(avail_indices, translation_map, instr, pe):
    if instr.check_dest("NI"):
        ni_dest = instr.get_dest("NI")
        if ni_dest.index in translation_map:
            ni_dest.index = translation_map[ni_dest.index]
        elif len(avail_indices) > 0:
            new_idx = avail_indices.pop(0)
            translation_map[ni_dest.index] = new_idx
            ni_dest.index = new_idx

    src0 = instr.srcs[0]
    if src0.location == "NI" and src0.index in translation_map:
        src0.index = translation_map[src0.index]

    if len(instr.srcs) > 1:
        src1 = instr.srcs[1]
        if src1.location == "NI" and src1.index in translation_map:
            src1.index = translation_map[src1.index]

    return translation_map, avail_indices

def reorder_ns_indices(arch):
    pes = [pe for _, pe in arch.category_component_dict["pe"].items() if isinstance(pe, PE)]

    for pe_id, pe in enumerate(pes):
        pe_ni = pe.get_namespace("NI")
        pe_instr = pe.all_instructions()

        working_indices = [i for i in range(pe_ni.item_count())]
        translation_map = {}
        for instr_num, instr in enumerate(pe_instr):
            if instr.check_dest("NI"):
                ni_dest = instr.get_dest("NI")
                if ni_dest.index in translation_map:
                    ni_dest.index = translation_map[ni_dest.index]
                elif len(working_indices) > 0:
                    new_idx = working_indices.pop(0)
                    translation_map[ni_dest.index] = new_idx
                    ni_dest.index = new_idx
                else:
                    raise RuntimeError

            if instr.srcs[0].location == "NI":
                src0 = instr.srcs[0]
                src0.index = translation_map[src0.index]

            if len(instr.srcs) > 1 and instr.srcs[1].location == "NI":
                src1 = instr.srcs[1]
                src1.index = translation_map[src1.index]

        working_indices = list(sorted(list(set([i.get_dest("NI").index for i in pe_instr if i.check_dest("NI")]))))
        original_indices = [i for i in range(pe_ni.item_count())]
        if original_indices != working_indices:
            storage = pe_ni.get_cycle_storage()
            new_storage = pickle.loads(pickle.dumps(storage))
            state_name = pe_ni.get_state_name(0)
            for i in original_indices:
                if i not in working_indices:
                    new_storage[i].invalidate()
                    pe_ni.decr_items()
            _ = pe_ni.update_cycle_state(0, state_name, {"storage": new_storage})

    return arch

def insert_pass_instr(arch):
    pes = [pe for _, pe in arch.category_component_dict["pe"].items() if isinstance(pe, PE)]

    for pe_id, pe in enumerate(pes):
        pe_instr, cycle_list = pe.all_instructions_with_cycles()
        if len(pe_instr) == 0:
            continue

        pe_ni = pe.get_namespace("NI")
        prev_instr = pe_instr[0]

        for instr_num, instr in enumerate(pe_instr):
            if len(instr.srcs) == 1 or instr_num == 0:
                prev_instr = instr
                continue

            if instr.srcs[0].location == "NI" and instr.srcs[1].location == "NI":

                    idx = 0
                    add_pass = True
                    for d in prev_instr.dests:
                        if d.location == "NI":
                            if d.index == instr.srcs[0].index:
                                add_pass = False
                                idx = 0
                                break
                            if d.index == instr.srcs[1].index:
                                add_pass = False
                                idx = 1
                                break

                    if add_pass:
                        pass_instr = pe.create_temp_pass_instr(instr.srcs[idx])
                        pe.insert_instruction(cycle_list[instr_num], pass_instr)
                    arch.replace_instruction_source(instr, idx, instr.srcs[idx].source_id, "ALU",
                                                    instr.srcs[idx].data_id, instr.srcs[idx].comp_id, optimizing=True)

            prev_instr = instr

    return arch

def get_avail_idx(info_dict):
    indices = []

    for k,v in info_dict.items():
        if v.finish < 0:
            indices.append(k)

    return (sorted(indices))

def ni_uses(instructions):
    uses = defaultdict(int)
    for instr in instructions:
        if instr.srcs[0].location == "NI":
            uses[instr.srcs[0].index] += 1
        if len(instr.srcs) > 1 and instr.srcs[1].location == "NI":
            uses[instr.srcs[1].index] += 1
    return uses

def reduce_indices():
    pass

def unused_ni_noops(arch, debug=True):
    instr_info = arch.ni_read_writes()
    pes = [pe for _, pe in arch.category_component_dict["pe"].items() if isinstance(pe, PE)]
    pbar = tqdm.tqdm(total=arch.total_instructions, file=sys.stdout, dynamic_ncols=True, desc="Removing unused NI Writes", disable=not debug)
    for pe_id, pe in enumerate(pes):
        pe_ni = pe.get_namespace("NI")
        pe_instr = pe.all_instructions()
        ni_info = instr_info[pe.category_id]
        uses = ni_uses(pe_instr)

        for instr_num, instr in enumerate(pe_instr):
            pbar.update(1)
            if instr.check_dest("NI"):
                ni_dest = instr.get_dest("NI")
                if ni_info[ni_dest.index].finish < 0 and uses[ni_dest.index] == 0:
                    instr.remove_dest("NI")
    return arch


def check_self_pass(instr, next_instr):
    if instr.op_name != "pass":
        return False
    elif instr.srcs[0].location != "NI":
        return False
    elif instr.check_dest("NI") and len(instr.dests) == 1 and \
            all([s.location != "ALU" for s in next_instr.srcs]):
        assert instr.srcs[0].index == instr.get_dest("NI").index
        return True
    else:
        return False

def optimize_pe_instructions(pe, arch):

    all_instr = pe.all_instruction_map()
    prev_ns = []

    for cycle, instr in all_instr.items():
        add_alu_source(instr, prev_ns, arch)
        prev_ns = [str(d) for d in instr.dests if d.location in NAMESPACES]

def add_alu_source(instr, prev_ns, arch):
    for idx, src in enumerate(instr.srcs):
        if str(instr.srcs[idx]) in prev_ns:
            arch.replace_instruction_source(instr, idx, instr.srcs[idx].source_id, "ALU", instr.srcs[idx].data_id, instr.srcs[idx].comp_id, optimizing=True)

    return instr

def optimize_instructions(arch, reorder_instr=True, apply_reuse=True, unused_ni_opt=True,
                          use_validate_ni_opt=True, apply_noop_unused_ni=True, debug=True):
    pes = [pe for _, pe in arch.category_component_dict["pe"].items() if isinstance(pe, PE)]
    generate_sorted_instr(arch)

    for pe in pes:
        optimize_pe_instructions(pe, arch)

    if unused_ni_opt:
        arch = unused_ni_noops(arch, debug=debug)
        arch = reorder_ns_indices(arch)

    if apply_reuse:
        arch = reuse_ni_optimization(arch, debug=debug)
        arch = reorder_ns_indices(arch)

    if use_validate_ni_opt:
        validate_ni_opt(arch)
    arch = insert_pass_instr(arch)

    return arch

def validate_ni_opt(arch):
    pes = [pe for _, pe in arch.category_component_dict["pe"].items() if isinstance(pe, PE)]
    for pe_id, pe in enumerate(pes):
        pe_ni = pe.get_namespace("NI")
        pe_instr = pe.all_instructions()
        index_vals = {i: (-1,-1) for i in range(pe_ni.item_count())}
        instr_vals = {i: (-1,-1) for i in range(pe_ni.item_count())}

        for instr_num, instr in enumerate(pe_instr):
            if instr.srcs[0].location == "NI":
                src0 = instr.srcs[0]
                if src0.data_id != index_vals[src0.index]:
                    print(f"PE{pe.category_id}\n:"
                          f"Incorrect data stored in namespace for data id {src0.data_id}:\n"
                          f"Instruction {instr_num}: {instr}\n"
                          f"Actual data id: {index_vals[src0.index]}\n"
                          f"Prev Instruction {instr_vals[src0.index][1]}: {instr_vals[src0.index][0]}\n")

            if len(instr.srcs) > 1 and instr.srcs[1].location == "NI":
                src1 = instr.srcs[1]
                if src1.data_id != index_vals[src1.index]:
                    print(f"PE{pe.category_id}\n:"
                          f"Incorrect data stored in namespace for data id {src1.data_id}:\n"
                          f"Instruction {instr_num}: {instr}\n"
                          f"Actual data id: {index_vals[src1.index]}\n"
                          f"Prev Instruction {instr_vals[src1.index][1]}: {instr_vals[src1.index][0]}\n")

            if instr.check_dest("NI"):
                ni_dest = instr.get_dest("NI")
                index_vals[ni_dest.index] = ni_dest.data_id
                instr_vals[ni_dest.index] = (instr, instr_num)


    return arch


def generate_sorted_instr(arch):
    for _, pe in arch.category_component_dict["pe"].items():
        if isinstance(pe, PE):
            all_instr = pe.all_instruction_map()
            instrs = list(all_instr.values())
            cycles = list(all_instr.keys())
            removed_cycles_idx = []
            removed_cycles = []

            for idx, instr in enumerate(instrs):
                cycle = cycles[idx]
                next_instr = instr if idx >= len(instrs) - 1 else instrs[idx + 1]
                if check_self_pass(instr, next_instr):
                    pe.remove_instruction(cycle)
                    removed_cycles_idx.append(idx)
                    removed_cycles.append(cycle)

            for r in removed_cycles:
                cycles.remove(r)
            pe.sorted_instr = pe.all_instructions()
            pe.sorted_cycles = cycles

def get_annotated_instr(pe, sched):
    instr_list = []
    for instr_num, instr in enumerate(pe.all_instructions()):

        src_ids = [str(i.data_id) for i in instr.srcs]
        dest_id = str(instr.node_id)

        src_annotation = str(instr.cycle_insert) + ", ".join(
            [f"{src_ids[i]} " for i in range(len(src_ids))])
        data_annotation = f"{src_annotation} --> {dest_id}\t"

        src_pes = ", ".join([str(sp) for sp in get_source_pe(instr, pe, sched)])
        dest_pes = ", ".join([str(dd) for dd in get_dest_pe(instr, pe, sched)])
        pe_annotation = f"{src_pes} --> {dest_pes}\t"
        instr_annotation = pe_annotation + data_annotation + f"{str(instr)}"
        instr_list.append(f"{instr_num}: {instr_annotation}")
    return "\n".join(instr_list)

def get_source_pe(instr, pe, sched):
    src_pes = []
    for src in instr.srcs:
        if src.location in ["PENB", "PEGB", "PUNB", "PUGB"]:
            src_pe = SRC_PE_ID_FN[src.location](pe, src.index, sched.pes_per_pu, sched.num_pus)

        #     src_pe = 8*src.index
        # if src.location == "PENB":
        #     src_pe = pe.category_id + 7 if (pe.is_head_pe) else pe.category_id - 1
        # elif src.location == "PEGB":
        #     src_pe = (pe.category_id // 8) * 8 + src.index
        # elif src.location == "PUNB":
        #     curr_pu = (pe.category_id // 8)
        #     src_pe = 56 if curr_pu == 0 else (curr_pu - 1)*8
        # elif src.location == "PUGB":
        #     src_pe = 8*src.index
        else:
            src_pe = pe.category_id

        assert type(src_pe) == type(pe.category_id)
        src_pes.append(src_pe)
    return src_pes

def get_dest_pe(instr, pe, sched):
    dst_pes = []
    if instr.check_dest("PENB"):
        # dst_pe = pe.category_id - 7 if (pe.category_id + 1) % 8 == 0 else pe.category_id + 1
        dst_pe = DEST_PE_ID_FN["PENB"](pe, -1, sched.pes_per_pu, sched.num_pus)

        assert type(dst_pe) == type(pe.category_id)

        dst_pes.append(dst_pe)

    if instr.check_dest("PEGB"):
        # dst_pe = (pe.category_id // 8) * 8 + instr._dest_pos["PEGB"][0].index
        dst_pe = DEST_PE_ID_FN["PEGB"](pe, instr._dest_pos["PEGB"][0].index, sched.pes_per_pu, sched.num_pus)

        dst_pes.append(dst_pe)


    if instr.check_dest("PUNB"):

        # curr_pu = (pe.category_id // 8)
        # dst_pe = 0 if curr_pu == 7 else (curr_pu + 1)*8
        dst_pe = DEST_PE_ID_FN["PUNB"](pe, -1, sched.pes_per_pu, sched.num_pus)

        dst_pes.append((dst_pe))


    if instr.check_dest("PUGB"):
        # dst_pe = 8*instr._dest_pos["PUGB"][0].index
        dst_pe = DEST_PE_ID_FN["PUGB"](pe, instr._dest_pos["PUGB"][0].index, sched.pes_per_pu, sched.num_pus)

        dst_pes.append((dst_pe))

    if not all([instr.check_dest(d) for d in ["NI", "NS"]]) and any([instr._dest_pos[d][0].index >= 0 for d in ["NI", "NS"]]):
        # dst_pe = pe.category_id
        dst_pe = DEST_PE_ID_FN["NI"](pe, -1, sched.pes_per_pu, sched.num_pus)

        dst_pes.append((dst_pe))

    return dst_pes



