from . import Schedule
from . import TablaTemplate
from . import PE
from . import NAMESPACES
from . import OP_SELECT_WIDTH, OP_WIDTH, MEM_INTERFACE_WIDTH, BUS_WIDTH
from .tabla_utils import SRC_PE_ID_FN, DEST_PE_ID_FN
from .vlg_templates import compute_instr_template
from .instruction_optimization import check_self_pass
import numpy as np
from collections import namedtuple, defaultdict
INSTR_FORMAT = ""
WriteInfo = namedtuple("WriteInfo", ['src', 'dst', 'eid', 'cycle'])

def generate_pe_instructions(schedule: Schedule, arch: TablaTemplate, package_path, debug="both", use_instr_list=False):
    if not schedule.is_dfg_loaded():
        raise RuntimeError(f"Schedule has not loaded a DFG yet.")
    log_size = lambda x: int(np.ceil(np.log2(x))) if x > 1 else 1
    pes = [pe for _, pe in arch.category_component_dict["pe"].items() if isinstance(pe, PE)]
    config, instr_summary = compute_instr_widths(arch)
    address_width = log_size(arch.max_instr)
    reads_comm_dict = defaultdict(list)
    writes_comm_dict = defaultdict(list)
    pe_blocks = ["generate\n"]
    max_pe = -1
    total_instr = 0
    for pe_id, pe in enumerate(pes):
        busy_cycles = pe.busy_cycles()
        if pe.component_id <= max_pe:
            raise RuntimeError(f"Not iterating over components in correct sequence")
        else:
            max_pe = pe.component_id

        pe_str = f"if(peId == {pe_id}) begin\n" \
            f"\talways @(*) begin\n" \
            f"\t\tcase(address)\n"
        if use_instr_list:
            pe_instr = pe.instructions
        else:
            pe_instr = pe.all_instructions()
        total_instr += len(pe_instr)
        max_added = -1
        ni_writes = {}
        for instr_num, instr in enumerate(pe_instr):
            if instr.cycle_insert[0] < max_added:
                print(f"Error! Ordering of added instructions is incorrect for {instr.node_id}")
            bin_str = f"\t\t\t{address_width}'d{instr_num} : rdata = {config['instr_len']}'b{instr.to_verilog_bin(config)};\n"
            if debug == "data_id":
                src_annotation = ", ".join([str(i.data_id) for i in instr.srcs])
                dest_annotation = str(instr.node_id)
            elif debug == "values":
                src_annotation, dest_annotation = get_data_values(instr, schedule)
            elif debug == "both":
                pe_annot = update_comm_dict(reads_comm_dict, writes_comm_dict, instr, pe, busy_cycles[instr_num], ni_writes, schedule)
                src_ids = [str(i.data_id) for i in instr.srcs]
                dest_id = str(instr.node_id)
                src_vals, dest_val = get_data_values(instr, schedule, as_list=True)

                assert len(src_vals) == len(src_ids)
                src_annotation = str(instr.cycle_insert) + ", ".join([f"({src_ids[i]}) {src_vals[i]}" for i in range(len(src_vals))])
                dest_annotation = f"({dest_id}) {dest_val}"
            else:
                src_annotation = ""
                dest_annotation = ""
            bin_str = f"\t\t\t// srcs: {src_annotation} --> {dest_annotation}:{instr}\n{bin_str}"

            if debug == "both":
                bin_str = f"\t\t\t// PEs: {pe_annot}\n{bin_str}"
            pe_str += bin_str
        pe_str += f"\t\t\tdefault : rdata = {config['instr_len']}'b{np.binary_repr(0, width=config['instr_len'])};\n"
        pe_str += f"\t\tendcase\n"
        pe_str += f"\tend\n"
        pe_str += f"end\n"
        pe_str += f"\n"
        pe_blocks.append(pe_str)


    if debug == "both":
        check_pe_comms(arch, reads_comm_dict, writes_comm_dict)

    print(f"Total instructions: {total_instr}\n")
    pe_blocks.append("endgenerate")
    write_instr_file(schedule, pe_blocks, package_path)
    return total_instr, instr_summary

def check_pe_comms(arch, reads_comm_dict, writes_comm_dict):
    checked = []
    num_errors = 0

    for source, dest in list(reads_comm_dict.keys()):
        if (source, dest) in checked:
            continue
        rcycles, rdata_ids, redge_ids = zip(*[(r.cycle, r.src, r.eid) for r in reads_comm_dict[(source, dest)]])
        if len(writes_comm_dict[(source, dest)]) == 0:
            if source != dest:
                num_errors += 1
                iwr = [i[1] for i in writes_comm_dict[(source, dest)]]
                ir = [i[1] for i in reads_comm_dict[(source, dest)]]
                print(f"Unequal W/R: PE{source} to PE{dest}:\n\t"
                      f"Instr Writes: {iwr}\n\t"
                      f"Instr reads: {ir}\n\t")
            continue
        # wcycles, wdata_ids = zip(*writes_comm_dict[(source, dest)])
        wcycles, wdata_ids, wedge_ids = zip(*[(w.cycle, w.src, w.eid) for w in writes_comm_dict[(source, dest)]])


        if rdata_ids != wdata_ids and source != dest:
            num_errors += 1
            print(f"Unequal W/R: PE{source} to PE{dest}:\n\t"
                  f"Instr Write ids: {wdata_ids}\n\t"
                  f"Instr read ids: {rdata_ids}\n\t"
                  f"Instr Write edge ids: {wedge_ids}\n\t"
                  f"Instr read edge ids: {redge_ids}\n\t"
                  f"")

        checked.append((source, dest))
    print(f"Totals errors: {num_errors}")

def update_comm_dict(reads_comm_dict, writes_comm_dict, instr, pe, cycle, ni_writes, sched):
    src_pes = []
    dst_pes = []
    for src in instr.srcs:
        if src.location in ["PENB", "PEGB", "PUNB", "PUGB"]:
            src_pe = SRC_PE_ID_FN[src.location](pe, src.index, sched.pes_per_pu, sched.num_pus)

        else:
            if src.location == "NI" and src.index not in ni_writes:
                assert src.index >= 0
                print(f"Instruction {instr} in PE{pe.category_id} with node id: {instr.node_id} "
                      f"is reading from {src} without the namespace having been written to.\n")
            src_pe = pe.category_id
        assert type(src_pe) == type(pe.category_id)
        # reads_comm_dict[(src_pe, pe.category_id)].append((cycle, src.data_id))
        edge = sched.get_schedule_edge(src.source_id)
        r_info = WriteInfo(src=edge.source_id, dst=edge.dest_id, eid=edge.edge_id, cycle=cycle)
        # reads_comm_dict[(src_pe, pe.category_id)].append((cycle, src.data_id))
        reads_comm_dict[(src_pe, pe.category_id)].append(r_info)

        src_pes.append(str(src_pe))

    if instr.check_dest("PENB"):
        # dst_pe = pe.category_id - 7 if (pe.category_id + 1) % 8 == 0 else pe.category_id + 1
        dst_pe = DEST_PE_ID_FN["PENB"](pe, -1, sched.pes_per_pu, sched.num_pus)
        assert type(dst_pe) == type(pe.category_id)

        dst_pes.append(str(dst_pe))
        edge = sched.get_schedule_edge(instr.get_dest("PENB").dest_id)
        wr_info = WriteInfo(src=edge.source_id, dst=edge.dest_id, eid=edge.edge_id, cycle=cycle)
        # writes_comm_dict[(pe.category_id, dst_pe)].append((cycle, instr.node_id))
        writes_comm_dict[(pe.category_id, dst_pe)].append(wr_info)

    if instr.check_dest("PEGB"):
        # dst_pe = (pe.category_id // 8) * 8 + instr._dest_pos["PEGB"][0].index
        dst_pe = DEST_PE_ID_FN["PEGB"](pe, instr._dest_pos["PEGB"][0].index, sched.pes_per_pu, sched.num_pus)

        dst_pes.append(str(dst_pe))
        edge = sched.get_schedule_edge(instr.get_dest("PEGB").dest_id)
        wr_info = WriteInfo(src=edge.source_id, dst=edge.dest_id, eid=edge.edge_id, cycle=cycle)
        assert type(dst_pe) == type(pe.category_id)
        # writes_comm_dict[(pe.category_id, dst_pe)].append((cycle, instr.node_id))
        writes_comm_dict[(pe.category_id, dst_pe)].append(wr_info)

    if instr.check_dest("PUNB"):

        # curr_pu = (pe.category_id // 8)
        # dst_pe = 0 if curr_pu == 7 else (curr_pu + 1)*8
        dst_pe = DEST_PE_ID_FN["PUNB"](pe, -1, sched.pes_per_pu, sched.num_pus)

        dst_pes.append(str(dst_pe))
        assert type(dst_pe) == type(pe.category_id)
        edge = sched.get_schedule_edge(instr.get_dest("PUNB").dest_id)
        wr_info = WriteInfo(src=edge.source_id, dst=edge.dest_id, eid=edge.edge_id, cycle=cycle)
        # writes_comm_dict[(pe.category_id, dst_pe)].append((cycle, instr.node_id))
        writes_comm_dict[(pe.category_id, dst_pe)].append(wr_info)

    if instr.check_dest("PUGB"):
        # dst_pe = 8*instr._dest_pos["PUGB"][0].index
        dst_pe = DEST_PE_ID_FN["PUGB"](pe, instr._dest_pos["PUGB"][0].index, sched.pes_per_pu, sched.num_pus)

        dst_pes.append(str(dst_pe))
        assert type(dst_pe) == type(pe.category_id)
        edge = sched.get_schedule_edge(instr.get_dest("PUGB").dest_id)
        wr_info = WriteInfo(src=edge.source_id, dst=edge.dest_id, eid=edge.edge_id, cycle=cycle)
        # writes_comm_dict[(pe.category_id, dst_pe)].append((cycle, instr.node_id))
        writes_comm_dict[(pe.category_id, dst_pe)].append(wr_info)
    if not all([instr.check_dest(d) for d in ["NI", "NS"]]) and any([instr._dest_pos[d][0].index >= 0 for d in ["NI", "NS"]]):
        # dst_pe = pe.category_id
        dst_pe = DEST_PE_ID_FN["NI"](pe, -1, sched.pes_per_pu, sched.num_pus)

        dst_pes.append(str(dst_pe))
        dst_loc = "NI" if instr._dest_pos["NI"][0].index >= 0 else "NS"
        edge = sched.get_schedule_edge(instr.get_dest(dst_loc).dest_id)
        wr_info = WriteInfo(src=edge.source_id, dst=edge.dest_id, eid=edge.edge_id, cycle=cycle)
        # writes_comm_dict[(pe.category_id, dst_pe)].append((cycle, instr.node_id))
        writes_comm_dict[(pe.category_id, dst_pe)].append(wr_info)
        if instr.check_dest("NI"):
            ni_dest = instr.get_dest("NI")
            assert ni_dest.index >= 0
            ni_writes[ni_dest.index] = 1 if ni_dest.index not in ni_writes else ni_writes[ni_dest.index] + 1

    return f"{', '.join(src_pes)} -> {', '.join(dst_pes)}"


def get_data_values(instr, schedule, as_list=False):
    src_vals = []
    assert instr.node_id in schedule.dfg_node_map
    instr_node = schedule.dfg_node_map[instr.node_id]
    for idx, i in enumerate(instr.srcs):
        if i.data_id in schedule.dfg_node_map and i.location != "ALU":
            src_vals.append(str(schedule.dfg_node_map[i.data_id].computed))
        elif instr_node.parents[idx] in schedule.dfg_node_map:
            src_vals.append(str(schedule.dfg_node_map[instr_node.parents[idx]].computed))
        else:
            print(f"Instruction source without node: {i.data_id} \t {instr}")
    if not as_list:
        src_annotation = ", ".join(src_vals)
    else:
        src_annotation = src_vals
    dest_annotation = str(schedule.dfg_node_map[instr.node_id].computed)

    return src_annotation, dest_annotation

def compute_instr_widths(arch, debug=True):
    config = arch.config
    max_util = arch.get_max_util()
    log_size = lambda x: int(np.ceil(np.log2(x))) if x > 1 else 1
    all_max = max(list(max_util.values()))
    # pe_idx_max = max(config["pes_per_pu"], log_size(config["num_pes"] / config["pes_per_pu"]))
    pe_idx_max = max(log_size(config["pes_per_pu"]), log_size(config["num_pes"] / config["pes_per_pu"])) + 1
    cfg_max = max([config["namespace_size"], config["namespace_interim_size"]])
    ni_op_index_bits = log_size(max_util['NI'])
    ns_op_index_bits = log_size(max(max_util['ND'], max_util['NW']))

    op_index_bits = max(pe_idx_max, log_size(min(all_max, cfg_max)))
    print(op_index_bits)
    pe_bus_width = log_size(config["pes_per_pu"]) + 1
    pu_bus_width = log_size(config["num_pes"] / config["pes_per_pu"]) + 1

    new_sum = OP_WIDTH + 2 * (op_index_bits + OP_SELECT_WIDTH) + (1 + ni_op_index_bits) + (
                1 + ns_op_index_bits) + 2 + pe_bus_width + pu_bus_width
    instr_bin_summary = f"Calc:\n\t" \
                        f"Op code: {OP_WIDTH}\n\t" \
                        f"Op1 total: {op_index_bits + OP_SELECT_WIDTH}\n\t" \
                        f"Op1 index: {op_index_bits}\n\t" \
                        f"Op2 total: {op_index_bits + OP_SELECT_WIDTH}\n\t" \
                        f"Op2 index: {op_index_bits}\n\t" \
                        f"NI Total: {1 + ni_op_index_bits}\n\t" \
                        f"NI index: {ni_op_index_bits}\n\t" \
                        f"NS Total: {1 + ns_op_index_bits}\n\t" \
                        f"NS Index: {ns_op_index_bits}\n\t" \
                        f"Nieghbors: {2}\n\t" \
                        f"PEGB: {pe_bus_width}\n\t" \
                        f"PUGB: {pu_bus_width}\n" \
                        f"Total bitwidth size: {new_sum}\n"
    if debug:
        print(instr_bin_summary)

    return {"op_index_bits": op_index_bits, "pe_bus_width": pe_bus_width,
            "pu_bus_width": pu_bus_width, "instr_len": new_sum,
            "ni_index_bits": ni_op_index_bits, "ns_index_bits": ns_op_index_bits}, instr_bin_summary


def write_instr_file(schedule: Schedule, pe_instrs, package_path):
    out_path = f"{package_path}/compute-inst/instruction_memory.v"
    instrs = "".join(pe_instrs)
    file_instr = compute_instr_template.format(compute_instr=instrs)
    with open(out_path, "w") as f:
        f.write(file_instr)


