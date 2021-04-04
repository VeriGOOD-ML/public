from . import Schedule
from . import TablaTemplate
from . import Namespace
from . import Bus, BusItem
from . import PE
from . import ScheduleEdge, ScheduleNode
from . import CYCLE_DELAYS, BUS_NAMES, NAMESPACES

def validate_graph(schedule: Schedule, arch: TablaTemplate):
    for node in schedule._dfg_nodes:
        if node.is_sink_node():
            check_sink_node(schedule, node, arch)
        if node.is_source_sink():
            continue
        elif node.is_data_node():
            check_data_node(schedule, node)
        else:
            check_node_instr(arch, schedule, node)
            if len(node.parents) == 1:
                check_single_input(schedule, node, arch)
            else:
                check_double_input(schedule, node)

def validate_instructions(arch: TablaTemplate):
    pes = [pe for _, pe in arch.category_component_dict["pe"].items() if isinstance(pe, PE)]
    for pe_id, pe in enumerate(pes):
        validate_pe_instr(pe)

def validate_pe_instr(pe):
    for cycle in range(pe.max_cycle + 1):
        if not pe.is_idle(cycle):
            check_instr(pe, cycle)

def check_instr(pe, cycle):
    instr = pe.get_instr(cycle)
    for s in instr.srcs:
        if s.location in NAMESPACES:
            ns = pe.get_namespace(s.location)
            if ns.find_all_data_use(s.data_id) < 0:
                raise ValueError(f"Data id {instr.node_id} not available in cycle {cycle}"
                                   f" in PE {pe.component_id}."
                                   f"\nInstruction: {instr}")
            elif ns.find_all_data_use(s.data_id) and not ns.is_data_present(cycle, s.data_id):
                first_use  = ns.find_first_use(s.data_id)
                raise ValueError(f"Data id {instr.node_id} not available in cycle {cycle}, first found in {first_use}"
                                   f" in PE {pe.component_id}."
                                   f"\nInstruction: {instr}")
            elif ns.find_data_index(s.data_id, cycle=cycle) != s.index:
                raise ValueError(f"Index for {instr.node_id} is incorrect in {cycle} in PE {pe.component_id}."
                                   f"\nCorrect index: {ns.find_data_index(s.data_id, cycle=cycle)}"
                                   f"\nInstruction: {instr}")



def check_sink_node(schedule: Schedule, node: ScheduleNode, arch: TablaTemplate):
    out_edge = schedule.get_schedule_edge(node.out_edges[0])
    for cid in out_edge.sink_components:
        ns = arch.component_map[cid].get_namespace("NW")
        if ns.find_all_data_use(out_edge.sink_node) < 0:
            raise RuntimeError(f"For node {node.node_id} not added\n")
        else:
            storage_item = ns.get_data(ns.max_cycle, out_edge.sink_node)
            if not storage_item.is_state_updated:
                raise ValueError(f"State node {out_edge.sink_node} is not updated for"
                                 f" edge {out_edge.edge_id} in namespace {ns.component_id} pe {cid}")

def check_edge_cycles(arch: TablaTemplate, schedule: Schedule, edge: ScheduleEdge):
    src_node = schedule.get_schedule_node(edge.source_id)

    if not src_node.is_data_node and src_node.exec_cycle != edge.path_cycles[0]:
        raise ValueError(f"Execution cycle for source node {src_node.node_id} is not equal to "
                         f"first cycle in edge {edge.edge_id}."
                         f"\n\tNode exec cycle: {src_node.exec_cycle}"
                         f"\n\tEdge path cycle: {edge.path_cycles[0]}")

    dst_node = schedule.get_schedule_node(edge.dest_id)
    if not dst_node.is_data_node and dst_node.exec_cycle != edge.path_cycles[-1]:
        raise ValueError(f"Execution cycle for dest node {dst_node.node_id} is not equal to "
                         f"first cycle in edge {edge.edge_id}."
                         f"\n\tNode exec cycle: {dst_node.exec_cycle}"
                         f"\n\tEdge path cycle: {edge.path_cycles[-1]}")

    if not dst_node.is_data_node and dst_node.exec_cycle != edge.ready_cycle:
        raise ValueError(f"Execution cycle for dest node {dst_node.node_id} is not equal to "
                         f"ready cycle {edge.edge_id}."
                         f"\n\tNode exec cycle: {dst_node.exec_cycle}"
                         f"\n\tEdge path cycle: {edge.path_cycles[-1]}")

    for idx, comp_id in enumerate(edge.path):
        component = arch.component_map[comp_id]

        if isinstance(component, Namespace):
            if edge.text_path[idx] != "NAMESPACE":
                raise ValueError(f"Edge {edge.edge_id} with data id {edge.data_id}"
                                 f" has mismatched text path and component:"
                                 f"\n\tText path: {edge.text_path}"
                                 f"\n\tNamespace: {component.component_subtype}")
            ns_added_cycle = component.find_first_use(edge.data_id)
            if ns_added_cycle < 0 and not edge.is_src_edge:
                raise ValueError(f"Edge {edge.edge_id} with data id {edge.data_id}"
                                 f" is not available in {component.component_subtype} "
                                 f"in any cycles.")

            if ns_added_cycle > edge.path_cycles[idx] and not edge.is_src_edge:
                raise ValueError(f"Edge {edge.edge_id} with data id {edge.data_id}"
                                 f" is added to {component.component_subtype} "
                                 f"in {ns_added_cycle} instead of {edge.path_cycles[idx]}.")
        elif isinstance(component, Bus):
            bus_item = BusItem(edge.path[idx - 1], edge.path[idx], edge.path[idx + 1], edge.data_id)
            if edge.text_path[idx] not in BUS_NAMES:
                raise ValueError(f"Edge {edge.edge_id} with data id {edge.data_id}"
                                 f" has mismatched text path and component:"
                                 f"\n\tText path: {edge.text_path}"
                                 f"\n\tBus: {component.component_subtype}")


            bus_added_cycle = component.find_first_use(bus_item)
            if bus_added_cycle < 0 and not edge.is_src_edge:
                raise ValueError(f"Edge {edge.edge_id} with data id {edge.data_id}"
                                 f" is not available in {component.component_subtype} "
                                 f"in any cycles.")

            if bus_added_cycle > edge.path_cycles[idx] and not edge.is_src_edge:
                raise ValueError(f"Edge {edge.edge_id} with data id {edge.data_id}"
                                 f" is added to {component.component_subtype} "
                                 f"in {bus_added_cycle} instead of {edge.path_cycles[idx]}.")


def check_node_instr(arch: TablaTemplate, schedule: Schedule, node: ScheduleNode):
    instr = node.get_instruction()
    srcs = instr.srcs

    for idx, in_edge_id in enumerate(node.in_edges):
        in_edge = schedule.get_schedule_edge(in_edge_id)
        if not in_edge.is_src_edge and not in_edge.is_sink_edge:
            start_cycle = in_edge.path_cycles[0]

            for cycle_costs in in_edge.path_cycles:
                if cycle_costs < 0:
                    raise ValueError(f"Edge {in_edge.edge_id} has invalid cycle cost in path: {in_edge.path_cycles}")
                elif cycle_costs < start_cycle:
                    raise ValueError(f"Edge {in_edge.edge_id} has non-increasing cycle cost order in path: {in_edge.path_cycles}")
                start_cycle = cycle_costs

        src_node = schedule.get_schedule_node(in_edge.source_id)
        if srcs[idx].data_id != in_edge.data_id and srcs[idx].location != "ALU":
            raise ValueError(f"Node with {node.node_id} with in-edge {in_edge_id} "
                             f"does not match the instruction data id: {srcs[idx].data_id}")

        if srcs[idx].source_id != in_edge.edge_id:
            raise ValueError(f"Node with {node.node_id} with in-edge {in_edge_id} "
                             f"does not match the instruction data id: {srcs[idx].source_id}")

        if srcs[idx].location in ["PEGB", "PUGB"]:
            pass
        elif srcs[idx].location in ["PENB", "PUNB"]:
            pass
        elif srcs[idx].location in ['NI','NW', 'NG', 'NM', 'ND']:
            src_comp = arch.component_map[node.component_id]

            if in_edge.namespace_name != srcs[idx].location:
                raise ValueError(f"Node with {node.node_id} with in-edge {in_edge_id} "
                                 f"does not match the instruction data type NS: {srcs[idx].location}"
                                 f" and instead has {in_edge.namespace_name}")
            namespace = src_comp.get_namespace(srcs[idx].location)
            if namespace.component_subtype == "NI":
                if namespace.is_data_present(src_node.exec_cycle + 1, in_edge.data_id) and not src_node.is_data_node():
                    raise ValueError(f"Source node with {src_node.node_id} with in-edge {in_edge_id} "
                                     f" has namespace data ready before it should be: "
                                     f"\n\tNamespace: {srcs[idx].location}"
                                     f"\n\tExec cycle: {src_node.exec_cycle}"
                                     f"\n\tAdded cycle: {namespace.find_first_use(in_edge.data_id)}")

            if not namespace.is_data_present(src_node.exec_cycle + CYCLE_DELAYS["PE"]["NAMESPACE"] + 1, in_edge.data_id):
                print(instr)
                raise ValueError(f"Source node with {src_node.node_id} with out-edge {src_comp.category_id} "
                                 f" does not have namespace data ready when it should be: "
                                 f"\n\tNamespace: {srcs[idx].location}"
                                 f"\n\tExec cycle: {src_node.exec_cycle + CYCLE_DELAYS['PE']['NAMESPACE']}"
                                 f"\n\tActual Cycle: {namespace.find_first_use(in_edge.data_id)}")

            index = namespace.find_data_index(in_edge.data_id, cycle=src_node.exec_cycle + CYCLE_DELAYS["PE"]["NAMESPACE"] + 1)
            if index != srcs[idx].index:
                raise ValueError(f"Source node with {src_node.node_id} with in-edge {in_edge_id} "
                                 f" has instruction with incorrect index: "
                                 f"\n\tNamespace: {srcs[idx].location}"
                                 f"\n\tInstr index: {srcs[idx].index}"
                                 f"\n\tNamespace index: {index}")
            check_edge_cycles(arch, schedule, in_edge)

    sink_parent = False
    if len(node.out_edges) == 1:
        out_edge = schedule.get_schedule_edge(node.out_edges[0])
        sink_parent = out_edge.is_sink_edge

    dest_ids = collect_dest_edges(node)

    # if set(node.out_edges) != set(dest_ids) and not sink_parent:
    #     raise ValueError(f"Number of destinations does not match the output edges for {node.node_id}:"
    #                      f"\n\tInstr edges: {dest_ids}"
    #                      f"\n\tOut edges: {node.out_edges}")

def collect_dest_edges(node: ScheduleNode):
    edge_ids = []

    dest_list = node.get_instruction().get_dests()
    for dest in dest_list:
        if dest.dest_id >= 0:
            edge_ids.append(dest.dest_id)

    for instr in node.comm_instructions:
        comm_dest_list = instr.get_dests()
        for comm_dest in comm_dest_list:
            if comm_dest.dest_id >= 0:
                edge_ids.append(comm_dest.dest_id)
    return edge_ids


def check_double_input(schedule: Schedule, node: ScheduleNode):
    # Check length of parents and input edges
    # check same dst component for in edges
    # check same src component for out edges

    if len(node.parents) > 2 or len(node.in_edges) > 2:
        raise ValueError(f"Multi input node {node.node_id} with edges {node.in_edges} "
                         f"and parents {node.parents} has too many inputs or parents")

    if node.exec_cycle < 0:
        raise ValueError(f"Multi input node {node.node_id} with edges {node.in_edges} "
                         f"has unset execution cycle")

    for in_edge_id in node.in_edges:
        in_edge = schedule.get_schedule_edge(in_edge_id)
        if in_edge.dst_component != node.component_id:
            raise ValueError(f"Multi input node {node.node_id} with in edges {node.in_edges} "
                             f"has edge {in_edge_id} with mismatched pe:"
                             f"\n\tNode component id: {node.component_id}"
                             f"\n\tEdge id: {in_edge.dst_component}")

        if in_edge.ready_cycle > node.exec_cycle:
            raise ValueError(f"Multi input node {node.node_id} with in edges {node.in_edges} "
                             f"has edge {in_edge_id} which is not ready to execute until "
                             f" cycle {in_edge.ready_cycle}, but the node has execution cycle "
                             f"{node.exec_cycle}")

    for out_edge_id in node.out_edges:
        out_edge = schedule.get_schedule_edge(out_edge_id)
        if out_edge.src_component != node.component_id:
            raise ValueError(f"Multi input node {node.node_id} with out edges {node.out_edges} "
                             f"has edge {out_edge_id} with mismatched pe:"
                             f"\n\tNode component id: {node.component_id}"
                             f"\n\tEdge id: {out_edge.src_component}")

def check_single_input(schedule: Schedule, node: ScheduleNode, arch: TablaTemplate):
    # Check the source and dest component are the same
    # Check the
    if len(node.in_edges) > 1:
        raise ValueError(f"Single node input Node {node.node_id} has mismatched parent amount "
                         f"and input edge count:\n\tInput edges: {node.in_edges}"
                         f"\n\tParents: {node.parents}")
    in_edge = schedule.get_schedule_edge(node.in_edges[0])
    parent_node = schedule.get_schedule_node(in_edge._src_id)
    if in_edge.src_component != in_edge.dst_component and arch.component_map[in_edge.src_component].is_idle(parent_node.exec_cycle + 1):
        raise ValueError(f"Single input node {node.node_id} with edge {in_edge.edge_id} "
                         f"does not have matching source and destinations:"
                         f"\n\tSource: {in_edge.src_component}\t"
                         f"Dest: {in_edge.dst_component}")


def check_data_node(schedule: Schedule, node: ScheduleNode):
    # Check each out edge has 0 as ready cycle
    # Check each out edge has same source and dest component
    # Check each path is length 3 and uses a namespace

    for out_edge_id in node.out_edges:
        out_edge = schedule.get_schedule_edge(out_edge_id)
        if out_edge.ready_cycle != 0 and out_edge.ready_cycle != 1 and schedule.is_src_child(node):
            raise ValueError(f"Data node {node.node_id} with edge {out_edge_id} "
                             f"does not have ready cycle 1: {out_edge.ready_cycle}")
        if out_edge.src_component != out_edge.dst_component:
            raise ValueError(f"Data node {node.node_id} with edge {out_edge_id} "
                             f"does not have matching source and destinations:"
                             f"\n\tSource: {out_edge.src_component}\t"
                             f"Dest: {out_edge.dst_component}")

        if len(out_edge.text_path) != 3:
            raise ValueError(f"Data node {node.node_id} with edge {out_edge_id} "
                             f"does not have length 3 path: {out_edge.text_path}")

        if out_edge.text_path[1] != "NAMESPACE":
            raise ValueError(f"Data node {node.node_id} with edge {out_edge_id} "
                             f"does not have length namespace path:"
                             f" {out_edge.text_path}")