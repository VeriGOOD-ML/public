from pytools import memoize_method
from . import Component
from . import Instruction, determine_edge_path
from . import Bus, BusItem
from . import State
from . import PU
from . import Namespace
from . import ScheduleEdge
import numpy as np
from . import PE
from .tabla_utils import get_instr_cat_pe, compute_cost, find_dest_loc, \
    update_temp_edge_paths, map_equality, map_rwr_equality, ReadWriteInfo, \
    EdgeReadWriteInfo, SRC_PE_ID_FN, DEST_PE_ID_FN
from typing import Dict, Union, List, Optional, Tuple
from . import CYCLE_DELAYS
from collections import defaultdict, namedtuple
import bisect
WriteInfo = namedtuple("WriteInfo", ['src', 'dst', 'eid', 'cycle'])
# from line_profiler import LineProfiler
# import atexit
# profile = LineProfiler()
# atexit.register(profile.print_stats)

class TablaTemplate(Component):

    possible_states = ['busy', 'free']
    idle_state = 'free'

    def __init__(self, config):
        """

        Parameters
        ----------
        config : Dict object
                 JSON configurations specifying the following:
                 - Total number of PEs (`num_pes`)
                 - The number of PEs per PU (`pes_per_pu`)
                 - The amount of allowed entries in a weight, gradient,
                    and data namespace (`namespace_size`)
                 - The amount of allowed entries in an interim namespace (`namespace_interim_size`)
                 - The amount of bits designated for an opcode (`op_bit`)
                 - The amount of bits designated for a namespace (`ns_bit`)
                 - The amount of bits designated for a bus (`nn_nb_bit`)
                 - Whether or not to generate instructions in hexadecimal format (`hex`)
        """
        super(TablaTemplate, self).__init__("tabla", "")
        self._config = config
        self.cycle_from_map = lambda x: x.cycle
        self.rcycle_edge_map = defaultdict(list)
        self.wcycle_edge_map = defaultdict(list)
        self.rcycle_data_map = defaultdict(list)
        self.wcycle_data_map = defaultdict(list)
        self.pe_r_map = defaultdict(list)
        self.pe_wr_map = defaultdict(list)
        self.component_id = self.resource_id

        self.num_pus = config["num_pes"] // config["pes_per_pu"]
        if self.num_pus == 1:
            self._pu_map = {}
            self._pu_map['PU0'] = PU(config["pes_per_pu"],None, config["namespace_size"], config["namespace_interim_size"])
        else:
            self._pu_map = {
                f"PU{Component.category_resource_counter['pu'] - 1}": PU(config["pes_per_pu"],
                                                                         None,
                                                                         config["namespace_size"],
                                                                         config["namespace_interim_size"])
                for _ in range(self.num_pus)
            }

        pus = [self._pu_map[pu].category_id for pu in self._pu_map]
        pugb = Bus("PUGB", buffer_size=4, pus=pus)
        for pu_id in self._pu_map:
            pu = self._pu_map[pu_id]
            pu._pugb = pugb.component_id
        self._bus_map = {"PUGB": pugb}


        for i in range(self.num_pus):
            self._bus_map[f"PUNB{Component.category_resource_counter['bus']}"] = Bus("PUNB")
            bus_id = self._bus_map[f"PUNB{Component.category_resource_counter['bus']}"].component_id
            self.cat_component_map['pu'][i].set_neighbor_bus(bus_id)
        self._ordered_pes = list(self.pe_costs.keys())
        self._pe_perms = np.arange(len(self._ordered_pes))

    def pe_utilization(self):
        utilization = 0
        for _, pe in self.cat_component_map['pe'].items():
            utilization += pe.utilization
        avg_utilization = 100.0*utilization/len(self.cat_component_map['pe'].keys())

        return avg_utilization

    def ni_read_writes(self, post_opt=False):
        instr_info = {}
        for pe_id, pe in self.cat_component_map['pe'].items():
            instr_info[pe.category_id] = {}
            pe_instr = pe.all_instructions()

            for instr_num, instr in enumerate(pe_instr):
                if instr.check_dest("NI"):
                    dst = instr.get_dest("NI")
                    ni_index = instr.get_dest("NI").index

                    if ni_index in instr_info[pe.category_id]:
                        finish = max(instr_info[pe.category_id][ni_index].finish, instr_num)
                        if instr_info[pe.category_id][ni_index].dataid != dst.data_id and instr_info[pe.category_id][ni_index].finish >= 0:
                            meta_info = f"DIFFERENT DATA IDS for {instr}: " \
                                f"({instr_info[pe.category_id][ni_index].dataid} --> {dst.data_id})"
                        elif instr.srcs[0].location == "NI" and instr.op_name == "pass":
                            meta_info = "second write, pass to self"
                        else:
                            meta_info = f"second write, overwrite: {instr}"

                    else:
                        finish = -1
                        meta_info = ""

                    instr_info[pe.category_id][ni_index] = ReadWriteInfo(start=instr_num, finish=finish, meta=meta_info, dataid=dst.data_id, eid=dst.dest_id)

                if instr.srcs[0].location == "NI":
                    ni_index = instr.srcs[0].index
                    src = instr.srcs[0]

                    prev = instr_info[pe.category_id][ni_index]
                    if prev.finish > instr_num:
                        raise RuntimeError
                    instr_info[pe.category_id][ni_index] = ReadWriteInfo(start=prev.start, finish=instr_num, meta=prev.meta, dataid=src.data_id, eid=src.source_id)

                if len(instr.srcs) > 1 and instr.srcs[1].location == "NI":
                    ni_index = instr.srcs[1].index
                    src = instr.srcs[1]

                    prev = instr_info[pe.category_id][ni_index]
                    if prev.finish > instr_num:
                        raise RuntimeError
                    instr_info[pe.category_id][ni_index] = ReadWriteInfo(start=prev.start, finish=instr_num, meta=prev.meta, dataid=src.data_id, eid=src.source_id)


        return instr_info

    @property
    def ordered_pes(self):
        ret = self._ordered_pes
        first = self._ordered_pes.pop(0)
        self._ordered_pes.append(first)
        return ret

    @property
    def perm_pes(self):
        return [self._ordered_pes[i] for i in np.random.permutation(self._pe_perms)]

    def ni_edge_read_writes(self, post_opt=False):
        instr_info = {}
        for pe_id, pe in self.cat_component_map['pe'].items():
            instr_info[pe.category_id] = {}
            pe_instr = pe.all_instructions()

            for instr_num, instr in enumerate(pe_instr):
                if instr.check_dest("NI"):
                    dst = instr.get_dest("NI")
                    ni_index = dst.index
                    ni_edge = dst.dest_id

                    if ni_edge in instr_info[pe.category_id]:
                        raise RuntimeError(f"Multiple writes for edge {ni_edge} with data id {dst.data_id}\n")
                    finish = -1
                    meta_info = ""
                    for eid in dst.all_dests:
                        instr_info[pe.category_id][eid] = EdgeReadWriteInfo(start=instr_num, finish=finish, meta=meta_info, dataid=dst.data_id, index=ni_index)

                if instr.srcs[0].location == "NI":
                    src = instr.srcs[0]
                    ni_index = src.index
                    ni_edge = src.source_id
                    if ni_edge not in instr_info[pe.category_id]:
                        raise KeyError(f"{ni_edge} with data id {src.data_id} not found in instr info on PE"
                                       f"{pe.category_id}.")
                    prev = instr_info[pe.category_id][ni_edge]
                    if prev.finish > instr_num or prev.finish >= 0 or prev.index != ni_index:
                        raise RuntimeError
                    instr_info[pe.category_id][ni_edge] = EdgeReadWriteInfo(start=prev.start, finish=instr_num, meta=prev.meta, dataid=src.data_id, index=ni_index)

                if len(instr.srcs) > 1 and instr.srcs[1].location == "NI":
                    src = instr.srcs[1]
                    ni_index = src.index
                    ni_edge = src.source_id
                    if ni_edge not in instr_info[pe.category_id]:
                        raise KeyError(f"{ni_edge} with data id {src.data_id} not found in instr info on PE"
                                       f"{pe.category_id}.")
                    prev = instr_info[pe.category_id][ni_edge]
                    if prev.finish > instr_num or prev.finish >= 0 or prev.index != ni_index:
                        raise RuntimeError
                    instr_info[pe.category_id][ni_edge] = EdgeReadWriteInfo(start=prev.start, finish=instr_num,
                                                                        meta=prev.meta, dataid=src.data_id,
                                                                        index=ni_index)
        return instr_info

    def namespace_utilization(self, namespaces=None):
        utilization = {}
        if not namespaces:
            namespaces = ["NW", "NI", "ND", "NM"]
        elif not isinstance(namespaces, list):
            namespaces = [namespaces]

        for pe_id, pe in self.cat_component_map['pe'].items():
            utilization[f"PE{pe_id}"] = {}
            for nsname in namespaces:
                ns = pe.get_namespace(nsname)
                utilization[f"PE{pe_id}"][nsname] = (ns.utilization(pe.max_cycle), ns.num_unique(pe.max_cycle))

        return utilization

    def pu_utilization(self):
        utilization = {}
        for pu_id, pu in self.cat_component_map['pu'].items():
            utilization[f"PU{pu_id}"] = pu.utilization
        return utilization

    def get_max_util(self):
        util = self.namespace_utilization(namespaces=["NI", "NW", "ND"])
        maxes = {"NI": -1, "ND": -1, "NW": -1}
        for pe, ns_util in util.items():
            for ns_name, util in ns_util.items():
                if util[1][0] > maxes[ns_name]:
                    maxes[ns_name] = util[1][0]
        return maxes

    @memoize_method
    def get_namespace_size(self, ns_name: str) -> int:
        namespace = self.cat_component_map['pe'][0].get_namespace(ns_name)
        return namespace.capacity

    @property
    def config(self) -> Dict[str,int]:
        return self._config

    @property
    @memoize_method
    def component_map(self) -> Dict[int, Union[PE, Component, PU, Bus, Namespace]]:
        """
        Resource ID to Component mapping.

        0: pu_obj,
        1: pe_obj

        Returns
        -------

        """
        cmp_dict = {}
        for _, pu in self.pu_map.items():
            cmp_dict[pu.component_id] = pu
            for _, pe in pu._pe_map.items():
                cmp_dict[pe.component_id] = pe
                for _, ns in pe._namespace_map.items():
                    cmp_dict[ns.component_id] = ns

            for _, bus in pu._bus_map.items():
                cmp_dict[bus.component_id] = bus

        for _, bus in self._bus_map.items():
            cmp_dict[bus.component_id] = bus

        return cmp_dict

    def path_instr_len(self, src_pe, dst_pe):
        if src_pe < 0:
            return 0
        path = self.determine_path(src_pe, dst_pe)
        sizes = 0
        for i in path:
            if i in [src_pe, dst_pe] or not isinstance(self.component_map[i], PE):
                continue
            else:
                sizes += self.component_map[i].all_instr_len()
        return sizes

    @property
    @memoize_method
    def pes_per_pu(self) -> int:
        return len(self.pu_map["PU0"].pe_ids)

    @property
    def total_instructions(self):
        pes = [pe for _, pe in self.category_component_dict["pe"].items() if isinstance(pe, PE)]
        return sum([len(pe.all_instructions()) for pe in pes])

    @property
    def max_instr(self) -> int:
        max_len = -1
        for pe_id, pe in self.category_component_dict["pe"].items():
            if isinstance(pe, PE):
                instr_len = len(pe.all_instructions())
                if instr_len > max_len:
                    max_len = instr_len
        return max_len

    @property
    @memoize_method
    def pe_paths(self) -> Dict[int, Dict[int,List[str]]]:
        """
        Path to get from one PE to another.

        0: pu_obj,
        1: pe_obj

        Returns
        -------

        """
        path_dict = {}
        for pe_id, pe in self.cat_component_map['pe'].items():
            path_dict[pe.component_id] = {}
            for _, dst_pe in self.cat_component_map['pe'].items():
                cost = self.determine_cost_path(pe.component_id, dst_pe.component_id)
                path_dict[pe.component_id][dst_pe.component_id] = cost
        return path_dict

    @property
    @memoize_method
    def pe_costs(self) -> Dict[int, Dict[int,int]]:
        """
        Dictionary cost in cycles from one PE to another PE.

        Returns
        -------

        """
        cost_dict = {}
        for src_pe_id, src_path_dict in self.pe_paths.items():
            cost_dict[src_pe_id] = {}
            for dst_pe_id, path in src_path_dict.items():
                cost_dict[src_pe_id][dst_pe_id] = compute_cost(path)

        return cost_dict

    @memoize_method
    def get_data_location(self, comp_id: int, edge: ScheduleEdge) -> str:
        component = self.component_map[comp_id]
        if edge.alu_edge:
            return "ALU"
        elif isinstance(component, Namespace) or isinstance(component, Bus):
            return component.component_subtype
        else:
            raise ValueError(f"Component id {comp_id}  with type {type(component)}"
                             f"is not a valid data location."
                             f"\n\tData id: {edge.data_id}")


    def compute_data_dest(self, cycle: int, comp_id: int, pe_id: int, edge: ScheduleEdge) -> Tuple[str, int]:

        component = self.component_map[comp_id]
        index = -1
        if isinstance(component, Namespace):
            ns_delay = cycle + CYCLE_DELAYS["PE"]["NAMESPACE"] + 1
            index = component.find_data_index(edge.data_id, cycle=ns_delay)
            location = component.component_subtype
        elif isinstance(component, Bus):
            if component.component_subtype == "PEGB":
                index = self.component_map[pe_id].category_id % self.pes_per_pu
            elif component.component_subtype == "PUGB":
                pu_id = self.component_map[pe_id].pu_id
                index = self.component_map[pu_id].category_id
            location = component.component_subtype
        elif edge.alu_edge:
            location = "ALU"
        else:
            raise ValueError(f"Component id {comp_id}  with type {type(component)}"
                             f"is not a valid data location."
                             f"\n\tData id: {edge.data_id}\n\tPE id: {pe_id}")
        return location, index


    def get_output_idx(self, cycle, pe_id, edge):
        ns_delay = cycle + CYCLE_DELAYS["PE"]["NAMESPACE"] + 1
        exec_pe = self.component_map[pe_id]
        ns = exec_pe.get_namespace(edge.namespace_name)
        self.add_namespace_data(ns_delay, pe_id, ns.component_subtype, edge)

        index = ns.find_data_index(edge.data_id, cycle=ns_delay)
        location = ns.component_subtype
        return location, index

    def compute_data_source(self, cycle: int, sub_path: List[int], edge: ScheduleEdge) -> Tuple[str, int]:
        component = self.component_map[sub_path[1]]
        index = -1

        if isinstance(component, Namespace):
            ns_delay = cycle + CYCLE_DELAYS["PE"]["NAMESPACE"] + 1
            self.add_namespace_data(ns_delay, sub_path[2], component.component_subtype, edge)
            edge.set_path_cycle(edge.get_component_index(sub_path[1]), ns_delay)
            index = component.find_data_index(edge.data_id, cycle=ns_delay)
            location = component.component_subtype

        elif isinstance(component, Bus):
            # TODO: Add data to namespace bus
            if component.component_subtype == "PEGB":
                index = self.component_map[sub_path[2]].category_id % self.pes_per_pu
            elif component.component_subtype == "PUGB":
                pu_id = self.component_map[sub_path[2]].pu_id
                index = self.component_map[pu_id].category_id
            bus_delay = cycle + CYCLE_DELAYS["PE"][component.component_subtype] + 1
            # component.add_buffer_data(bus_delay, sub_path[0], sub_path[1], sub_path[2], edge.data_id)
            edge.set_path_cycle(edge.get_component_index(sub_path[1]), bus_delay)
            location = component.component_subtype
        elif edge.alu_edge:
            # TODO: add data to alu and add alu to PE metadata
            location = "ALU"
        else:
            raise ValueError(f"Component id {sub_path[1]}  with type {type(component)}"
                             f"is not a valid data location."
                             f"\n\tData id: {edge.data_id}\n\tPE id: {sub_path[2]}")
        return location, index

    @memoize_method
    def determine_cost_path(self, src_pe_id: int, dst_pe_id: int) -> List[str]:
        path = []
        src_pe = self.component_map[src_pe_id]
        dst_pe = self.component_map[dst_pe_id]

        # TODO: Need to determine if same pe means through namespace or ALU
        # TODO: Determine if destination is a PE or namespace.
        if src_pe_id == dst_pe_id:
            path.append("NAMESPACE")
        elif self.get_pe_neighbor(src_pe_id) == dst_pe.component_id:
            path.append("PENB")
        elif src_pe.pu_id == dst_pe.pu_id:
            path.append("PEGB")
        elif self.get_pu_neighbor(src_pe.pu_id) == dst_pe.pu_id:
            src_pu_head_id = self.get_head_pe(src_pe.pu_id)
            dst_pu_head_id = self.get_head_pe(dst_pe.pu_id)
            if src_pe_id != src_pu_head_id:
                if self.get_pe_neighbor(src_pe_id) == src_pu_head_id:
                    path.append("PENB")
                else:
                    path.append("PEGB")

                path.append("PE")

            path.append("PUNB")
            path.append("PE")
            if dst_pe_id != dst_pu_head_id:
                if self.get_pe_neighbor(dst_pu_head_id) == dst_pe_id:
                    path.append("PENB")
                else:
                    path.append("PEGB")

        else:
            src_pu_head_id = self.get_head_pe(src_pe.pu_id)
            dst_pu_head_id = self.get_head_pe(dst_pe.pu_id)
            if src_pe_id != src_pu_head_id:
                if self.get_pe_neighbor(src_pe_id) == src_pu_head_id:
                    path.append("PENB")
                else:
                    path.append("PEGB")

                path.append("PE")

            path.append("PUGB")
            path.append("PE")
            if dst_pe_id != dst_pu_head_id:
                if self.get_pe_neighbor(dst_pu_head_id) == dst_pe_id:
                    path.append("PENB")
                else:
                    path.append("PEGB")

        if path[-1] != "PE":
            path.append("PE")

        return ["PE"] + path

    @memoize_method
    def determine_path(self, src_pe_id: int, dst_pe_id: int) -> List[int]:
        path = []
        src_pe = self.component_map[src_pe_id]
        dst_pe = self.component_map[dst_pe_id]

        # TODO: Need to determine if same pe means through namespace or ALU
        # TODO: Determine if destination is a PE or namespace.
        if src_pe_id == dst_pe_id:
            path.append(src_pe.get_namespace('NI').component_id)
        elif self.get_pe_neighbor(src_pe_id) == dst_pe.component_id:
            path.append(src_pe.neighbor_bus)
        elif src_pe.pu_id == dst_pe.pu_id:
            path.append(src_pe.global_bus)
        elif self.get_pu_neighbor(src_pe.pu_id) == dst_pe.pu_id:
            src_pu_head_id = self.get_head_pe(src_pe.pu_id)
            dst_pu_head_id = self.get_head_pe(dst_pe.pu_id)
            if src_pe_id != src_pu_head_id:
                if self.get_pe_neighbor(src_pe_id) == src_pu_head_id:
                    path.append(src_pe.neighbor_bus)
                else:
                    path.append(src_pe.global_bus)

                path.append(src_pu_head_id)

            path.append(self.component_map[src_pe.pu_id].neighbor_bus)
            path.append(dst_pu_head_id)
            if dst_pe_id != dst_pu_head_id:
                if self.get_pe_neighbor(dst_pu_head_id) == dst_pe_id:
                    path.append(self.component_map[dst_pu_head_id].neighbor_bus)
                else:
                    path.append(self.component_map[dst_pu_head_id].global_bus)

        else:
            src_pu_head_id = self.get_head_pe(src_pe.pu_id)
            dst_pu_head_id = self.get_head_pe(dst_pe.pu_id)
            if src_pe_id != src_pu_head_id:
                if self.get_pe_neighbor(src_pe_id) == src_pu_head_id:
                    path.append(src_pe.neighbor_bus)
                else:
                    path.append(src_pe.global_bus)

                path.append(dst_pu_head_id)

            path.append(self.component_map[src_pe.pu_id].global_bus)
            path.append(dst_pu_head_id)
            if dst_pe_id != dst_pu_head_id:
                if self.get_pe_neighbor(dst_pu_head_id) == dst_pe_id:
                    path.append(self.component_map[dst_pu_head_id].neighbor_bus)
                else:
                    path.append(self.component_map[dst_pu_head_id].global_bus)

        if path[-1] != dst_pe_id:
            path.append(dst_pe_id)

        return [src_pe_id] + path

    @property
    @memoize_method
    def cat_component_map(self) -> Dict[str, Dict[int, Union[PE, Component, PU, Bus, Namespace]]]:
        """
        Category string to Category ID to Component mapping.

        'pu' : {0: obj}
        'pe' : {0: pe_obj}

        Returns
        -------

        """
        cat_cmp_dict = {'pu' : {},
                        'pe' : {},
                        'bus' : {},
                        'namespace' : {}}
        for _, pu in self.pu_map.items():
            cat_cmp_dict['pu'][pu.category_id] = pu
            for _, pe in pu._pe_map.items():
                cat_cmp_dict['pe'][pe.category_id] = pe
                for _, ns in pe._namespace_map.items():
                    cat_cmp_dict['namespace'][ns.category_id] = ns

            for _, bus in pu._bus_map.items():
                cat_cmp_dict['bus'][bus.category_id] = bus

        for _, bus in self._bus_map.items():
            cat_cmp_dict['bus'][bus.category_id] = bus

        return cat_cmp_dict

    @memoize_method
    def get_pe_neighbor(self, pe_id: int) -> int:
        pe = self.component_map[pe_id]
        relative_peid = pe.category_id % self.pes_per_pu
        if relative_peid + 1 == self.pes_per_pu:
            return self.cat_component_map['pe'][pe.category_id - self.pes_per_pu + 1].component_id
        else:
            return self.cat_component_map['pe'][pe.category_id + 1].component_id

    @memoize_method
    def get_pu_neighbor(self, pu_id: int) -> int:
        pu = self.component_map[pu_id]

        if pu.category_id + 1 == len(self.pu_map.keys()):
            return self.cat_component_map['pu'][0].component_id
        else:
            return self.cat_component_map['pu'][pu.category_id + 1].component_id

    @memoize_method
    def get_head_pe(self, pu_id: int) -> int:
        pu = self.component_map[pu_id]
        return self.cat_component_map['pe'][min(pu.pe_ids)].component_id

    @property
    def pu_map(self) -> Dict[str, PU]:
        return self._pu_map

    def get_category_id(self, component_id: int) -> int:
        return self.component_map[component_id].category_id

    @property
    def bus_map(self) -> Dict[str, Bus]:
        return self._bus_map

    @memoize_method
    def get_bus(self, bus_name: str) -> Bus:
        if bus_name not in self._bus_map:
            raise KeyError(f"{bus_name} not found in PE{self.component_id} map")
        return self._bus_map[bus_name]

    def create_initial_state(self) -> State:
        cycle = 0
        metadata = {}
        state_name = self.idle_state
        init_state = self.add_cycle_state(cycle, state_name, metadata)
        return init_state

    @memoize_method
    def get_namespace_pe(self, namespace_id: int) -> int:
        for pe_id, pe in self.cat_component_map['pe'].items():
            if namespace_id in pe.subcomponent_ids:
                return pe.component_id
        raise RuntimeError(f"No such PE with namespace id {namespace_id}")

    def add_namespace_data(self, cycle: int, pe_id: int, ns_name, edge: ScheduleEdge) -> int:

        pe: PE = self.component_map[pe_id]

        namespace = pe.get_namespace(ns_name)
        index = namespace.insert_data(cycle, edge.edge_id, edge.data_id)
        return index

    def find_all_empty_pes(self, cycle: int) -> List[int]:
        idle_pes = []
        for pe_id, pe in self.cat_component_map['pe'].items():
            if pe.is_idle(cycle):
                idle_pes.append(pe.component_id)
        return idle_pes

    def find_optimal_pe(self, cycle: int, ns_name: str) -> int:
        opt_pe = -1
        min_items = self.get_namespace_size(ns_name)
        for pe_id, pe in self.cat_component_map['pe'].items():
            namespace = pe.get_namespace(ns_name)
            if pe.is_idle(cycle) and namespace.item_count() < min_items:
                opt_pe = pe.component_id
                min_items = namespace.item_count()
        return opt_pe

    def find_optimal_exec_pe(self, cycle: int, ns_name: str) -> int:
        opt_pe = -1
        min_items = self.get_namespace_size(ns_name)
        for pe_id, pe in self.cat_component_map['pe'].items():
            namespace = pe.get_namespace(ns_name)

            if pe.is_idle(cycle) and namespace.item_count() < min_items and not pe.is_head_pe:

                opt_pe = pe.component_id
                min_items = namespace.item_count()
        return opt_pe

    def find_empty_pe(self, cycle: int) -> int:
        for pe_id, pe in self.cat_component_map['pe'].items():
            if pe.is_idle(cycle):
                return pe.component_id
        return -1

    def add_pe_instruction(self, cycle: int, sched, instruction: Instruction, pe_id: int, check_maps=True, instruction_fn="unset"):

        pe = self.component_map[pe_id]
        return self.update_comm_dict(sched, instruction, pe, cycle, check_maps=check_maps, instruction_fn=instruction_fn)


    def add_instruction_dest(self, cycle, instr, edge: ScheduleEdge, location: str, data_id: int, comp_id, index: int = -1):
        pe = self.component_map[instr.component_id]

        instr.add_dest(edge.edge_id, location, data_id, comp_id, index)
        if location == "PENB":
            # dst_pe = pe.category_id - 7 if (pe.category_id + 1) % 8 == 0 else pe.category_id + 1
            # dst_pe = pe.category_id - (self.pes_per_pu - 1) if (pe.category_id + 1) % self.pes_per_pu == 0 else pe.category_id + 1
            dst_pe = DEST_PE_ID_FN[location](pe, index, self.pes_per_pu, self.num_pus)
            assert type(dst_pe) == type(pe.category_id)
            key = (pe.category_id, dst_pe)
            self.update_write_map(key, cycle, instr.node_id, edge, index=-1)

        if location == "PEGB":
            # dst_pe = (pe.category_id // 8) * 8 + instr._dest_pos["PEGB"][0].index
            # dst_pe = (pe.category_id // self.num_pus) * self.num_pus + instr._dest_pos["PEGB"][0].index

            dst_pe = DEST_PE_ID_FN[location](pe, instr._dest_pos["PEGB"][0].index, self.pes_per_pu, self.num_pus)


            assert type(dst_pe) == type(pe.category_id)
            key = (pe.category_id, dst_pe)
            self.update_write_map(key, cycle, instr.node_id, edge, index=-1)


        if location == "PUNB":
            # curr_pu = (pe.category_id // 8)
            # dst_pe = 0 if curr_pu == 7 else (curr_pu + 1) * 8

            # curr_pu = (pe.category_id // self.num_pus)
            # dst_pe = 0 if curr_pu == (self.num_pus - 1) else (curr_pu + 1) * self.pes_per_pu
            dst_pe = DEST_PE_ID_FN[location](pe, index, self.pes_per_pu, self.num_pus)


            assert type(dst_pe) == type(pe.category_id)
            key = (pe.category_id, dst_pe)
            self.update_write_map(key, cycle, instr.node_id, edge, index=-1)


        if location == "PUGB":
            # dst_pe = 8 * instr._dest_pos["PUGB"][0].index
            # dst_pe = self.pes_per_pu * instr._dest_pos["PUGB"][0].index
            dst_pe = DEST_PE_ID_FN[location](pe, instr._dest_pos["PUGB"][0].index, self.pes_per_pu, self.num_pus)

            assert type(dst_pe) == type(pe.category_id)
            key = (pe.category_id, dst_pe)

            self.update_write_map(key, cycle, instr.node_id, edge, index=-1)

        if location in ["NI", "NS", "NW"]:
            dst_pe = DEST_PE_ID_FN[location](pe, index, self.pes_per_pu, self.num_pus)

            key = (pe.category_id, dst_pe)
            if edge.edge_id not in [e.eid for e in self.wcycle_edge_map[key]]:
                self.update_write_map(key, cycle, instr.node_id, edge, index=-1)

    def add_instruction(self, cycle: int, instruction: Instruction):
        pe = self.component_map[instruction.component_id]
        _ = pe.add_instruction(cycle, instruction)

    def replace_instruction_source(self, instr: Instruction, position: int, edge_id: int, location: str, data_id: int, comp_id, index: int = -1, optimizing=False):
        instr_pe = self.component_map[comp_id]

        new_pe_id = get_instr_cat_pe(instr.srcs[position], instr_pe, self.pes_per_pu, self.num_pus, loc_idx=(location, index))
        old_pe_id = get_instr_cat_pe(instr.srcs[position], instr_pe, self.pes_per_pu, self.num_pus)
        if new_pe_id == old_pe_id and not optimizing:
            return
        instr.replace_source(position, edge_id, location, data_id, comp_id, index)

    def update_read_cycle(self, key, pos, start_cycle, new_cycle, src_edge, instr):
        if start_cycle != new_cycle:
            ridx = [i for i, ri in enumerate(self.rcycle_edge_map[key]) if ri.eid == src_edge.edge_id][0]
            self.update_read_map(key, new_cycle, instr.srcs[pos].data_id, src_edge, index=ridx)
            start_cycle = new_cycle
        return start_cycle


    def check_instruction_order(self, sched, instr, pe, cycle, check_maps=True, instruction_fn="unset"):
        start_cycle = cycle
        instr_info = defaultdict(dict)
        equiv = []
        for i, src in enumerate(instr.srcs):
            src_pe = get_instr_cat_pe(src, pe, sched.pes_per_pu, sched.num_pus)
            key = (src_pe, pe.category_id)
            src_edge = sched.get_schedule_edge(src.source_id)
            r_index = self.update_read_map(key, cycle, src.data_id, src_edge, index=-1)
            maps_eq = map_rwr_equality(key, src_edge, self.rcycle_edge_map[key], self.wcycle_edge_map[key])
            instr_info[i]['src_pe'] = src_pe
            instr_info[i]['key'] = key
            instr_info[i]['edge'] = src_edge
            instr_info[i]['r_index'] = r_index
            instr_info[i]['maps_eq'] = maps_eq
            instr_info[i]['pass_instr'] = None
            instr_info[i]['pass_cycle'] = -1
            equiv.append(maps_eq)
        other = -1
        iters = 0

        while any(equiv):

            if equiv[0]:
                cycle, r_index, pass_instr, pass_cycle = self.check_instruction_source(sched, pe, instr, cycle,
                                                               instr_info[0]['r_index'], 0,
                                                               instr_info[0]['pass_instr'], instr_info[0]['pass_cycle'])
                instr_info[0]['r_index'] = r_index
                instr_info[0]['pass_instr'] = pass_instr
                instr_info[0]['pass_cycle'] = pass_cycle
                other = 1


            elif equiv[1]:
                cycle, r_index, pass_instr, pass_cycle = self.check_instruction_source(sched, pe, instr, cycle,
                                                               instr_info[1]['r_index'], 1,
                                                               instr_info[1]['pass_instr'], instr_info[1]['pass_cycle'])
                instr_info[1]['r_index'] = r_index
                instr_info[1]['pass_instr'] = pass_instr
                instr_info[1]['pass_cycle'] = pass_cycle
                other = 0

            if len(instr.srcs) > 1:
                assert other >= 0
                start_cycle = self.update_read_cycle(instr_info[other]['key'], other, start_cycle, cycle, instr_info[other]['edge'], instr)

            other = -1
            for i in range(len(instr.srcs)):
                equiv[i] = map_rwr_equality(instr_info[i]['key'], instr_info[i]['edge'], self.rcycle_edge_map[instr_info[i]['key']], self.wcycle_edge_map[instr_info[i]['key']])
                instr_info[i]['maps_eq'] = equiv[i]
            iters += 1

        for i in range(len(instr.srcs)):
            if instr_info[i]['pass_instr']:
                src_pe = self.cat_component_map['pe'][instr_info[i]['key'][0]]
                _ = src_pe.add_instruction(instr_info[i]['pass_cycle'], instr_info[i]['pass_instr'])

        return cycle

    def check_instruction_source(self, sched, pe, instr, cycle, r_index, pos, pass_instr, pass_cycle):

        src_pe = get_instr_cat_pe(instr.srcs[pos], pe, self.pes_per_pu, self.num_pus)
        src_edge = sched.get_schedule_edge(instr.srcs[pos].source_id)
        key = (src_pe, pe.category_id)
        maps_eq = True
        # pass_instr = None
        # pass_cycle = -1

        w_index = [i for i, wi in enumerate(self.wcycle_edge_map[key]) if wi.eid == src_edge.edge_id][0]
        while maps_eq:
            if r_index > w_index:
                cycle, pass_instr, pass_cycle = self.split_write(cycle, instr, instr.srcs[pos], key, src_edge, pass_instr=pass_instr, pass_cycle=pass_cycle)
                w_index = [i for i, wi in enumerate(self.wcycle_edge_map[key]) if wi.eid == src_edge.edge_id][0]
            else:
                cycle = self.split_read(key, instr.srcs[pos], src_edge, cycle)
                r_index = [i for i, ri in enumerate(self.rcycle_edge_map[key]) if ri.eid == src_edge.edge_id][0]

            maps_eq = map_rwr_equality(key, src_edge, self.rcycle_edge_map[key], self.wcycle_edge_map[key])

        return cycle, r_index, pass_instr, pass_cycle

    def update_comm_dict(self, sched, instr, pe, cycle, check_maps=True, instruction_fn="unset"):

        cycle = self.check_instruction_order(sched, instr, pe, cycle, check_maps, instruction_fn)

        self.add_instr_destinations(instr, pe, cycle, sched)
        return cycle


    def split_write(self, start_cycle, instr, src, key, edge, pass_instr=None, pass_cycle=-1):
        src_pe = self.cat_component_map['pe'][key[0]]
        dst_pe = self.cat_component_map['pe'][key[1]]

        wr_index = [i for i, wi in enumerate(self.wcycle_edge_map[key]) if wi.eid == edge.edge_id][0]
        wr_cycle = self.wcycle_edge_map[key][wr_index].cycle
        if not pass_instr:
            pass_instr, pass_cycle = self.create_pass_instr(key, src, start_cycle, edge)

        else:
            r_index = [i for i, ri in enumerate(self.rcycle_edge_map[key]) if ri.eid == edge.edge_id][0]
            pass_cycle = self.wcycle_edge_map[key][r_index].cycle
            while not src_pe.is_idle(pass_cycle):
                pass_cycle += 1
        dest_name = self.find_dest_loc(src_pe, dst_pe, pass_instr)
        dest_loc = pass_instr.get_dest(dest_name)

        assert src.data_id == edge.source_id

        new_wr_index = self.update_write_map(key, pass_cycle, src.data_id, edge, index=wr_index)


        comm_delay = CYCLE_DELAYS["PE"][dest_loc.location] + CYCLE_DELAYS[dest_loc.location]["PE"] + 1

        if pass_cycle >= self.rcycle_edge_map[key][new_wr_index].cycle - comm_delay:
            new_read_cycle = pass_cycle + comm_delay
            r_index = [i for i, ri in enumerate(self.rcycle_edge_map[key]) if ri.eid == edge.edge_id][0]
            while not dst_pe.is_idle(new_read_cycle):
                new_read_cycle += 1
            new_r_index = self.update_read_map(key, new_read_cycle, src.data_id, edge, index=r_index)

            start_cycle = new_read_cycle
        update_temp_edge_paths(edge, pass_cycle, start_cycle, wr_cycle, src_pe)

        return start_cycle, pass_instr, pass_cycle

    def create_pass_instr(self, key, src, start_cycle, edge):
        wr_index = [i for i, wi in enumerate(self.wcycle_edge_map[key]) if wi.eid == edge.edge_id][0]

        wr_cycle = self.wcycle_edge_map[key][wr_index].cycle
        src_pe = self.cat_component_map['pe'][key[0]]
        dst_pe = self.cat_component_map['pe'][key[1]]
        wr_instr = src_pe.get_instr(wr_cycle)
        ni = src_pe.get_namespace("NI")
        delay = start_cycle + CYCLE_DELAYS["PE"]["NAMESPACE"] + 1
        dest_name = self.find_dest_loc(src_pe, dst_pe, wr_instr)
        dest_loc = wr_instr.get_dest(dest_name)

        if not wr_instr.check_dest("NI") or not ni.is_data_present(delay, wr_instr.node_id):
            ni_idx = self.add_namespace_data(delay, src_pe.component_id, "NI", edge)
            wr_instr.replace_dest(dest_name,
                                  dest_loc.dest_id,
                                  "NI",
                                  wr_instr.node_id,
                                  ni.component_id,
                                  index=ni_idx)

        ni_idx = ni.find_data_index(src.data_id, delay)
        if wr_instr.check_dest(dest_name):
            wr_instr.replace_dest(dest_name,
                                  dest_loc.dest_id,
                                  "NI",
                                  wr_instr.node_id,
                                  ni.component_id,
                                  index=ni_idx)
        if edge.edge_id not in wr_instr.get_dest("NI").all_dests:
            wr_instr.get_dest("NI").add_edge(edge.edge_id)
        r_index = [i for i, ri in enumerate(self.rcycle_edge_map[key]) if ri.eid == edge.edge_id][0]

        r_cycle = self.wcycle_edge_map[key][r_index].cycle
        pass_cycle = max(delay, r_cycle)

        while not src_pe.is_idle(pass_cycle):
            pass_cycle += 1

        # NEED TO STORE THIS INSTRUCTION FOR FUTURE ITERATIONS
        pass_instr = Instruction(wr_instr.node_id, "pass")
        dest_info = (dest_loc.dest_id, dest_loc.location, dest_loc.data_id, dest_loc.comp_id, dest_loc.index)
        pass_instr.add_dest(*dest_info)
        pass_instr.add_source(edge.edge_id, "NI", src.data_id, src_pe.component_id, index=ni_idx)

        return pass_instr, pass_cycle

    def split_read(self, key, src, edge, cycle):

        w_index = [i for i, wi in enumerate(self.wcycle_edge_map[key]) if wi.eid == edge.edge_id][0]
        start_cycle = self.rcycle_edge_map[key][w_index].cycle

        dest_pe = self.cat_component_map['pe'][key[1]]
        new_cycle = start_cycle
        while not dest_pe.is_idle(new_cycle):
            new_cycle += 1

        r_index = [i for i, ri in enumerate(self.rcycle_edge_map[key]) if ri.eid == edge.edge_id][0]
        new_r_index = self.update_read_map(key, new_cycle, src.data_id, edge, index=r_index)
        path_index = edge.path.index(dest_pe.component_id)

        edge.add_path_overhead(path_index + 1, new_cycle - cycle)

        #if self.pe_wr_map[key] == self.pe_r_map[key]:
            #assert all([self.wcycle_data_map[key][i][0] < self.rcycle_data_map[key][i][0] for i in
                    #range(len(self.pe_r_map[key]))])

        return new_cycle


    def add_instr_destinations(self, instr, pe, cycle, sched):

        pe.add_instruction(cycle, instr)
        keys = []
        if instr.check_dest("PENB"):
            # dst_pe = pe.category_id - 7 if (pe.category_id + 1) % 8 == 0 else pe.category_id + 1
            # dst_pe = pe.category_id - 7 if (pe.category_id + 1) % 8 == 0 else pe.category_id + 1
            dst_pe = DEST_PE_ID_FN["PENB"](pe, -1, self.pes_per_pu, self.num_pus)

            assert type(dst_pe) == type(pe.category_id)
            key = (pe.category_id, dst_pe)
            dst_edge = sched.get_schedule_edge(instr.get_dest("PENB").dest_id)
            self.update_write_map(key, cycle, instr.node_id, dst_edge, index=-1)
            keys.append(key)

        if instr.check_dest("PEGB"):
            # dst_pe = (pe.category_id // 8) * 8 + instr._dest_pos["PEGB"][0].index
            dst_pe = DEST_PE_ID_FN["PEGB"](pe, instr._dest_pos["PEGB"][0].index, self.pes_per_pu, self.num_pus)

            assert type(dst_pe) == type(pe.category_id)
            key = (pe.category_id, dst_pe)
            dst_edge = sched.get_schedule_edge(instr.get_dest("PEGB").dest_id)
            self.update_write_map(key, cycle, instr.node_id, dst_edge, index=-1)
            keys.append(key)

        if instr.check_dest("PUNB"):
            # curr_pu = (pe.category_id // 8)
            # dst_pe = 0 if curr_pu == 7 else (curr_pu + 1) * 8
            dst_pe = DEST_PE_ID_FN["PUNB"](pe, -1, self.pes_per_pu, self.num_pus)

            assert type(dst_pe) == type(pe.category_id)
            key = (pe.category_id, dst_pe)
            dst_edge = sched.get_schedule_edge(instr.get_dest("PUNB").dest_id)
            self.update_write_map(key, cycle, instr.node_id, dst_edge, index=-1)
            keys.append(key)


        if instr.check_dest("PUGB"):
            # dst_pe = 8 * instr._dest_pos["PUGB"][0].index
            dst_pe = DEST_PE_ID_FN["PUGB"](pe, instr._dest_pos["PUGB"][0].index, self.pes_per_pu, self.num_pus)

            assert type(dst_pe) == type(pe.category_id)
            key = (pe.category_id, dst_pe)
            dst_edge = sched.get_schedule_edge(instr.get_dest("PUGB").dest_id)
            self.update_write_map(key, cycle, instr.node_id, dst_edge, index=-1)
            keys.append(key)



        if not all([instr.check_dest(d) for d in ["NI", "NS", "NW"]]) and any(
                [instr._dest_pos[d][0].index >= 0 for d in ["NI", "NS"]]):
            # dst_pe = pe.category_id
            dst_pe = DEST_PE_ID_FN["NI"](pe, -1, self.pes_per_pu, self.num_pus)

            key = (pe.category_id, dst_pe)

            dst_loc = "NI" if instr._dest_pos["NI"][0].index >= 0 else "NW"
            dst_edge = sched.get_schedule_edge(instr.get_dest(dst_loc).dest_id)

            if dst_edge.edge_id not in [e.eid for e in self.wcycle_edge_map[key]]:
                self.update_write_map(key, cycle, instr.node_id, dst_edge, index=-1)
                keys.append(key)

    def find_dest_loc(self, source_pe, dest_pe, instr):
        assert source_pe != dest_pe

        if source_pe.category_id + 1 == dest_pe.category_id or \
                (dest_pe.is_head_pe and (source_pe.category_id - self.pes_per_pu + 1) == dest_pe.category_id):
            assert instr.check_dest("PENB")
            return "PENB"
        elif source_pe.component_subtype == dest_pe.component_subtype:
            assert instr.check_dest("PEGB")
            return "PEGB"
        elif self.component_map[int(source_pe.component_subtype)].category_id + 1 == self.component_map[int(dest_pe.component_subtype)].category_id or (self.component_map[int(dest_pe.component_subtype)].category_id == 0 and (self.component_map[int(source_pe.component_subtype)].category_id + 1 == self.num_pus)):
            assert source_pe.is_head_pe and dest_pe.is_head_pe
            assert instr.check_dest("PUNB")
            return "PUNB"
        else:
            assert source_pe.is_head_pe and dest_pe.is_head_pe
            assert instr.check_dest("PUGB")
            return "PUGB"

    def update_write_map(self, key, cycle, data_id, edge: ScheduleEdge, index=-1):

        if index >= 0:
            self.wcycle_data_map[key].pop(index)
            self.wcycle_edge_map[key].pop(index)
            self.pe_wr_map[key].pop(index)
        wcycles = list(map(self.cycle_from_map, self.wcycle_edge_map[key]))
        new_index = bisect.bisect_right(wcycles, cycle)
        wr_info = WriteInfo(src=edge.source_id, dst=edge.dest_id, eid=edge.edge_id, cycle=cycle)
        self.wcycle_edge_map[key].insert(new_index, wr_info)
        self.wcycle_data_map[key].insert(new_index, (cycle, data_id))
        self.pe_wr_map[key].insert(new_index, data_id)

        assert len(set([wi.eid for wi in self.wcycle_edge_map[key]])) == len(self.wcycle_data_map[key])
        assert len(self.pe_wr_map[key]) == len(self.wcycle_edge_map[key])
        assert edge.source_id == data_id
        tnew_index = [i for i, wi in enumerate(self.wcycle_edge_map[key]) if wi.eid == edge.edge_id][0]
        assert tnew_index == new_index
        return new_index

    def update_read_map(self, key, cycle, data_id, edge: ScheduleEdge, index=-1):
        if index >= 0:

            self.rcycle_data_map[key].pop(index)
            self.rcycle_edge_map[key].pop(index)
            self.pe_r_map[key].pop(index)

        rcycles = list(map(self.cycle_from_map, self.rcycle_edge_map[key]))
        new_index = bisect.bisect_right(rcycles, cycle)

        r_info = WriteInfo(src=edge.source_id, dst=edge.dest_id, eid=edge.edge_id, cycle=cycle)
        self.rcycle_edge_map[key].insert(new_index, r_info)
        self.rcycle_data_map[key].insert(new_index, (cycle, data_id))
        self.pe_r_map[key].insert(new_index, data_id)
        assert len(set([wi.eid for wi in self.rcycle_edge_map[key]])) == len(self.rcycle_data_map[key])

        assert len(self.pe_r_map[key]) == len(self.rcycle_edge_map[key])
        assert edge.source_id == data_id
        tnew_index = [i for i, ri in enumerate(self.rcycle_edge_map[key]) if ri.eid == edge.edge_id][0]
        assert tnew_index == new_index
        return new_index


    def print_update_info(self, when, key, instr, pos, start_cycle, new_cycle, r_index, w_index, instruction_fn, edge):
        read_cycles = [e.cycle for e in self.rcycle_edge_map[key]]
        write_cycles = [e.cycle for e in self.wcycle_edge_map[key]]

        read_edges = [e.eid for e in self.rcycle_edge_map[key]]
        write_edges = [e.eid for e in self.wcycle_edge_map[key]]
        print(f"{when} - {key} - {instruction_fn}\n"
              f"Maps {pos}\n"
              f"Instruction id: {instr.node_id}\n"
              f"Data id: {instr.srcs[pos].data_id}\n"
              f"edge id: {edge.edge_id}\n"
            f"r1 index: {r_index}\n"
            f"w1 index: {w_index}\n"
              f"Start cycle: {start_cycle}\n"
              f"Start: {new_cycle}\n"
              f"Instr: {instr}\n"
              f"R: {self.pe_r_map[key]}\n"
              f"W: {self.pe_wr_map[key]}\n"
              f"Read edge ids: {read_edges}\n"
              f"Write edge ids: {write_edges}\n"
              f"Read cycles: {read_cycles}\n"
              f"Write  cycles: {write_cycles}\n\n"
              f"Reads: {self.rcycle_data_map[key]}\n"
              f"Writes: {self.wcycle_data_map[key]}\n\n")





