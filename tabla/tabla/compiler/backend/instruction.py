from typing import List, Dict
import numpy as np
from . import OP_SELECT_WIDTH, OP_WIDTH, OP_SELECT_BIN, OP_CODE_BIN
class DataLocation(object):

    local_namespaces = ['NW', 'NI', 'NG', 'NM', 'ND']
    bus_types = ['PENB', 'PEGB', 'PUNB', 'PUGB']

    def __init__(self, data_id=None, location=None, comp_id=-1, index=-1):
        self.data_id = data_id
        self.location = location
        self.index = index
        self.comp_id = comp_id
        self.cycles_available = []

    def __hash__(self):
        return hash((self.location, self.data_id, self.index, self.comp_id))

    def update_loc(self, comp_id, data_id=None, location=None, index=-1):
        self.comp_id = comp_id
        self.data_id = data_id
        self.location = location
        self.index = index

    def __str__(self):
        return f"{self.location}{self.index}" if self.index >= 0 else f"{self.location}"

    def is_local(self) -> bool:
        if self.location in self.local_namespaces:
            return True
        else:
            return False


class Source(DataLocation):

    def __init__(self, source_id: int, **kwargs):
        super(Source, self).__init__(**kwargs)
        self.source_id = source_id
        self.source_type = 'ZERO'
        self.source_index = 0

    def verilog_bin(self, index_bits):
        if self.location in ["PEGB", "PENB"]:
            return f"{np.binary_repr(OP_SELECT_BIN[self.location], width=OP_SELECT_WIDTH)}" \
                f"{np.binary_repr(self.index, width=index_bits-1)}0"
        elif self.location in ["PUGB", "PUNB"]:
            return f"{np.binary_repr(OP_SELECT_BIN[self.location], width=OP_SELECT_WIDTH)}" \
                f"{np.binary_repr(self.index, width=index_bits-1)}1"
        else:
            return f"{np.binary_repr(OP_SELECT_BIN[self.location], width=OP_SELECT_WIDTH)}" \
                f"{np.binary_repr(self.index, width=index_bits)}"

    def set_type(self, src_type: str, idx: int):
        self.source_type = src_type
        self.source_index = idx


class Dest(DataLocation):
    local_namespaces = ['NW', 'NI', 'NG', 'NM', 'ND']
    namespaces = ['NW', 'NG', 'NM', 'ND']
    bus_types = ['PENB', 'PEGB', 'PUNB', 'PUGB']

    def __init__(self, dest_id: int, **kwargs):
        super(Dest, self).__init__(**kwargs)
        self.dest_id = dest_id
        self.all_dests = []
        if dest_id >= 0:
            self.all_dests.append(dest_id)
        self.dest_bus = False
        self.dest_ni = False
        self.dest_ns = False

    def to_verilog_bin(self, index_bits):
        if self.location == "PEGB":

            return f"{np.binary_repr(1, width=1)}" \
                f"{np.binary_repr(self.index, width=index_bits-1)}"
        elif self.location == "PUGB":

            return f"{np.binary_repr(1, width=1)}" \
                f"{np.binary_repr(self.index, width=index_bits-1)}"
        elif self.location in ["PENB", "PUNB"]:
            return f"{np.binary_repr(1, width=1)}"
        elif self.location == "NI":
            return f"{np.binary_repr(1, width=1)}" \
                f"{np.binary_repr(self.index, width=index_bits)}"
        elif self.location == "NS" or self.location in self.local_namespaces:

            return f"{np.binary_repr(1, width=1)}" \
                    f"{np.binary_repr(self.index, width=index_bits)}"
        else:
            raise RuntimeError(f"Invalid location {self.location} for destination.")


    def update_dest(self, dest_id: int, **kwargs):
        self.dest_id = dest_id
        self.all_dests.append(dest_id)
        self.update_loc(**kwargs)

    def add_edge(self, edge_id: int):
        self.all_dests.append(edge_id)

    def set_type(self, location: str, index: int = -1):
        if location in self.bus_types:
            self.dest_bus = True
            self.bus_index = index
            self.location = location
        elif location == 'NI':
            self.dest_ni = True
            self.ni_index = index
            self.location = location
        elif location in self.local_namespaces:
            self.dest_ns = True
            self.ns_index = index
            self.location = location
        else:
            raise RuntimeError(f"Invalid location {location} for destination.")


class Instruction(object):
    src_count = 2
    dst_count = 6
    local_namespaces = ['NW', 'NI', 'NG', 'NM', 'ND']
    namespaces = ['NW', 'NG', 'NM', 'ND']
    bus_types = ['PENB', 'PEGB', 'PUNB', 'PUGB']
    def __init__(self, node_id: int, op_name: str, cycle_insert=None):
        self.node_id = node_id
        self._srcs = []
        self._dests = []
        self._dest_pos = {"NI": [Dest(-1, data_id=-1)],
                          "NS": [Dest(-1, data_id=-1)],
                          "PUNB": [Dest(-1, data_id=-1)],
                          "PENB": [Dest(-1, data_id=-1)],
                          "PEGB": [Dest(-1, data_id=-1)],
                          "PUGB": [Dest(-1, data_id=-1)],
                          "ALU": [Dest(-1, data_id=-1)]}
        self.cycle_insert = cycle_insert
        self.dest_dict = {}
        self.op_name = op_name
        self._comp_id = -1

    def __str__(self):
        return f"{self.src_str()}, {self.op_name}, {self.dest_str()}"

    def __hash__(self):
        return hash((tuple(self.srcs), self.node_id, self.op_name, tuple(self.dests)))

    @property
    def dests(self) -> List[Dest]:
        set_dest = []
        for key, dest_list in self._dest_pos.items():
            assert len(dest_list) == 1
            if dest_list[0].data_id >= 0:
                set_dest.append(dest_list[0])
        return set_dest

    def remove_dest(self, location):
        self._dest_pos[location][0] = Dest(-1, data_id=-1)

    @property
    def srcs(self) -> List[Source]:
        return self._srcs

    @property
    def component_id(self):
        return self._comp_id

    def to_verilog_bin(self, config: Dict[str, int]) -> str:
        ni_index_bits = config["ni_index_bits"]
        ns_index_bits = config["ns_index_bits"]
        op_index_bits = config["op_index_bits"]
        pe_bus_width = config["pe_bus_width"]
        pu_bus_width = config["pu_bus_width"]
        op_code = np.binary_repr(OP_CODE_BIN[self.op_name], width=OP_WIDTH)
        if len(self.srcs) == 0:
            raise RuntimeError(f"Zero sources for {self.node_id}")
        op1 = self.srcs[0].verilog_bin(op_index_bits)

        if len(self.srcs) == 1:
            op2 = f"{np.binary_repr(0, width=OP_SELECT_WIDTH)}{np.binary_repr(0, width=op_index_bits)}"
        else:
            op2 = self.srcs[1].verilog_bin(op_index_bits)
        if len(self.srcs) > 2:
            raise ValueError(f"Whoops, this has three ops for some reason {self}")

        ni_bin = self.get_ni_bin(ni_index_bits)
        ns_bin = self.get_ns_bin(ns_index_bits)
        neighbor_bin = self.get_neighbor_bin()
        pegb_bin = self.get_pegb_bin(pe_bus_width)
        pugb_bin = self.get_pugb_bin(pu_bus_width)
        instr_bin = f"{op_code}{op1}{op2}{ni_bin}{ns_bin}{neighbor_bin}{pegb_bin}{pugb_bin}"

        if len(instr_bin) != config["instr_len"]:
            print(f"Instr: {self}\n"
                  f"Length of sources: {len(self.srcs)}\n"
                  f"Config Instruction: {config['instr_len']}\n"
                  f"Binary length: {len(instr_bin)}\n"
                  f"Op index bits: {len(op_code)}\n"
                  f"Op1: {len(op1)}\n"
                  f"Op2: {len(op2)}\n"
                  f"NI Length: {len(ni_bin)}\n"
                  f"NS length: {len(ns_bin)}\n"
                  f"Nieghbor: {len(neighbor_bin)}\n"
                  f"PEGB: {len(pegb_bin)}\n"
                  f"PUGB: {len(pugb_bin)}\n"
                  f"Config: {config}\n")
        if len(instr_bin) != config["instr_len"]:
            raise RuntimeError(f"Instruction binary length uneqal to computed instruction length:\n"
                               f"INstruction length: {len(instr_bin)}\n"
                               f"Config length: {config['instr_len']}\n")
        # assert len(instr_bin) == config["instr_len"]
        return instr_bin

    def get_dest(self, key):
        assert self.check_dest(key)
        return self._dest_pos[key][0]

    def get_dest_ns_index(self, ns):
        assert self.check_dest(ns)
        return self._dest_pos[ns][0].index

    def get_ni_bin(self, ni_index_bits) -> str:
        if self.check_dest("NI"):
            return self._dest_pos["NI"][0].to_verilog_bin(ni_index_bits)
        else:
            return f"{np.binary_repr(0, width=ni_index_bits+1)}"

    def get_ns_bin(self, ns_index_bits) -> str:
        if self.check_dest("NS"):
            return self._dest_pos["NS"][0].to_verilog_bin(ns_index_bits)
        else:
            return f"{np.binary_repr(0, width=ns_index_bits+1)}"

    def get_pegb_bin(self, pe_bits):
        if self.check_dest("PEGB"):
            return self._dest_pos["PEGB"][0].to_verilog_bin(pe_bits)
        else:
            return str(np.binary_repr(0, width=pe_bits))

    def get_pugb_bin(self, pu_bits):
        if self.check_dest("PUGB"):
            return self._dest_pos["PUGB"][0].to_verilog_bin(pu_bits)
        else:
            return str(np.binary_repr(0, width=pu_bits))

    def get_neighbor_bin(self):
        if self.check_dest("PENB"):
            penb = self._dest_pos["PENB"][0].to_verilog_bin(0)
        else:
            penb = np.binary_repr(0,width=1)

        if self.check_dest("PUNB"):
            punb = self._dest_pos["PUNB"][0].to_verilog_bin(0)
        else:
            punb = np.binary_repr(0,width=1)


        return f"{punb}{penb}"

    def replace_source(self, position: int, edge_id: int, location: str, data_id: int, comp_id, index: int = -1):
        kwargs = {'data_id' : data_id, 'location': location, 'index': index, 'comp_id': comp_id}
        new_source = Source(edge_id, **kwargs)
        self.srcs[position] = new_source

    def replace_dest(self, o_key: str, edge_id: int, location: str, data_id: int, comp_id, index: int):
        null_dest = Dest(-1, data_id=-1)
        idx = self.dests.index(self._dest_pos[o_key][0])
        self.dests.pop(idx)
        self.add_dest(edge_id, location, data_id, comp_id, index)

        if self._dest_pos[o_key][0].dest_id >= 0:
            self._dest_pos[o_key][-1] = null_dest
        else:
            self._dest_pos[o_key][0] = null_dest


    def add_source(self, edge_id: int, location: str, data_id: int, comp_id, index: int = -1):

        kwargs = {'data_id' : data_id, 'location': location, 'index': index, 'comp_id': comp_id}

        new_source = Source(edge_id, **kwargs)
        self.srcs.append(new_source)

    def add_dest(self, edge_id: int, location: str, data_id: int, comp_id, index: int = -1):

        kwargs = {'data_id': data_id, 'location': location, 'index': index, 'comp_id': comp_id}
        new_dest = Dest(edge_id, **kwargs)

        if location == "NI":
            key = "NI"
        elif location in self.namespaces or location == "NS":
            key = "NS"
        elif location in self.bus_types:
            key = location
        elif location == "ALU":
            key = "ALU"
        else:
            raise ValueError(f"Invalid location destination {location} for "
                             f"node {self.node_id} with {data_id}")

        if not self._dest_pos[key][0].dest_id >= 0:
        #     assert self._dest_pos[key][0] == new_dest
        # else:
            assert len(self._dest_pos[key]) == 1
            self._dest_pos[key][0].update_dest(edge_id, **kwargs)
        return self._dest_pos[key][0]

    def check_dest(self, location: str) -> bool:

        if location == "NI":
            return self._dest_pos["NI"][0].dest_id > 0
        elif location in self.namespaces or location == "NS":
            return self._dest_pos["NS"][0].dest_id > 0
        elif location in self.bus_types or location == "ALU":
            return self._dest_pos[location][0].dest_id > 0
        else:
            raise ValueError(f"Invalid location destination {location} for "
                             f"node {self.node_id}")

    def get_dests(self) -> List[Dest]:
        dest_list = []
        for key, item in self._dest_pos.items():
            for dest in item:
                dest_list.append(dest)
        return dest_list

    def get_sources(self) -> List[Source]:
        return self.srcs

    @property
    def dest_count(self) -> int:
        count = 0
        for _, d in self._dest_pos.items():
            if d[0].dest_id >= 0:
                if len(d) > 1:
                    count += len(d)
                else:
                    count += 1
        return count

    def set_component_id(self, comp_id: int):
        self._comp_id = comp_id

    def dest_str(self) -> str:
        return ', '.join([str(dest) for dest in self.dests])

    def src_str(self) -> str:
        return ', '.join([str(src) for src in self.srcs])

    def check_source(self, key, index=-1):
        for s in self.srcs:
            if s.location == key and s.index == index:
                return True
        return False

    def __repr__(self):
        return f"{self.node_id}: {str(self)}"