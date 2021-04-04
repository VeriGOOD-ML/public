from __future__ import print_function, division, absolute_import

CYCLE_DELAYS = {
    "PE": {"PEGB": 2, "PENB": 1, "NAMESPACE": 1, "PUGB": 2, "PUNB": 1, "PE": 0},
    "PEGB": {"PE": 2},
    "PENB": {"PE": 0},
    "PUGB": {"PE": 2},
    "PUNB": {"PE": 0},
    "ALU" : {"PE": 1},
    "NAMESPACE": {"PE": 0}
}

NAMESPACES = ["NW", "NI", "NG", "NM", "ND"]
BUS_NAMES = ["PEGB", "PENB", "PUGB", "PUNB"]
NEIGHBORS = ["PENB", "PUNB"]

OP_SELECT_WIDTH = 3
OP_WIDTH = 5
MEM_INTERFACE_WIDTH = 1
NS_WEIGHT_WIDTH = 2
BUS_WIDTH = 3
OP_SELECT_BIN = {
    "ZERO": 0,
    "ALU": 1,
    "NW": 2,
    "ND": 3,
    "NG": 4,
    "NM": 4,
    "NI": 5,
    "PENB": 6,
    "PUNB": 6,
    "PEGB": 7,
    "PUGB": 7
}
OP_CODE_BIN = {
    "+": 1,
    "-": 2,
    "*": 3,
    "/": 4,
    "<": 5,
    "<=": 6,
    ">": 7,
    ">=": 8,
    "==": 9,
    "!=": 10,
    "sigmoid": 16,
    "gaussian": 17,
    "sqrt": 18,
    "sigmoid_symmetric": 19,
    "log": 20,
    "square": 21,
    "pass": 24,
}
from .schedule_utils import determine_edge_path, get_cost_type, get_edge_dest,\
    cycle_cost, compute_default_cost, find_exec_cycle, get_ordered_cycle, \
    ordered_cycle_val, create_temp_instr, create_inter_pe_instr, create_comm_instruction
from .state import State
from .instruction import Source, Dest, DataLocation, Instruction
from .schedule_objects import ScheduleEdge, ScheduleNode
from .component import Component
from .bus import Bus, BusItem
from .namespace_item import NamespaceItem
from .namespace import Namespace
from .pe import PE
from .pu import PU
from .tabla_template import TablaTemplate
from .instruction_optimization import optimize_instructions
from .schedule import Schedule
from .schedule_validation import validate_graph
from .instruction_gen import generate_pe_instructions
from .template_codelets import VectorMul


