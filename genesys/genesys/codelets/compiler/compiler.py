from typing import Tuple, Dict
from codelets.adl.graph import ArchitectureNode
from .program import CodeletProgram

TileConstraint = Dict[Tuple[str, str], Tuple[int, int]]

def initialize_program(program_graph, hag: ArchitectureNode, metadata=None, mode="inference"):

    program = CodeletProgram(program_graph, hag, metadata=metadata, program_mode=mode)
    return program
