from typing import Tuple, Dict
from codelets.adl.graph import ArchitectureNode
from .program import CodeletProgram
from .compilation_stages import tile, hoist
import polymath as pm

TileConstraint = Dict[Tuple[str, str], Tuple[int, int]]

def initialize_program(program_graph, hag: ArchitectureNode):
    program = CodeletProgram(program_graph, hag)
    return program

