import json
from typing import List, Callable, Dict, List, Any
from collections import defaultdict

from codelets.adl.operation import OperandTemplate, Loop, Compute, Transfer, Configure, Operation
from codelets.adl.graph import ArchitectureNode
from codelets.codelet_impl import Codelet
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
import polymath as pm
from .relocation_table import RelocationTable

EMIT_OPTIONS = ["decimal", "operations", "string_final", "string_placeholders", "binary"]

@dataclass
class CompilationStage:
    name: str
    level: int
    compilation_fn: Callable
    dependencies: List[str]
    fn_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # TODO: Check function signature
        pass

    def run(self, *args):
        return self.compilation_fn(*args, **self.fn_kwargs)

class CodeletProgram(object):

    def __init__(self, graph: pm.Node, hag: ArchitectureNode, program_mode: str="inference"):
        self._name = graph.name
        self._hag = hag
        self._graph = graph
        self._codelets = []
        self._relocatables = RelocationTable()
        self._compilation_pipeline = defaultdict(list)
        self._preproc_steps = defaultdict(list)
        self._program_mode = program_mode
        self._side_effect_params = {'program': {}, 'codelet': {}, 'op': {}}

    @property
    def name(self) -> str:
        return self._name

    @property
    def hag(self) -> ArchitectureNode:
        return self._hag

    @property
    def program_mode(self):
        return self._program_mode

    @property
    def graph(self):
        return self._graph

    @property
    def codelets(self) -> List[Codelet]:
        return self._codelets

    @property
    def relocatables(self) -> RelocationTable:
        return self._relocatables

    @property
    def compilation_pipeline(self) -> Dict[int, List[CompilationStage]]:
        return self._compilation_pipeline

    @property
    def preproc_steps(self) -> Dict[int, List]:
        return self._preproc_steps

    @property
    def side_effect_params(self):
        return self._side_effect_params

    def add_side_effect_param(self, name, scope, init_val):
        if scope == 'program':
            self._side_effect_params[scope][name] = init_val
        elif scope == 'codelet':
            for c in self.codelets:
                self._side_effect_params[scope][c.cdlt_uid][name] = init_val
        else:
            raise RuntimeError(f"Currently no other scopes are supported for side effects other than 'program'"
                               f" and 'codelet'. Request side effect: {scope}")


    def update_side_effect_param(self, name, scope, value, codelet_id=None, operation_id=None):
        if scope == 'program':
            self._side_effect_params[scope][name] = value
        elif scope == 'codelet':
            assert codelet_id in self.side_effect_params['codelet']
            self._side_effect_params[scope][codelet_id][name] = value

    def add_codelet(self, cdlt: Codelet):
        self._codelets.append(cdlt)

    def get_codelet(self, cdlt_id: int):
        for cdlt in self.codelets:
            if cdlt.instance_id == cdlt_id:
                return cdlt
        raise KeyError(f"Unable to get codelet with id {cdlt_id} in codelet list")


    def save(self, output_path=None, save_format="json"):
        if output_path:
            if output_path[-1] != "/":
                output_path = output_path + "/"
        else:
            output_path = str(Path.cwd()) + "/"

        if save_format == "json":
            full_path = f"{output_path}{self.name}.json"
            self.save_json(full_path)
        elif save_format == "text":
            full_path = f"{output_path}{self.name}.txt"
            self.save_text(full_path)
        else:
            raise ValueError(f"Invalid file output: {save_format}")

    def save_binary(self, full_path):
        raise NotImplementedError

    def get_tiling_dims(self, inputs: List[OperandTemplate], outputs: List[OperandTemplate]):
        assert all([i.is_instantiated() for i in inputs])
        assert all([o.is_instantiated() for o in outputs])

    def add_compilation_step(self, name: str,
                             compilation_fn: Callable,
                             level=0,
                             dependencies=None,
                             stage_kwargs=None,
                             insert_idx=-1,
                             preproc=False
                             ):
        if not callable(compilation_fn):
            raise TypeError(f"Compilation step must be a callable function:\n"
                            f"Name: {name}\n"
                            f"Compilation arg: {compilation_fn}, Type: {type(compilation_fn)}")
        elif name in self.compilation_pipeline[level]:
            raise KeyError(f"Compilation for compilation stage already exists:\n"
                           f"Name: {name}")
        stage_kwargs = stage_kwargs or {}
        dependencies = dependencies or []
        level_names = [comp_stage.name for comp_stage in self.compilation_pipeline[level]]
        for d in dependencies:
            assert d in level_names

        fn_obj = CompilationStage(name, level, compilation_fn, dependencies, stage_kwargs)
        if preproc:
            if insert_idx >= 0:
                self._preproc_steps[level].insert(insert_idx, fn_obj)
            else:
                self._preproc_steps[level].append(fn_obj)
        else:
            if insert_idx >= 0:
                self._compilation_pipeline[level].insert(insert_idx, fn_obj)
            else:
                self._compilation_pipeline[level].append(fn_obj)

    INSTR_FN_TEMPLATE = """def param_fn{FN_ID}(hag, op, cdlt, relocation_table, program, fixed_val=None): return {FN_BODY}"""

    def set_instruction_templates(self, cdlt: Codelet):
        for o in cdlt.ops:

            template = self.hag.get_operation_template(o)
            template_copy = [ft.template_copy() for ft in template]

            o.set_template(template_copy)

    def instantiate_instructions(self, cdlt: Codelet, fixed_val=None):
        # TODO: Replace this with evaluate
        cdlt._num_instr = 0
        for o in cdlt.ops:
            args = (self, self.hag, o.global_op_id, cdlt.instance_id)
            for flex_temp in o.instructions:
                flex_temp.set_instruction_length(*args)
                cdlt._num_instr += flex_temp.num_instructions

        self.relocatables.add_instr_relocation(cdlt)

        for o in cdlt.ops:
            args = (self, self.hag, o.global_op_id, cdlt.instance_id)

            for ft in o.instructions:
                ft.evaluate(*args)

    def instantiate_codelet(self, node):
        cdlt = self.hag.get_codelet_template(node.op_name)
        self.add_codelet(cdlt)
        return cdlt

    def emit(self, output_type):
        codelet_strings = []
        for c in self.codelets:
            codelet_strings.append(c.emit(output_type))
        if output_type not in ["json", "json_no_ops"]:
            return "\n".join(codelet_strings)
        else:
            return {"mode": self.program_mode, "program": codelet_strings}


    def instantiate_instructions_templates(self, node, cdlt):
        self.set_instruction_templates(cdlt)
        self.relocatables.add_data_relocation(node)
        self.instantiate_instructions(cdlt)


    # TODO: Fix these
    def save_json(self, full_path):
        json_blob = []
        with open(full_path, 'w') as outfile:
            json.dump(json_blob, outfile, indent=4)

    def save_text(self, full_path):
        instructions = []
        instructions = "\n".join(instructions)
        with open(full_path, 'w') as outfile:
            outfile.write(instructions)

    def sequence_nodes(self, sequence_algorithm):
        # TODO: Add support for different sequencing algos
        node_list = []
        all_ops = []
        if sequence_algorithm == "default":
            for name, node in self.graph.nodes.items():
                if not isinstance(node, (pm.write, pm.placeholder)) and node.op_name not in all_ops:
                    all_ops.append(node.op_name)
                if self.hag.has_codelet(node.op_name):
                    node_list.append(node)
        else:
            raise RuntimeError(f"{sequence_algorithm} is not a valid sequencing algorithm")
        return node_list

    def compile(self, sequence_algorithm="default"):
        node_sequence = self.sequence_nodes(sequence_algorithm)

        # This function performs breadth-first compilation, with coarsest abstractions first:
        # 1. Generate codelets from nodes
        # 2. Generate operands/operations within codelets
        # 3. Generate instruction templates within operations
        codelets = {}

        for n in node_sequence:
            cdlt = self.instantiate_codelet(n)
            assert n.name not in codelets
            codelets[n.name] = cdlt

        for level, fns in self.preproc_steps.items():
            for n in node_sequence:
                cdlt = codelets[n.name]
                for fn in fns:
                    cdlt = fn.run(self, n, cdlt)

                assert n.name in codelets and codelets[n.name].instance_id == cdlt.instance_id
                codelets[n.name] = cdlt

        for n in node_sequence:
            cdlt = codelets[n.name]
            cdlt.instantiate_operations(n, self.hag)
            # TODO: Check if certain optimizations are necessary
            codelets[n.name] = cdlt

        for level, fns in self.compilation_pipeline.items():
            for n in node_sequence:
                cdlt = codelets[n.name]
                for fn in fns:
                    cdlt = fn.run(self, n, cdlt)
                codelets[n.name] = cdlt

        for n in node_sequence:
            self.instantiate_instructions_templates(n, codelets[n.name])


def generate_possible_tilings(shape_dict, memory_paths):
    possible_tilings = {}
    for k, v in shape_dict.items():
        tile_permutations = []

def tiling_constraint(shapes, node_capacities, tile_sizes):

    for i, t in enumerate(tile_sizes):
        data_size = np.prod([t[s] for s in shapes])
        if data_size >= node_capacities[i]:
            return False
    return True
