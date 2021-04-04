from typing import List, Union
from dataclasses import dataclass, field
from .instruction import Instruction
from codelets.adl.flex_param import FlexParam
from .compiler_side_effect import SideEffect


@dataclass
class FlexTemplate:
    base_instructions: List[Instruction]
    instructions: List[Instruction] = field(default_factory=list)
    conditional: FlexParam = field(default=None)
    side_effects: List[FlexParam] = field(default_factory=list)
    iter_args: List[str] = field(default_factory=list)
    iterables: List[FlexParam] = field(default_factory=list)
    num_instructions: int = field(init=False, default=1)

    def __post_init__(self):
        if isinstance(self.base_instructions, Instruction):
            self.base_instructions = [self.base_instructions]

    def add_base_instruction(self, instruction):
        if isinstance(instruction, FlexTemplate):
            assert len(instruction.base_instructions) == 1
            if len(instruction.iterables) > 0:
                assert len(self.iterables) == len(instruction.iterables)
                for i, iterable in enumerate(self.iterables):
                    new_iter = instruction.iterables[i]
                    new_arg = instruction.iter_args[i]
                    assert new_arg == self.iter_args[i]
                    self.iterables[i].validate_equal(new_iter)

                if instruction.conditional is not None:
                    assert self.conditional is not None
                    self.conditional.validate_equal(instruction.conditional)

            self.base_instructions.append(instruction.base_instructions[0])
        else:
            assert isinstance(instruction, Instruction)
            self.base_instructions.append(instruction)

    def add_side_effect_param(self, name: str,  scope: str, init_val: Union[str, int], side_effect_str):
        if scope not in ['codelet', 'program', 'operation']:
            raise ValueError(f"Invalid level for side effect. Must be one of:\n"
                             f"'program', 'codelet', 'operation'")
        args = Instruction.DEFAULT_FN_ARGS + (name,)
        fp_se = FlexParam(f"side_effect_{name}", args, side_effect_str)
        new_side_effect = SideEffect(name, scope, init_val, fp_se)
        self.side_effects.append(new_side_effect)

    def evaluate_side_effects(self, program, hag, relocatables, cdlt, op):
        base_args = (program, hag, relocatables, cdlt, op)
        for se in self.side_effects:
            pass


    def get_base_instr_by_name(self, instr_name):
        for bi in self.base_instructions:
            if bi.name == instr_name:
                return bi
        raise RuntimeError(f"Could not retrieve name for instruction {instr_name}. Current instructions:\n"
                           f"{self.base_instr_str()}")

    def get_base_instr_by_index(self, index):
        if len(self.base_instructions) <= index:
            raise RuntimeError(f"Could not retrieve instruction index {index}. Current instructions:\n"
                               f"{self.base_instr_str()}")
        return self.base_instructions[index]

    def add_instruction(self, instruction: Instruction):
        self.instructions.append(instruction)

    def add_condition(self, condition_str: str):
        base_args = Instruction.DEFAULT_FN_ARGS + self.iter_args
        self.conditional = FlexParam("conditional", base_args, condition_str)

    def add_iterable(self, arg_name: str, iterable):
        iterable_param = FlexParam(arg_name, Instruction.DEFAULT_FN_ARGS, iterable)
        self.iter_args.append(arg_name)
        self.iterables.append(iterable_param)
        if self.conditional is not None:
            self.conditional.add_fn_arg(arg_name)

    def base_instr_str(self):
        return '\n'.join([str(instr) for instr in self.base_instructions])

    def set_instructions(self, instruction_list: List[Instruction]):
        if len(self.instructions) > 0:
            raise RuntimeError(f"Instructions have already been evaluated!:\n"
                               f"Base instructions: {self.base_instr_str()}")
        self.instructions = instruction_list

    def evaluate(self, program, hag, op_idx, cdlt_id):
        fn_args = self.create_fn_args(program, hag, cdlt_id, op_idx)

        instructions = self.evaluate_iterable_instructions(fn_args, 0, {})
        self.set_instructions(instructions)

    def set_field_by_name(self, field_name, field_value, instr_name=None):
        if len(self.base_instructions) > 1 and instr_name is None:
            raise RuntimeError(f"Instruction name is a required parameter for setting field names in "
                               f"FlexTemplates with more than one base Instruction.\n"
                               f"Field name: {field_name}\n"
                               f"Value: {field_value}\n"
                               f"Base instructions: {self.base_instr_str()}")
        elif instr_name is None:
            base_instr = self.base_instructions[0]
        elif isinstance(instr_name, str):
            base_instr = self.get_base_instr_by_name(instr_name)
        else:
            assert isinstance(instr_name, int)
            base_instr = self.get_base_instr_by_index(instr_name)
        base_instr.set_field_by_name(field_name, field_value)

    def set_field_flex_param(self, field_name, param_fn_str, instr_name=None):

        if len(self.base_instructions) > 1 and instr_name is None:
            raise RuntimeError(f"Instruction name is a required parameter for setting flex params in "
                               f"FlexTemplates with more than one base Instruction.\n"
                               f"Field name: {field_name}\n"
                               f"param func: {param_fn_str}\n"
                               f"Base instructions: {self.base_instr_str()}")
        elif instr_name is None:
            base_instr = self.base_instructions[0]

        elif isinstance(instr_name, str):
            base_instr = self.get_base_instr_by_name(instr_name)
        else:
            assert isinstance(instr_name, int)
            base_instr = self.get_base_instr_by_index(instr_name)
        base_instr.set_field_flex_param(field_name, param_fn_str)


    def set_field_value(self, field_name, value, value_str=None, instr_name=None):
        if len(self.base_instructions) > 1 and instr_name is None:
            raise RuntimeError(f"Instruction name is a required parameter for setting field value in "
                               f"FlexTemplates with more than one base Instruction.\n"
                               f"Field name: {field_name}\n"
                               f"value: {value}\n"
                               f"Value str: {value_str}\n"
                               f"Base instructions: {self.base_instr_str()}")
        elif instr_name is None:
            base_instr = self.base_instructions[0]
        elif isinstance(instr_name, str):
            base_instr = self.get_base_instr_by_name(instr_name)
        else:
            assert isinstance(instr_name, int)
            base_instr = self.get_base_instr_by_index(instr_name)
        base_instr.set_field_value(field_name, value, value_str=value_str)

    def evaluate_instr_len(self, idx, fn_args, instr_size):
        if len(self.iterables) < idx:
            iterable = self.iterables[idx].evaluate_fn(*fn_args)
            for i in iterable:
                fn_args_i = fn_args + (i,)
                instr_size += self.evaluate_instr_len(idx + 1, fn_args_i, instr_size)
        else:
            instr_size += 1

        return instr_size

    def set_instruction_length(self, program, hag, op_idx, cdlt_id):
        instr_size = 0
        fn_args = self.create_fn_args(program, hag, cdlt_id, op_idx)

        for idx in range(len(self.iterables)):
            iterable = self.iterables[idx].evaluate_fn(*fn_args)
            instr_size *= len(iterable)
        self.num_instructions = instr_size
        # self.num_instructions = self.evaluate_instr_len(0, fn_args, instr_size)

    def evaluate_conditional(self, fn_args, iter_args):

        if self.conditional is not None:
            fn_args = fn_args + tuple(iter_args.values())
            condition = self.conditional.evaluate_fn(*fn_args, force_evaluate=True)
        else:
            condition = True
        return condition


    def evaluate_iterable_instructions(self, fn_args: tuple, iter_idx: int, iter_args: dict):
        instructions = []

        if iter_idx >= len(self.iterables):
            for bi in self.base_instructions:
                instruction = bi.instruction_copy()
                condition = self.evaluate_conditional(fn_args, iter_args)
                if condition:
                    instruction.evaluate_fields(fn_args, iter_args)
                    instructions.append(instruction)
                    self.evaluate_side_effects(*fn_args)
        else:
            iter_arg_name = self.iter_args[iter_idx]
            iterable_fnc = self.iterables[iter_idx]
            # TODO: Add checks for validation here
            iterable = iterable_fnc.evaluate_fn(*fn_args)
            iter_idx += 1

            for i in iterable:
                iter_args[iter_arg_name] = i
                instructions += self.evaluate_iterable_instructions(fn_args, iter_idx, iter_args)

        return instructions

    def template_copy(self):
        return FlexTemplate([bi.instruction_copy() for bi in self.base_instructions],
                            side_effects=[v.copy() for v in self.side_effects],
                            iter_args=self.iter_args.copy(),
                            iterables=self.iterables.copy(),
                            conditional=None if not self.conditional else self.conditional.copy()
                            )

    def create_fn_args(self, program, hag, cdlt_id, op_id):
        cdlt = program.get_codelet(cdlt_id)
        op = cdlt.get_op(op_id)
        args = [program, hag, program.relocatables, cdlt, op]
        return tuple(args)

    def emit(self, output_type="string_final"):

        if len(self.instructions) == 0:
            return []

        assert len(self.instructions) > 0
        instr_strings = []
        for i in self.instructions:
            instr = i.emit(output_type)
            instr_strings.append(instr)

        return instr_strings