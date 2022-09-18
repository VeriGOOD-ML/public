from typing import List, Union
from dataclasses import dataclass, field
from .instruction import Instruction
from codelets.adl.flex_param import FlexParam
from .compiler_side_effect import SideEffect
from time import time

DEFAULT_TEMPLATE_ARGS = Instruction.DEFAULT_FN_ARGS + ["template"]
NUM_OP_FN_ARGS = 6

@dataclass
class FlexTemplate:
    base_instructions: List[Instruction]
    instructions: List[Instruction] = field(default_factory=list)
    conditional: FlexParam = field(default=None)
    side_effect_args: List[str] = field(default_factory=list)
    side_effects: List[SideEffect] = field(default_factory=list)
    iter_args: List[str] = field(default_factory=list)
    iterables: List[FlexParam] = field(default_factory=list)
    num_instructions: int = field(init=False, default=1)
    flex_tabs: FlexParam = field(default=None)
    arg_names: List[str] = field(default_factory=lambda: Instruction.DEFAULT_FN_ARGS.copy())
    template_type: str = field(default="instruction")

    def __post_init__(self):
        if isinstance(self.base_instructions, Instruction):
            self.base_instructions = [self.base_instructions]
        assert self.template_type in Instruction.INSTR_TYPE_ARGS
        self.arg_names = Instruction.INSTR_TYPE_ARGS[self.template_type].copy()

    def add_side_effect_param(self, name: str,  scope: str, init_val: Union[str, int], side_effect_str):
        if scope not in ['codelet', 'program', 'operation']:
            raise ValueError(f"Invalid level for side effect. Must be one of:\n"
                             f"'program', 'codelet', 'operation'")
        args = self.arg_names + [name]
        fp_se = FlexParam(f"side_effect_{name}", args, side_effect_str)
        new_side_effect = SideEffect(name, scope, init_val, fp_se)
        self.side_effects.append(new_side_effect)
        assert name not in self.get_flex_arg_names()
        self.side_effect_args.append(f"{name}")
        assert self.template_type in Instruction.INSTR_TYPE_ARGS
        self.arg_names = Instruction.INSTR_TYPE_ARGS[self.template_type].copy() + self.iter_args + self.side_effect_args

        if self.conditional is not None:
            self.conditional.reset_fn_args(self.arg_names)
            # self.conditional.add_fn_arg(name)

        if self.flex_tabs is not None:
            self.flex_tabs.reset_fn_args(self.arg_names)
            # self.flex_tabs.add_fn_arg(name)
    @property
    def instr_length(self):
        return self.base_instructions[0].instr_length

    def get_flex_arg_names(self, include_instruction=False):
        if include_instruction:
            assert self.template_type not in ["program", "codelet"]
            args = Instruction.DEFAULT_FN_ARGS + Instruction.SELF_ARG + self.iter_args + self.side_effect_args
            return args
        else:
            return self.arg_names

    def evaluate_side_effects(self, fn_args, iter_args):
        fn_args = fn_args + tuple(iter_args.values()) + tuple(self.current_sideeffects().values())

        for se in self.side_effects:
            se.run_side_effect(fn_args)
            if len(self.instructions) > 0:
                print(f"New sideffect value is {se.value}\n"
                      f"Instruction: {self.instructions[-1]}")

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
        base_args = self.get_flex_arg_names(include_instruction=self.template_type == "instruction")
        self.conditional = FlexParam("conditional", base_args, condition_str)

    def set_print_tabs(self, tab_val: Union[str, int]):
        base_args = self.get_flex_arg_names()

        if isinstance(tab_val, str):
            self.flex_tabs = FlexParam(f"num_tabs", base_args, tab_val)
        else:
            assert isinstance(tab_val, int)
            self.flex_tabs = FlexParam(f"num_tabs", base_args)
            self.flex_tabs.value = tab_val

    def add_iterable(self, arg_name: str, iterable: str):

        iterable_param = FlexParam(arg_name, Instruction.INSTR_TYPE_ARGS[self.template_type], iterable)
        self.iter_args.append(arg_name)
        self.iterables.append(iterable_param)

        self.arg_names = Instruction.INSTR_TYPE_ARGS[self.template_type] + self.iter_args + self.side_effect_args

        if self.conditional is not None:
            self.conditional.reset_fn_args(self.arg_names)
            # self.conditional.add_fn_arg(name)

        if self.flex_tabs is not None:
            self.flex_tabs.reset_fn_args(self.arg_names)
            # self.flex_tabs.add_fn_arg(name)

        for se in self.side_effects:
            se.side_effect_fp.reset_fn_args(self.arg_names)

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

    def lazy_evaluate(self, program, hag, op_idx, cdlt_id):
        fn_args = self.create_fn_args(program, hag, cdlt_id, op_idx)
        self.lazy_evaluate_iterable_instructions(fn_args, 0, {}, 0)


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

    def set_field_flex_param(self, field_name, param_fn_str, instr_name=None, lazy_eval=False):

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
        base_instr.set_field_flex_param(field_name, param_fn_str, self.template_type, lazy_eval=lazy_eval)

    def set_field_flex_param_str(self, field_name, param_fn_str, instr_name=None, lazy_eval=False):

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
        base_instr.set_field_flex_param_str(field_name, param_fn_str, self.template_type, lazy_eval=lazy_eval)

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
            # fn_args = fn_args + tuple(iter_args.values())
            fn_args = fn_args + tuple(iter_args.values()) + tuple(self.current_sideeffects().values())
            condition = self.conditional.evaluate_fn(*fn_args, force_evaluate=True)
        else:
            condition = True
        return condition

    def evaluate_tabs(self, fn_args, iter_args) -> int:
        if self.flex_tabs is not None:
            # fn_args = fn_args + tuple(iter_args.values())
            fn_args = fn_args + tuple(iter_args.values()) + tuple(self.current_sideeffects().values())
            num_tabs = self.flex_tabs.evaluate_fn(*fn_args, force_evaluate=True)
        elif len(fn_args) != NUM_OP_FN_ARGS:
            num_tabs = 0
        else:
            num_tabs = fn_args[-2].loop_level

        return num_tabs

    def current_sideeffects(self):
        return {se.name: se.value for se in self.side_effects}

    def evaluate_iterable_instructions(self, fn_args: tuple, iter_idx: int, iter_args: dict):
        instructions = []

        if iter_idx >= len(self.iterables):
            for bi in self.base_instructions:
                instruction = bi.instruction_copy()
                if self.template_type == "instruction":
                    cond_args = fn_args + (instruction,)
                else:
                    cond_args = fn_args
                condition = self.evaluate_conditional(cond_args, iter_args)
                if condition:
                    field_args = dict(list(iter_args.items()) + list(self.current_sideeffects().items()))
                    instruction.evaluate_fields(fn_args, field_args)
                    num_tabs = self.evaluate_tabs(fn_args, iter_args)
                    instruction.set_tabs(num_tabs)
                    instructions.append(instruction)
                    self.evaluate_side_effects(fn_args, iter_args)
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

    def lazy_evaluate_iterable_instructions(self, fn_args: tuple, iter_idx: int, iter_args: dict, instr_idx: int):

        if len(self.instructions) <= instr_idx:
            return instr_idx
        elif iter_idx >= len(self.iterables):
            for bi_idx in range(len(self.base_instructions)):
                bi_instr = self.base_instructions[bi_idx]
                condition = self.evaluate_conditional(fn_args + (bi_instr,), iter_args)

                if condition:
                    instruction = self.instructions[instr_idx]
                    field_args = dict(list(iter_args.items()) + list(self.current_sideeffects().items()))
                    instruction.evaluate_lazy_fields(fn_args, field_args)
                    self.evaluate_side_effects(fn_args, iter_args)
                    instr_idx += 1
        else:
            iter_arg_name = self.iter_args[iter_idx]
            iterable_fnc = self.iterables[iter_idx]
            # TODO: Add checks for validation here
            iterable = iterable_fnc.evaluate_fn(*fn_args)
            iter_idx += 1

            for i in iterable:
                iter_args[iter_arg_name] = i
                instr_idx = self.lazy_evaluate_iterable_instructions(fn_args, iter_idx, iter_args, instr_idx)
        return instr_idx

    def template_copy(self):
        return FlexTemplate([bi.instruction_copy() for bi in self.base_instructions],
                            side_effects=[v.copy() for v in self.side_effects],
                            iter_args=self.iter_args.copy(),
                            iterables=self.iterables.copy(),
                            conditional=None if not self.conditional else self.conditional.copy(),
                            flex_tabs=None if not self.flex_tabs else self.flex_tabs.copy(),
                            template_type=self.template_type
                            )

    def create_fn_args(self, program, hag, cdlt_id, op_id):

        args = [program, hag, program.relocatables]
        if cdlt_id >= 0:
            cdlt = program.get_codelet(cdlt_id)
            args.append(cdlt)
            if op_id >= 0:
                op = cdlt.get_op(op_id)
                args.append(op)

        args.append(self)
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

    def update_template_type(self, tmplt_type):
        if tmplt_type != self.template_type:
            prev_template = Instruction.INSTR_TYPE_ARGS[self.template_type]
            assert prev_template == self.arg_names[:len(prev_template)], "Invalid default args"
            new_args = Instruction.INSTR_TYPE_ARGS[tmplt_type].copy() + self.arg_names[len(prev_template):]
            assert len(self.instructions) == 0
            self.template_type = tmplt_type
            for bi in self.base_instructions:
                bi.update_instr_args_from_type(self.arg_names, new_args)
            self.arg_names = new_args