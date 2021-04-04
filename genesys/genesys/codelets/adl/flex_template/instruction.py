import numpy as np
from collections import namedtuple
from typing import Callable, List, Dict, Optional, Union
from . import Field
from codelets.adl.flex_param import FlexParam

CodeletOperand = namedtuple('OperandTemplate', ['field_name', 'supported_dtypes'])


INSTR_TEMPLATE = """
<%inherit file="{OP_NAME}"/>
<%def field_name="emit_single(output_type{ITER_ARGS})">
% if {CONDITIONAL_STATEMENT}:
    % if output_type == "binary":
        ${{instruction.bin()}}\
        % for field in instruction.fields:
            ${{field.emit(output_type{ITER_ARGS})}}\
        % endfor
    % else:
        ${{instruction.opname}} \
        % for field in instruction.fields:
            ${{field.emit(output_type)}}${{'' if loop.last else ','}}\
        % endfor  
    % endif
% endif
</%def>
<%def field_name="emit(output_type)">
    % if len(instruction.iter_args) == 0:
        ${{emit_single(output_type{ITER_ARGS})}}
    % else:    
        % for iter_val in instruction.iter_args:
            ${{emit_single(output_type{ITER_ARGS})}}
        % endfor 
    % endif 
</%def>
"""



class Instruction(object):
    STR_FN_ARGS = "program, hag, relocation_table, cdlt, op{OTHERS}"
    DEFAULT_FN_ARGS = ["program", "hag", "relocation_table", "cdlt", "op"]
    # TODO: Add test for
    def __init__(self, opname, opcode, opcode_width, fields,
                 target=None,
                 field_values=None,
                 latency=1,
                 **kwargs):
        self._opname = opname
        self._opcode = opcode
        self._opcode_width = opcode_width
        self._extra_params = kwargs
        self._latency = latency
        self._target = target
        self._fields = fields
        self._field_values = field_values or {}
        self._field_map = {f.field_name: f for f in fields}


    @property
    def name(self) -> str:
        return self._opname

    @property
    def opcode(self) -> int:
        return self._opcode

    @property
    def target(self):
        return self._target

    @property
    def opcode_width(self) -> int:
        return self._opcode_width

    @property
    def extra_params(self) -> Dict:
        return self._extra_params

    def bin(self) -> str:
        return np.binary_repr(self.opcode, self.opcode_width)

    @property
    def latency(self) -> Union[int, Callable]:
        return self._latency

    @target.setter
    def target(self, target: str):
        self._target = target

    @property
    def opname(self):
        return self._opname

    def instruction_copy(self):
        fields = [f.copy() for f in self.fields]
        instr = Instruction(self.opname, self.opcode, self.opcode_width, fields,
                            target=self.target,
                            field_values=self.field_values,
                            latency=self.latency)
        return instr

    def __str__(self):
        start = f"{self.opname} "
        rest = []
        for f in self.fields:
            if f.isset:
                val = f.value_str if isinstance(f.value_str, str) else str(f.value)
            else:
                val = f"$({f.field_name})"
            rest.append(val)
        return start + ", ".join(rest)

    @property
    def field_values(self) -> Dict[str, Field]:
        return self._field_values

    @property
    def field_map(self) -> Dict[str, Field]:
        return self._field_map

    @property
    def fields(self) -> List[Field]:
        return self._fields

    @property
    def field_names(self) -> List[str]:
        return [f.field_name for f in self.fields]

    @property
    def unset_field_names(self) -> List[str]:
        return [f.field_name for f in self.fields if not f.isset]

    @property
    def field_widths(self) -> Dict[str, int]:
        return {k: v.bitwidth for k, v in self.field_values.items()}

    @property
    def set_fields(self) -> List[str]:
        return [k for k, v in self.field_values.items() if v.value is not None]

    @property
    def set_field_map(self) -> Dict[str, str]:
        return {k: v.get_string_value() for k, v in self.field_values.items() if v.isset}

    def get_field_value(self, name: str) -> int:
        if name not in self.field_names:
            raise ValueError(f"{name} is not a field for this capability:\n"
                             f"Fields: {self.fields}")
        elif self.field_map[name].value is None:
            raise KeyError(f"{name} is not currently set for this capability:\n"
                             f"Set fields: {self.set_fields}")
        return self.field_map[name].value

    def get_field_value_str(self, name: str) -> str:
        if name not in self.field_names:
            raise ValueError(f"{name} is not a field for this capability:\n"
                             f"Fields: {self.fields}")
        elif self.field_map[name].value_str is None:
            raise KeyError(f"{name} is not currently set for this capability:\n"
                             f"Set fields: {self.set_fields}")
        return self.field_map[name].value_str

    def get_field(self, name: str) -> Field:
        if name not in self.field_names:
            raise ValueError(f"{name} is not a field for this capability:\n"
                             f"Fields: {self.field_names}")
        return self.field_map[name]

    def set_field_by_name(self, field_name: str, value_name: str):
        field = self.get_field(field_name)
        field.set_value_by_string(value_name)

    def set_field_value(self, name: str, value: int, value_str: Optional[str] = None):
        if name not in self.field_names:
            raise ValueError(f"{name} is not a field for this capability:\n"
                             f"Fields: {self.fields}")
        elif name in self.field_values:
            raise KeyError(f"{name} is already set for this capability:\n"
                             f"Set fields: {self.set_fields}")
        self.field_map[name].value = value

        if value_str:
            self.field_map[name] = value_str

    def set_field_flex_param(self, field_name: str, param_fn: str):
        if field_name not in self.field_names:
            raise ValueError(f"{field_name} is not a field for instruction  {self.name}:\n"
                             f"Fields: {self.fields}")
        field = self.get_field(field_name)
        if field.param_fn is not None:
            raise ValueError(f"Param function for {field_name} is already set:\n"
                             f"Set param fn: {field.param_fn}\n"
                             f"New param fn: {param_fn}")
        flex_param = FlexParam(self.name, Instruction.DEFAULT_FN_ARGS, param_fn)
        field.set_param_fn(flex_param)

    def set_field_flex_param_str(self, field_name: str, param_fn: str):
        if field_name not in self.field_names:
            raise ValueError(f"{field_name} is not a field for this capability:\n"
                             f"Fields: {self.fields}")
        field = self.get_field(field_name)
        if field.param_fn is not None:
            raise ValueError(f"Param function for {field_name} is already set:\n"
                             f"Set param fn: {field.param_fn}\n"
                             f"New param fn: {param_fn}")
        flex_param = FlexParam(self.name, Instruction.DEFAULT_FN_ARGS, param_fn)
        field.set_param_fn(flex_param, eval_type="string")


    def evaluate_fields(self, fn_args: tuple, iter_args: dict):
        for f in self.fields:
            if not f.isset:
                f.set_value_from_param_fn(*fn_args, **iter_args)

    # TODO: perform check for evaluated params
    def emit(self, output_type):
        instruction = []
        if output_type == "binary":
            instruction.append(self.bin())
            for f in self.fields:
                f_bin = f.emit(output_type)
                instruction.append(f_bin)
            return "".join(instruction)
        else:
            start = f"{self.opname} "
            for f in self.fields:
                instruction.append(f.emit(output_type))
            return start + ", ".join(instruction)
