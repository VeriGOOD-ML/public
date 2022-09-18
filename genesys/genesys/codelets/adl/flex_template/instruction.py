import numpy as np
from collections import namedtuple
from typing import Callable, List, Dict, Optional, Union
from types import FunctionType
from . import Field
import inspect
from codelets.adl.flex_param import FlexParam

CodeletOperand = namedtuple('Operand', ['field_name', 'supported_dtypes'])


def default_str_format(opname, fields, tabs):
    fields = ', '.join([f.emit("string_final") for f in fields])
    instr_str = tabs * "\t" + f"{opname} {fields}"
    return instr_str


class Instruction(object):
    STR_FN_ARGS = "program, hag, relocation_table, cdlt, op{OTHERS}"
    DEFAULT_FN_ARGS = ["program", "hag", "relocation_table", "cdlt", "op", "template"]
    PROGRAM_FN_ARGS = ["program", "hag", "relocation_table", "template"]
    CDLT_FN_ARGS = ["program", "hag", "relocation_table", "cdlt", "template"]
    INSTR_TYPE_ARGS = {"instruction": DEFAULT_FN_ARGS,
                       "program": PROGRAM_FN_ARGS,
                       "codelet": CDLT_FN_ARGS}

    SELF_ARG = ["instruction"]
    DEFAULT_FORMAT = ""
    # TODO: Add test for
    def __init__(self, opname, opcode, opcode_width, fields,
                 target=None,
                 field_values=None,
                 latency=1,
                 num_output_supported=True,
                 str_output_supported=True,
                 format_str_fn=None,
                 instr_type="instruction",
                 **kwargs):
        self._str_output_supported = str_output_supported
        self._num_output_supported = num_output_supported
        self._instr_type = instr_type

        if format_str_fn is not None:
            assert isinstance(format_str_fn, FunctionType)
            args = inspect.getfullargspec(format_str_fn)[0]
            assert len(args) == 3 and args == ['opname', 'fields', 'tabs']
            self._format_str_fn = format_str_fn
        else:
            self._format_str_fn = default_str_format

        self._opname = opname
        self._opcode = opcode
        self._opcode_width = opcode_width
        self._extra_params = kwargs
        self._latency = latency
        self._target = target
        self._fields = fields
        self._field_values = field_values or {}
        self._field_map = {f.field_name: f for f in fields}
        self._instr_length = opcode_width + sum([f.bitwidth for f in fields])
        self._tabs = None


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
    def tabs(self):
        return self._tabs

    @property
    def instr_type(self):
        return self._instr_type

    @property
    def opcode_width(self) -> int:
        return self._opcode_width

    @property
    def extra_params(self) -> Dict:
        return self._extra_params

    @property
    def instr_length(self):
        return self._instr_length

    @property
    def format_str_fn(self):
        return self._format_str_fn

    def set_format_str_fn(self, format_str_fn):
        assert isinstance(format_str_fn, FunctionType)
        args = inspect.getfullargspec(format_str_fn)[0]
        assert len(args) == 3 and args == ['opname', 'fields', 'tabs']
        self._format_str_fn = format_str_fn

    def bin(self) -> str:
        return np.binary_repr(self.opcode, self.opcode_width)

    def decimal(self) -> str:
        return str(self.opcode)

    def set_tabs(self, tabs: int):
        assert self.tabs is None and tabs >= 0
        self._tabs = tabs

    @property
    def num_output_supported(self) -> bool:
        return self._num_output_supported

    @property
    def str_output_supported(self) -> bool:
        return self._str_output_supported

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
                            latency=self.latency,
                            format_str_fn=self.format_str_fn
                            )
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

    def set_field_flex_param(self, field_name: str, param_fn: str, instr_type: str, lazy_eval=False):
        if field_name not in self.field_names:
            raise ValueError(f"{field_name} is not a field for instruction  {self.name}:\n"
                             f"Fields: {self.fields}")
        field = self.get_field(field_name)
        if field.param_fn is not None:
            raise ValueError(f"Param function for {field_name} is already set:\n"
                             f"Set param fn: {field.param_fn}\n"
                             f"New param fn: {param_fn}")
        flex_param = FlexParam(self.name, Instruction.INSTR_TYPE_ARGS[instr_type] + Instruction.SELF_ARG, param_fn)
        field.set_param_fn(flex_param)
        field.lazy_eval = lazy_eval

    def set_field_flex_param_str(self, field_name: str, param_fn: str, instr_type: str, lazy_eval=False):
        if field_name not in self.field_names:
            raise ValueError(f"{field_name} is not a field for this capability:\n"
                             f"Fields: {self.fields}")
        field = self.get_field(field_name)
        if field.param_fn is not None:
            raise ValueError(f"Param function for {field_name} is already set:\n"
                             f"Set param fn: {field.param_fn}\n"
                             f"New param fn: {param_fn}")

        flex_param = FlexParam(self.name, Instruction.INSTR_TYPE_ARGS[instr_type] + Instruction.SELF_ARG, param_fn)
        field.set_param_fn(flex_param, eval_type="string")
        field.lazy_eval = lazy_eval



    def evaluate_fields(self, fn_args: tuple, iter_args: dict):
        fn_args = fn_args + (self,)
        for f in self.fields:

            if not f.isset and not f.lazy_eval:
                assert f.param_fn is not None
                res = f.set_value_from_param_fn(*fn_args, **iter_args)
                if res is not None:
                    assert isinstance(res, str)
                    raise RuntimeError(f"Unable to evaluate instruction {self.opname}\n"
                                       f"Function: {f.param_fn.fn_body_str}\n"
                                       f"Field error: {res}")

    def evaluate_lazy_fields(self, fn_args: tuple, iter_args: dict):
        fn_args = fn_args + (self,)
        for f in self.fields:
            if f.lazy_eval:
                f.set_value_from_param_fn(*fn_args, **iter_args)

    # TODO: perform check for evaluated params
    def emit(self, output_type):
        instruction = []
        if output_type in ["binary", "decimal"]:
            assert self.num_output_supported
            instruction.append(self.bin())
            for f in self.fields:
                f_bin = f.emit(output_type)
                instruction.append(f_bin)
            if output_type == "decimal":
                return str(int("".join(instruction), 2))
            else:
                return "".join(instruction)
        else:
            assert self.str_output_supported
            assert self.tabs is not None
            return self.format_str_fn(self.opname, self.fields, self.tabs)

    def update_instr_args_from_type(self, old_args, new_args):
        for f in self.fields:
            f.update_fn_arg_names(old_args, new_args)