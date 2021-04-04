import numpy as np
from typing import List, Dict, Union
from codelets.adl.flex_param import FlexParam
from itertools import count
from dataclasses import dataclass, field
from numbers import Integral

field_cnt = count()
INSTR_FN_NAME_TEMPLATE = """param_fn{FN_ID}"""
INSTR_FN_TEMPLATE = """def param_fn{FN_ID}(program, hag, relocation_table, cdlt, op{ITER_ARGS}):\n\treturn {FN_BODY}"""
FIELD_HEADER = """<%def field_name="{field_name}(hag, op, cdlt, relocation_table, program, iter_arg=None, cond_arg=None)">"""
FIELD_END = "</%def>"
FIELD_TEMPLATE = """
<%def field_name="{field_name}(cdlt, op, output_type, iter_arg=None, cond_arg=None)">
    % if 
</%def>
"""

@dataclass
class Field:
    field_name: str
    bitwidth: int
    field_id: int = field(default_factory=lambda: next(field_cnt))
    value: int = field(default=None)
    value_names: Dict[str, int] = field(default_factory=dict)
    value_str: str = field(default=None)
    param_fn: FlexParam = field(default=None, init=False)
    param_fn_type: str = field(default="int")


    @property
    def isset(self) -> bool:
        return self.value is not None

    @property
    def value_name_list(self) -> List[str]:
        return list(self.value_names.keys())

    def set_param_fn(self, fn: FlexParam, eval_type="int"):
        assert isinstance(fn, FlexParam)
        self.param_fn = fn
        assert eval_type in ['string', 'int']
        self.param_fn_type = eval_type

    def set_value(self, value):
        if self.value is not None:
            raise ValueError(f"{self.field_name} has already been set to {self.value}")
        self.value = value

    def set_value_by_string(self, value_name) -> None:
        if value_name not in self.value_names:
            raise ValueError(f"{value_name} is not mapped to any values for this field.\n"
                             f"Possible values: {self.value_name_list}")
        elif self.value is not None and self.value_str != value_name:
            raise ValueError(f"Value for {self.field_name} has already been set to {self.value_str}, cannot set to {value_name}")
        else:
            self.value_str = value_name
            self.value = self.value_names[value_name]

    def get_string_value(self) -> Union[str]:
        if self.value_str is not None:
            return self.value_str
        return str(self.value)


    def set_value_from_param_fn(self, *args, **iter_args):
        param_fn_args = list(args)
        for k, v in iter_args.items():
            if k not in self.param_fn.fn_args:
                self.param_fn.add_fn_arg(k)
                param_fn_args.append(v)
        param_fn_args = tuple(param_fn_args)
        result = self.param_fn.evaluate_fn(*param_fn_args)

        if isinstance(result, str) and result in self.value_names:
            self.value = self.value_names[result]
            self.value_str = result
        elif not isinstance(result, Integral):
            raise RuntimeError(f"Non-integer result value which is not found in value names:\n"
                               f"Value: {result}, Type: {type(result)}\n"
                               f"Possible string values: {list(self.value_names.keys())}"
                  f"Arg name: {self.field_name}")
        else:
            self.value = result

    def template_header(self):
        return FIELD_HEADER.format(field_name=self.field_name)

    def emit(self, output_type):
        if output_type == "string_final":
            if self.isset:
                return self.get_string_value()
            else:
                return f"${{{self.param_fn}}}"
        elif output_type == "string_placeholders":
            return f"$({self.field_name})"
        elif output_type == "decimal":
            assert self.isset
            return f"{self.value}"
        else:
            assert output_type == "binary"
            assert self.isset
            bin_rep = np.binary_repr(self.value, self.bitwidth)
            return f"{bin_rep}"

    def copy(self):
        field = Field(self.field_name, self.bitwidth, self.field_id, self.value, self.value_names.copy(),
                     self.value_str)

        flex_param = None if not self.param_fn else self.param_fn.copy()
        field.param_fn = flex_param
        return field
