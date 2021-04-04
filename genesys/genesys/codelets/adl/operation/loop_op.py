from . import Operation
from copy import copy
from sympy import symbols, Idx, Expr

ARITHMETIC_LOOP_EVAL = """
"""

class LoopTypes(object):
    LINEAR = 0
    OFFSET = 1
    SCALED = 2

class Loop(Operation):

    loop_ids = 0

    def __init__(self, start,
                 end=None,
                 stride=1,
                 offset=0,
                 loop_op_params=None,
                 add_codelet=True,
                 **kwargs
                 ):
        if end is not None:
            self._start = start
            self._end = end
        else:
            self._start = 0
            self._end = start
        self._stride = stride
        self._offset = offset

        req_params = []
        if loop_op_params:
            req_params += loop_op_params

        if isinstance(self.start, str):
            req_params.append(self.start)

        if isinstance(self.end, str):
            req_params.append(self.end)

        if isinstance(stride, str):
            req_params.append(stride)

        if isinstance(offset, str):
            req_params.append(stride)

        super(Loop, self).__init__("loop", req_params,
                                   add_codelet=add_codelet,
                                   **kwargs)
        if isinstance(self.stride, str):
            stride = symbols(self.stride, integer=True)
            self.param_symbols[self.stride] = stride
        else:
            assert isinstance(self.stride, int)
            stride = self.stride

        if isinstance(self.start, str):
            start = symbols(self.start, integer=True)
            self.param_symbols[self.start] = start
        else:
            start = self.start

        if isinstance(self.end, str):
            end = symbols(self.end, integer=True)
            self.param_symbols[self.end] = end
        else:
            end = self.end


        if isinstance(self.offset, str):
            offset = symbols(self.offset, integer=True)
            self.param_symbols[self.offset] = offset
        else:
            offset = self.offset
        self.param_symbols[self.op_str] = Idx(self.op_str, (start, end))*stride + offset


    def __enter__(self):
        Operation.loop_ctxt_level += 1
        Operation.loop_stack.append(self.loop_id)
        Operation.loop_ctx_dependencies.append(self.op_str)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        Operation.loop_ctxt_level -= 1
        Operation.loop_stack.pop()
        Operation.loop_ctx_dependencies.pop()


    def set_loop_level(self, level):
        pass

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, start):
        self._start = start

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, end):
        self._end = end

    @property
    def stride(self):
        return self._stride

    @stride.setter
    def stride(self, stride):
        self._stride = stride

    @property
    def iter_count(self) -> int:
        args = [self.end, self.start, self.stride, self.offset]
        if not all(isinstance(o, int) for o in args):
            raise TypeError(f"Cannot compute iter count because some parameters are not numbers:\n"
                            f"{args}")
        return (self.end - self.start)//self.stride + self.offset

    @property
    def loop_domains(self):
        return [self.loop_id]

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, offset):
        self._offset = offset

    def loop_parameters(self):
        return {"start": self.start, "extent": self.end, "stride": self.stride, "offset": self.offset}

    # TODO: Need to better define this
    def min(self):
        if not all([isinstance(param, int) for param in self.loop_parameters().values()]):
            raise TypeError(f"Unable to compute minimum value because all loop parameters are not defined:\n"
                            f"Params: {self.loop_parameters()}")
        return self.start*self.stride + self.offset

    def max(self):
        if not all([isinstance(param, int) for param in self.loop_parameters().values()]):
            raise TypeError(f"Unable to compute minimum value because all loop parameters are not defined:\n"
                            f"Params: {self.loop_parameters()}")
        return (self.end - 1)*self.stride + self.offset + 1

    def get_symbol(self):
        indices = list(self.param_symbols[self.op_str].atoms(Idx))
        for i in indices:
            if str(i) == self.op_str:
                return i
        raise KeyError

    def __add__(self, other):
        if isinstance(other, str) and other not in self.param_symbols:
            Operation.current_codelet.add_required_param(other, check_key=False)
            sym = symbols(other, integer=True)
            self.param_symbols[other] = sym
            return self.param_symbols[self.op_str] + sym
        elif isinstance(other, Expr):
            return self.param_symbols[self.op_str] + other
        elif isinstance(other, Operation) and other.op_type == "loop":
            return self.param_symbols[self.op_str] + other.param_symbols[other.op_str]
        else:
            assert isinstance(other, int)
            return self.param_symbols[self.op_str] + other

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, str) and other not in self.param_symbols:
            Operation.current_codelet.add_required_param(other, check_key=False)
            sym = symbols(other, integer=True)
            self.param_symbols[other] = sym
            return self.param_symbols[self.op_str]*sym
        elif isinstance(other, Expr):
            return self.param_symbols[self.op_str] * other
        elif isinstance(other, Operation) and other.op_type == "loop":
            return self.param_symbols[self.op_str] * other.param_symbols[other.op_str]
        else:
            assert isinstance(other, int)
            return self.param_symbols[self.op_str] * other

    def __rmul__(self, other):
        return self.__mul__(other)

    def eval_start(self):
        if isinstance(self.start, int):
            return self.start
        elif isinstance(self.start, Loop):
            raise RuntimeError(f"Unable to handle loop eval currently:\n"
                               f"Loop: {self.op_str}\n"
                               f"Param: {self.start}")
        elif isinstance(self.start, str):
            return self.eval_string_param(self.start)

    def eval_end(self):
        if isinstance(self.end, int):
            return self.end
        elif isinstance(self.end, Loop):
            raise RuntimeError(f"Unable to handle loop eval currently:\n"
                               f"Loop: {self.op_str}\n"
                               f"Param: {self.end}")
        elif isinstance(self.end, str):
            return self.eval_string_param(self.end)

    def eval_stride(self):
        if isinstance(self.stride, int):
            return self.stride
        elif isinstance(self.stride, Loop):
            raise RuntimeError(f"Unable to handle loop eval currently:\n"
                               f"Loop: {self.op_str}\n"
                               f"Param: {self.stride}")
        elif isinstance(self.stride, str):
            return self.eval_string_param(self.stride)
        else:
            raise RuntimeError(f"Cannot evaluate stride because of invalid type: {self.stride}")

    def eval_offset(self):
        if isinstance(self.offset, int):
            return self.offset
        elif isinstance(self.offset, Loop):
            raise RuntimeError(f"Unable to handle loop eval currently:\n"
                               f"Loop: {self.op_str}\n"
                               f"Param: {self.offset}")
        elif isinstance(self.offset, str):
            return self.eval_string_param(self.offset)

    def eval_string_param(self, param_name):
        if param_name not in self.resolved_params:
            raise RuntimeError(f"Error! Start value for {self} was unevaluated: {self.start}")
        else:
            return self.resolved_params[param_name].value

    # TODO: Need to normalize loops
    def evaluate_parameters(self, node, hag, cdlt):
        domain_shape_map = cdlt.get_operand_shapes()


        # if not isinstance(self.start, str) and str(self.end) in domain_shape_map:
        if not isinstance(self.start, str):
            assert self.stride == 1
            cdlt.domain_loop_map[self.op_str] = str(self.end)

        self.start = self.eval_start()
        self.end = self.eval_end()
        self.stride = self.eval_stride()
        self.offset = self.eval_offset()
        if not isinstance(self.start, int):
            raise TypeError(f"Unable to evaluate parameter value as integer:"
                            f"Param name: start\n"
                            f"Loop: {self.op_str}\n"
                            f"Value: {self.start}\n"
                            f"Type: {type(self.start)}")
        if not isinstance(self.end, int):
            raise TypeError(f"Unable to evaluate parameter value as integer:"
                            f"Param name: end\n"
                            f"Loop: {self.op_str}\n"
                            f"Value: {self.end}\n"
                            f"Type: {type(self.end)}")

        if not isinstance(self.stride, int):
            raise TypeError(f"Unable to evaluate parameter value as integer:"
                            f"Param name: stride\n"
                            f"Loop: {self.op_str}\n"
                            f"Value: {self.stride}\n"
                            f"Type: {type(self.stride)}")

        if not isinstance(self.offset, int):
            raise TypeError(f"Unable to evaluate parameter value as integer:"
                            f"Param name: offset\n"
                            f"Loop: {self.op_str}\n"
                            f"Value: {self.offset}\n"
                            f"Type: {type(self.offset)}")


    def op_type_params(self):
        op_params = [f"LO: {self.start}", f"HI: {self.end}", f"stride: {self.stride}"]
        return op_params

    def emit(self, output_type):
        # TODO: Add template
        if output_type == "operations":
            op_str = f"{self.op_str}[{self.loop_level}]: START={self.start}; STOP={self.end}; STRIDE={self.stride}; OFFSET:{self.offset}"
        elif output_type == "json":
            op_str = {"op_type": self.op_type,
                      "op_id": self.global_op_id,
                      "start": self.start,
                      "end": self.end,
                      "offset": self.offset,
                      "stride": self.stride
                      }
        else:
            op_str = []
            for ft in self.instructions:
                ft_out = ft.emit(output_type)
                if len(ft_out) == 0:
                    continue
                op_str += ft_out
        return op_str


    def copy(self, cdlt, start=None, stride=None, end=None, offset=None, **kwargs):
        obj = super(Loop, self).copy(cdlt, **kwargs)
        obj._start = start or copy(self.start)
        obj._end = end or copy(self.end)
        obj._stride = stride or copy(self.stride)
        obj._offset = offset or copy(self.offset)
        if obj.op_str not in obj.param_symbols:
            obj_idx = Idx(obj.op_str, (obj._start, obj._end))
            old_idx = obj.param_symbols.pop(self.op_str)
            new_idx = old_idx.subs(old_idx, obj_idx)
            obj.param_symbols[obj.op_str] = new_idx

        return obj
