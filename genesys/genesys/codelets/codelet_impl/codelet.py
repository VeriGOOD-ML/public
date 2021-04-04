from typing import List, Union, Dict


from codelets.adl.flex_param import FlexParam
from codelets.adl.operation import Operation, Loop, Transfer, Compute, Configure, OperandTemplate, Datatype
from types import LambdaType
from pytools import memoize_method
from collections import defaultdict
from copy import deepcopy

import polymath as pm

class Codelet(object):
    codelet_instance_id = 0
    codelet_id = 0
    operations = []

    def __init__(self, op_name,
                 inputs: List[OperandTemplate],
                 outputs: List[OperandTemplate],
                 hag,
                 is_instance: bool = False,
                 cdlt_id: int = None,
                 required_params: Dict[str, Union[int, str, FlexParam, LambdaType]] = None):

        self._op_name = op_name
        self._inputs = inputs
        self._outputs = outputs
        self._ops = []
        self._op_map = {}
        self._global_op_map = {}
        self._num_instr = -1
        self._hag = hag
        # Added, possibly need to consolidate
        self._domain_tiling = {}
        self._tile_levels = defaultdict(list)
        self._domain_loop_map = {}

        self._id_counter = 0
        self._loop_ctxt_level = 0
        self._op_id_counters = defaultdict(int)
        self._compilation_params = {}
        self._size_map = {}

        if required_params is not None:
            self._required_params = {}
            for k, v in required_params.items():
                self.add_required_param(k, v)
        else:
            self._required_params = {}
        self._is_instance = is_instance
        self._instance_id = None
        if self.is_instance:
            self._instance_id = Codelet.codelet_instance_id
            Codelet.codelet_instance_id += 1

        if cdlt_id:
            self._cdlt_id = cdlt_id
        else:
            self._cdlt_id = Codelet.codelet_id
            Codelet.codelet_id += 1

    def __enter__(self):
        Operation.current_codelet = self
        Operation.loop_stack.append(-1)
        Operation.id_counter = 0
        Operation.loop_ctxt_level = 0
        Operation.op_id_counters = defaultdict(int)
        OperandTemplate.current_codelet = self

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Operation.current_codelet = None
        OperandTemplate.current_codelet = None
        last_id = Operation.loop_stack.pop()
        self._id_counter = deepcopy(Operation.id_counter)
        self._loop_ctxt_level = deepcopy(Operation.loop_ctxt_level)
        self._op_id_counters = deepcopy(Operation.op_id_counters)
        assert last_id == -1

    @property
    def num_loop_levels(self):
        return max([k for k in self.tile_levels.keys()])

    @property
    def is_instance(self):
        return self._is_instance

    @property
    def compilation_params(self):
        return self._compilation_params

    @property
    def instance_id(self):
        return self._instance_id

    @property
    def hag(self):
        return self._hag

    @property
    def cdlt_id(self):
        return self._cdlt_id

    @property
    def op_name(self):
        return self._op_name

    @property
    def required_params(self) -> Dict[str, Union[None, FlexParam]]:
        return self._required_params

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def ops(self) -> List[Operation]:
        return self._ops

    @property
    def cdlt_uid(self):
        return f"{self.op_name}{self.instance_id}"

    @property
    def op_map(self) -> Dict[str, Union[Loop, Compute, Transfer, Configure]]:
        return self._op_map

    @property
    def global_op_map(self) -> Dict[int, Union[Loop, Compute, Transfer, Configure]]:
        return self._global_op_map

    @property
    def size_map(self):
        return self._size_map

    # TODO: Memoize this method
    @property
    def operands(self):
        return self.inputs + self.outputs

    @ops.setter
    def ops(self, ops):
        self._ops = ops

    @property
    def num_instr(self):
        return self._num_instr

    @property
    def op_id_counters(self):
        return self._op_id_counters

    @property
    def id_counter(self):
        return self._id_counter

    @id_counter.setter
    def id_counter(self, id_counter):
        self._id_counter = id_counter


    @memoize_method
    def operand_dimensions(self) -> List[str]:
        operands = self.inputs + self.outputs
        operand_dims = []
        for o in operands:
            operand_dims += o.shape_list
        return list(set(operand_dims))

    @property
    def domain_tiling(self):
        return self._domain_tiling

    @property
    def domain_loop_map(self):
        return self._domain_loop_map

    @property
    def tile_levels(self):
        return self._tile_levels

    def operand_dim_mapping(self):
        operands = self.inputs + self.outputs
        operand_dims = {}
        for o in operands:
            operand_dims.update(o.shape_symbols)
        return operand_dims

    def add_compilation_param(self, key, value):
        self._compilation_params[key] = value

    def unset_params(self):
        unset_params = []
        for k, v in self.required_params.items():
            if v is None:
                unset_params.append(k)
            else:
                assert isinstance(v, FlexParam)
                if not v.is_set():
                    unset_params.append(k)
        return unset_params

    def get_operand(self, op_name: str):
        for o in (self.inputs + self.outputs):
            if o.name == op_name:
                return o
        raise KeyError(f"Unable to find operand {op_name}: {self.inputs + self.outputs}")

    def get_ops_by_type(self, op_type):
        ops = []
        for o in self.ops:
            if o.op_type == op_type:
                ops.append(o)
        return ops

    def get_max_loop_level(self):
        max_level = -1
        for l in self.ops:
            if l.loop_level > max_level:
                max_level = l.loop_level
        return max_level

    def extract_bands(self):
        bands = []
        start_idx = None
        prev_loop_level = -1
        found_band = False

        for i, o in enumerate(self.ops):
            if o.op_type == "loop":
                if not found_band:
                    found_band = True
                    start_idx = i
                    prev_loop_level = o.loop_level
                elif o.loop_level < prev_loop_level:
                    bands.append((start_idx, i))
                    found_band = False
            elif o.loop_level == prev_loop_level:
                prev_loop_level = -1
                bands.append((start_idx, i-1))
                found_band = False

        if found_band:
            assert start_idx >= 0
            bands.append((start_idx, len(self.ops) - 1))

        return bands

    def copy(self, pre_increment=False):
        if pre_increment:
            Codelet.codelet_instance_id += 1
        obj = type(self).__new__(self.__class__)
        obj._op_name = self.op_name
        obj._cdlt_id = self.cdlt_id
        obj._inputs = [i.copy() for i in self.inputs]
        obj._outputs = [o.copy() for o in self.outputs]
        obj._required_params = self.copy_required_params()
        obj._hag = self.hag
        obj._ops = []
        obj._op_map = {}
        obj._global_op_map = {}
        obj._num_instr = self._num_instr
        obj._cdlt_id = self._cdlt_id
        obj._instance_id = Codelet.codelet_instance_id
        obj._domain_tiling = deepcopy(self._domain_tiling)
        obj._tile_levels = deepcopy(self._tile_levels)
        obj._domain_loop_map = deepcopy(self._domain_loop_map)
        obj._op_id_counters = deepcopy(self._op_id_counters)
        obj._id_counter = self._id_counter
        obj._loop_ctxt_level = self._loop_ctxt_level
        obj._compilation_params = deepcopy(self._compilation_params)
        for o in self.ops:
            obj.add_op(o.copy(obj))
        return obj

    def copy_required_params(self):
        params = {}
        for k, v in self.required_params.items():
            if isinstance(v, FlexParam):
                params[k] = v.copy()
            elif v is None:
                params[k] = v
            else:
                raise TypeError(f"Invalid type when copying params:\n"
                                f"Name: {k}\n"
                                f"Param: {v}")
        return params

    def get_op(self, global_op_id: int) -> Operation:
        for o in self.ops:
            if o.global_op_id == global_op_id:
                return o
        raise KeyError(f"Unable to find global op id {global_op_id}")


    def get_loop_order(self):
        operand_dim_map = self.operand_dim_mapping()

        loop_order = []
        for loop_name in self.domain_loop_map[0].keys():
            loop = self.op_map[str(loop_name)]
            rpl = loop.required_params
            for rp in rpl:
                if rp in operand_dim_map:
                    loop_order.append((rp, loop.loop_level))
                    break
        loop_order = [k[0] for k in sorted(loop_order, key=lambda x: x[1])]
        return loop_order

    def emit(self, output_type):
        if output_type == "operations":
            op_str = f"CODELET:\t{self.op_name}{self.instance_id}\n"
            for o in self.ops:
                ostr = f"\t" * (o.loop_level + 1)
                ostr += f"{o.emit(output_type)}\n"
                op_str += ostr
        elif output_type == "operations_idx":
            op_str = f"CODELET:\t{self.op_name}{self.instance_id}\n"
            for i, o in enumerate(self.ops):
                ostr = f"{i}" + f"\t" * (o.loop_level + 1)
                ostr += f"{o.emit(output_type[:-4])}\n"
                op_str += ostr
        elif output_type == "json":
            op_params = {}
            operand_dim_map = self.operand_dim_mapping()
            for k, v in self.required_params.items():
                if k not in operand_dim_map:
                    assert isinstance(v, FlexParam)
                    op_params[k] = v.value

            op_str = {}
            loop_order = self.get_loop_order()

            op_str['operation'] = self.op_name
            op_str['instance_id'] = self.instance_id
            op_str['iterable_dimensions'] = {k: operand_dim_map[k] for k in loop_order}
            op_str['operation_parameters'] = op_params
            op_str['inputs'] = [i.emit(output_type) for i in self.inputs]
            op_str['outputs'] = [o.emit(output_type) for o in self.outputs]
            op_str['operation_sequence'] = [op.emit(output_type) for op in self.ops]
        elif output_type == "json_no_ops":
            op_params = {}
            operand_dim_map = self.operand_dim_mapping()

            loop_order = self.get_loop_order()

            for k, v in self.required_params.items():
                if k not in operand_dim_map:
                    assert isinstance(v, FlexParam)
                    op_params[k] = v.value

            op_str = {}
            op_str['operation'] = self.op_name
            op_str['instance_id'] = self.instance_id
            op_str['iterable_dimensions'] = {k: operand_dim_map[k] for k in loop_order}
            op_str['operation_parameters'] = op_params
            op_str['inputs'] = [i.emit("json") for i in self.inputs]
            op_str['outputs'] = [o.emit("json") for o in self.outputs]

        elif output_type not in ["decimal", "binary"]:
            op_str = f"CODELET:\t{self.op_name}{self.instance_id}\n"
            for o in self.ops:
                instr_list = o.emit(output_type)
                if len(instr_list) > 0:
                    ostr = f"\t" * (o.loop_level + 1)
                    instr_list = f"\n{ostr}".join(instr_list)
                    ostr += f"{instr_list}\n"
                    op_str += ostr
        else:
            op_str = []
            for o in self.ops:
                instr_list = o.emit(output_type)
                op_str += instr_list
            op_str = "\n".join(op_str)
        return op_str

    def add_op(self, op: Operation):
        for rp_key in op.required_params:
            if rp_key not in self.required_params:
                self.add_required_param(rp_key)
        self.ops.append(op)
        self.op_map[op.op_str] = op
        self.global_op_map[op.global_op_id] = op

    def insert_op(self, op: Operation, insert_idx: int, **kwargs):
        if op in self.ops:
            self.ops.insert(insert_idx, self.ops.pop(self.ops.index(op)))
        else:
            for rp_key in op.required_params:
                if rp_key not in self.required_params:
                    self.add_required_param(rp_key)
            self.ops.insert(insert_idx, op)
            self.op_map[op.op_str] = op
            self.global_op_map[op.global_op_id] = op


    def add_required_param(self, key, value=None, check_key=True):

        if key in self.required_params:
            if check_key:
                raise KeyError(f"Key {key} already exists in params:\n"
                               f"Previous value: {self.required_params[key]}\n"
                               f"Updated value: {value}")
            else:
                return

        if isinstance(value, LambdaType):
            flex_param = FlexParam(key, fn=value)
            for a in flex_param.fn_args:
                if a not in self.required_params:
                    self.add_required_param(a)

            self.required_params[key] = flex_param
        elif isinstance(value, int):

            flex_param = FlexParam(key)
            flex_param.value = value
            self.required_params[key] = flex_param
        elif value is None:
            self.required_params[key] = FlexParam(key)
        elif isinstance(value, FlexParam):
            self.required_params[key] = value
        else:
            raise TypeError(f"Invalid type for required param:\n"
                            f"Name: {key}\n"
                            f"Value: {value}")

    def set_required_param(self, key: str, value: int):

        value = value.value if isinstance(value, FlexParam) else value
        if key not in self.required_params:
            raise KeyError(f"Key {key} for updating param does not exist:\n"
                           f"All Keys: {self.required_params.keys()}\n"
                           f"Updated value: {value}")


        # TODO: Check back on this
        if self.required_params[key].is_set() and self.required_params[key].value != value and\
                not isinstance(self.required_params[key].value, LambdaType):
            raise RuntimeError(f"Param {key} has already been set:\n"
                               f"Previous value: {self.required_params[key]}\n"
                               f"New value: {value}")
        if value is None:
            raise RuntimeError(f"Cannot self None value for required parameter:\n"
                               f"Value: {value}\n"
                               f"Key: {key}")
        self.required_params[key].value = value

    def configure(self, start_end, target_name, **kwargs):
        cfg = Configure(start_end, target_name,
                        add_codelet=False, **kwargs)
        self.add_op(cfg)

    def is_loop_node_target(self, loop, hag_node):
        for o in self.ops:
            if o.op_type == 'compute' and o.target == hag_node and loop.loop_level <= o.loop_level:
                return True
            elif o.op_type == 'transfer' and hag_node in o.path and loop.loop_level <= o.loop_level:
                return True
        return False

    def is_direct_loop_dep(self, loop, hag_node):
        for o in self.ops:
            if o.op_type == 'compute' and o.target == hag_node and loop.op_str in o.dependencies:
                return True
            elif o.op_type == 'transfer' and hag_node in o.path and loop.op_str in o.dependencies:
                return True
        return False

    def ordered_loop_ops(self):
        ops = []
        for o in self.ops:
            if o.op_type == 'loop':
                ops.append(o)
        return ops

    def num_op_type(self, op_type):
        count = 0
        for i in self.ops:
            if i.op_type == op_type:
                count += 1
        return count

    def compute(self, op_name, sources, dests, **kwargs):
        comp = Compute(op_name, sources, dests,
                        add_codelet=False, **kwargs)
        self.add_op(comp)

    def transfer(self, operand, path, sizes=None, **kwargs):
        xfer = Transfer(operand, path, sizes=sizes,
                        add_codelet=False, **kwargs)
        self.add_op(xfer)

    def set_domain_tile(self, tile_level: str, domain_key: str, split_factor: int):

        if tile_level not in self.domain_tiling:
            self.domain_tiling[tile_level] = {}

        if domain_key in self.domain_tiling[tile_level] and self.domain_tiling[tile_level][domain_key] != split_factor:
            raise RuntimeError(f"The tile split factor has already been set for level{tile_level}: "
                               f"{self.tile_levels[tile_level]} in domain"
                               f" {domain_key}:\n"
                               f"Previous value: {self.domain_tiling[tile_level][domain_key]}\n"
                               f"New value: {split_factor}")
        # TODO: Add other checks here to validate split
        self.domain_tiling[tile_level][domain_key] = split_factor

    def set_tile_levels(self):
        self._tile_levels = deepcopy(self.hag.node_levels.copy())
        self._tile_levels = {i: self._tile_levels[i] for i in sorted(list(self._tile_levels.keys()))}

    def set_dim_values(self, node: pm.Node, operand: OperandTemplate):

        if not operand.is_instantiated():
            for j, s in enumerate(node.shape):
                key = operand.shape_list[j]
                operand.update_shape_symbols(key, s)

                if key not in self.required_params:
                    self.add_required_param(key, s)
                elif key in self.required_params:
                    if not self.required_params[key].is_set():
                        self.set_required_param(key, s)
                    elif self.required_params[key].value != s:
                        raise RuntimeError(f"Inconsistent dimension sizes for operation {self.op_name}{self.instance_id}\n"
                                           f"Key: {key}\n"
                                           f"Size: {self.required_params[key].value}\n"
                                           f"Node shape: {node.shape}\n"
                                           f"Node name: {operand.name}\n"
                                           f"Shape list: {operand.shape_list}")

            if len(operand.shape_list) != len(list(operand.shape_symbols.keys())):
                raise RuntimeError(f"All shape values were not set for node {node.name}, operand {operand.name}:\n"
                                   f"Node shape: {node.shape}\n"
                                   f"Operand shape variables: {operand.shape_list}")

    def get_tile_level(self, node_name: str):
        for i in self.tile_levels.keys():
            if node_name in self.tile_levels[i]:
                return i
        raise KeyError(f"Unable to find tile level for node {node_name}")

    def get_tile_splits(self, node_name: str):
        level = self.get_tile_level(node_name)
        return self.domain_tiling[level]

    def get_node_shape_map(self, op_tmplt: OperandTemplate, node: pm.Node) -> Dict[str, Dict]:
        shape_map = {}
        for i, s in enumerate(node.shape):

            key = op_tmplt.shape_symbols[i]
            shape_map[key] = {'value': s,
                              'dimension': i}
        return shape_map

    def set_dtype(self, node, operand):

        if "hag_dtype" in node.kwargs:
            assert operand.is_dtype_supported(node.kwargs['hag_dtype'])
            dtype = Datatype.from_str(node.kwargs['hag_dtype'])
        elif operand.dtype is not None:
            dtype = operand.dtype
        else:
            dtype = operand.supported_dtypes[0]
            node.add_attribute("hag_dtype", str(dtype))
        operand.set_dtype(dtype)

    def set_op_node_name(self, node: pm.Node, operand: OperandTemplate):
        if operand.node_name is not None and operand.node_name != node.name:
            raise RuntimeError(f"Name already set to different value for operand:\n"
                               f"Previous name: {operand.node_name}\n"
                               f"New name: {node.name}")
        operand.set_node_name(node.name)

    def instantiate_operands(self, node: pm.Node):
        all_cdlt_ops = self.inputs + self.outputs
        all_node_ops = node.inputs + node.outputs

        for i, n in enumerate(all_node_ops):
            operand = all_cdlt_ops[i]
            for rp_key in operand.required_params:
                if rp_key not in self.required_params:
                    self.add_required_param(rp_key)
            self.set_dim_values(n, operand)
            self.set_dtype(n, operand)
            self.set_op_node_name(n, operand)

    def instantiate_node_params(self, node, hag):
        fn_params = []
        for key, param in self.required_params.items():
            if key in node.kwargs:
                self.set_required_param(key, node.kwargs[key])
            elif isinstance(param, FlexParam) and param.value_type == "function" and not param.is_set():
                fn_params.append(key)

        for name in fn_params:
            flex_param = self.required_params[name]
            arg_vals = tuple([self.required_params[a].value for a in flex_param.fn_args])
            self.set_required_param(name, flex_param.evaluate_fn(*arg_vals))
        for k, v in self.required_params.items():
            if isinstance(v, FlexParam) and not v.is_set():
                raise RuntimeError(f"Unable to set parameter {v.name}\n"
                                   f"Key: {k}\n"
                                   f"Kwargs: {node.kwargs.keys()}")

    def has_hag_edge(self, src: str, dst: str):
        return self.hag.has_edge(src, dst)

    def get_new_op_ids(self, op: Operation):
        global_id = self.id_counter
        op_id = self.op_id_counters[op.op_type]
        self.id_counter = self.id_counter + 1
        self._op_id_counters[op.op_type] += 1
        return op_id, global_id

    def instantiate_operations(self, node: pm.Node, hag):
        # First initialize shapes and symbols for operands, as well as datatypes
        self.instantiate_operands(node)

        # next, set the parameters supplied by the PolyMath node (e.g., stride, pad, etc)
        self.instantiate_node_params(node, hag)

        for o in self.ops:
            for rp in o.required_params:
                if rp in self.required_params and rp in o.unset_params():
                    o.set_required_param(rp, self.required_params[rp])
            o.dependencies = list(set(o.dependencies))
            if o.op_str in o.dependencies:
                o.dependencies.remove(o.op_str)

            assert isinstance(o, Operation)

        # Now set the required parameters in each operation, as specified
        for o in self.ops:
            o.evaluate_parameters(node, hag, self)

        for operand in self.operands:
            operand.evaluate_operand(node, hag, self)


    def get_operand_shapes(self):
        shape_dims = {}
        operands = (self.inputs + self.outputs)
        for o in operands:
            shape_dims.update(o.shape_symbols)

        return shape_dims






