from typing import List, Union, Dict, Tuple, Any


from codelets.adl.flex_param import FlexParam
from codelets.adl.operation import Operation, Loop, Transfer, Compute, Configure, Operand, Datatype, LoopEnd
from types import LambdaType
from pytools import memoize_method
from collections import defaultdict
from copy import deepcopy
from contextlib import contextmanager
import numpy as np
import polymath as pm

USE_LOOP_END = Loop.USE_LOOP_END

class Codelet(object):
    codelet_instance_id = 0
    codelet_id = 0
    operations = []

    def __init__(self, op_name,
                 inputs: List[Operand],
                 outputs: List[Operand],
                 hag,
                 is_instance: bool = False,
                 cdlt_id: int = None,
                 required_params: Dict[str, Union[int, str, FlexParam, LambdaType]] = None):

        self._op_name = op_name
        self._inputs = inputs
        self._outputs = outputs
        self._temps = []
        self._ops = []
        self._op_map = {}
        self._global_op_map = {}
        self._hag = hag
        # Added, possibly need to consolidate
        self._domain_tiling = {}
        self._tile_levels = defaultdict(list)
        self._domain_loop_map = {}

        self._id_counter = 0
        self._loop_ctxt_level = 0
        self._op_id_counters = defaultdict(int)
        self._compilation_params = {}
        self._derived_fps = defaultdict(dict)
        self._loop_param_map = {}

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
        Operand.current_codelet = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Operation.current_codelet = None
        Operand.current_codelet = None
        last_id = Operation.loop_stack.pop()
        self._id_counter = deepcopy(Operation.id_counter)
        self._loop_ctxt_level = deepcopy(Operation.loop_ctxt_level)
        self._op_id_counters = deepcopy(Operation.op_id_counters)
        assert last_id == -1, f"Last operation id is invalid when exiting codelet ctxt: {last_id}\n" \
                              f"loop{last_id}\n"

    @contextmanager
    def exit_context(self):
        yield

    @staticmethod
    def reset():
        Codelet.codelet_instance_id = 0
        Codelet.codelet_id = 0
        Codelet.operations = []

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
    def used_inputs(self):
        return [i for i in self.inputs if i.used]

    @property
    def read_operands(self):
        return sum([c.sources for c in self.get_ops_by_type("compute")], start=[])

    @property
    def write_operands(self):
        return sum([c.dests for c in self.get_ops_by_type("compute")], start=[])

    @property
    def temps(self):
        return self._temps

    @property
    def used_temps(self):
        return [t for t in self.temps if t.used]

    @property
    def outputs(self):
        return self._outputs

    @property
    def used_outputs(self):
        return [o for o in self.outputs if o.used]

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
    def loop_param_map(self):
        return self._loop_param_map

    # TODO: Memoize this method
    @property
    def operands(self):
        return self.inputs + self.outputs

    @property
    def used_operands(self):
        return [o for o in self.operands if o.used]

    @ops.setter
    def ops(self, ops):
        self._ops = ops

    @property
    def num_instr(self):
        ilen = 0
        for o in self.ops:
            ilen += sum([len(ft.instructions) for ft in o.instructions])
        return ilen

    @property
    def op_id_counters(self):
        return self._op_id_counters

    @property
    def id_counter(self):
        return self._id_counter

    @id_counter.setter
    def id_counter(self, id_counter):
        self._id_counter = id_counter

    def __repr__(self):
        return f"codelet {self.op_name}{self.instance_id}"

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

    @property
    def all_operands(self):
        return self.inputs + self.temps + self.outputs

    @property
    def all_oploc_indices(self):
        indices = defaultdict(list)
        all_operands = self.operands + self.temps
        for o in all_operands:
            for l in o.data_path:
                if o.name in indices[l]:
                    continue
                indices[l].append(o.name)
        return indices

    @property
    def param_tiling(self):
        ptiling = {}
        for l, tiling in self.domain_loop_map.items():
            ptiling[l] = {}
            for loopname, tile_size in tiling.items():
                ptiling[l][self.loop_param_map[loopname]] = tile_size
        return ptiling

    @property
    def param_splits(self):
        ptiling = {}
        for l, tiling in self.domain_tiling.items():
            ptiling[l] = {}
            for loopname, tile_size in tiling.items():
                ptiling[l][self.loop_param_map[loopname]] = tile_size
        return ptiling

    def num_instr_by_group(self, group_name):
        start_op = None
        end_op = None
        for o in self.ops:
            if o.op_type == "config" and o.target_name == group_name:
                if o.start_or_finish == "start":
                    start_op = o
                elif o.start_or_finish == "end":
                    end_op = o
                    assert start_op is not None
                    break
        if start_op is None:
            raise RuntimeError(f"Unable to find start operation for {group_name} in {self.cdlt_id}")
        if end_op is None:
            raise RuntimeError(f"Unable to find end operation for {group_name} in {self.cdlt_id}")

        start_idx = self.ops.index(start_op)
        end_idx = self.ops.index(end_op)
        ilen = 0
        for o in self.ops[start_idx: end_idx]:
            ilen += sum([len(ft.instructions) for ft in o.instructions])
        return ilen

    def remove_input(self, operand):
        self._inputs.remove(operand)

    def filtered_read_operands(self, compute_name):
        ops = []
        for c in self.get_ops_by_type("compute"):
            if c.target == compute_name:
                ops += c.sources
        return ops

    def filtered_write_operands(self, compute_name):
        ops = []
        for c in self.get_ops_by_type("compute"):
            if c.target == compute_name:
                ops += c.dests
        return ops


    def is_tiling_set(self, level: int):
        return level in self.domain_tiling

    def is_noop(self):
        return len(self.ops) == 0

    def innermost_loop_ids(self) -> List[int]:
        loops = []
        max_loop = 0
        max_level = 0
        for o in self.ops:
            if isinstance(o, Loop) and o.loop_level >= max_level:
                max_loop = o.loop_id
                max_level = o.loop_level
            elif o.loop_level < max_level:
                loops.append(max_loop)
                if isinstance(o, Loop):
                    max_level = o.loop_level
                    max_loop = o.loop_id
        return list(set(loops))

    def get_level_loop_params(self, level):
        loop_params = defaultdict(list)
        for key, lp in self.loop_param_map.items():
            loop_params[lp].append(key)
        loop_params = {k: list(sorted(v, key=lambda x: int(x.split('loop')[1]))) for k,v in loop_params.items()}
        out_params = {k: loop_params[k][level] for k in loop_params.keys()}
        return out_params

    def outermost_loop_ids(self) -> List[int]:
        loops = []
        inner_loops = self.innermost_loop_ids()
        for o in self.ops:
            if isinstance(o, Loop) and o.loop_id not in inner_loops:
                loops.append(o.loop_id)
        return list(set(loops))

    def operand_dim_mapping(self):
        operands = self.inputs + self.outputs + self.temps
        operand_dims = {}
        for o in operands:
            operand_dims.update(o.shape_symbols)
        return operand_dims

    def add_compilation_param(self, key, value):
        self._compilation_params[key] = value

    def update_compilation_param(self, key, value):
        if key in self.compilation_params:
            prev = self.compilation_params[key]
            new_param = f"({prev}) and ({value})"
            self._compilation_params[key] = new_param
        else:
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
        for o in self.operands:
            if o.name == op_name:
                return o
        for t in self.temps:
            if t.name == op_name:
                return t
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

    def get_tile_count(self):
        return np.prod(list(self.domain_tiling[1].values()))


    def execute(self, program, operand_mappings):
        pass

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
                if USE_LOOP_END:
                    assert o.op_type == "loop_end"
                    bands.append((start_idx, i))
                else:
                    bands.append((start_idx, i-1))
                found_band = False

        if found_band:
            assert start_idx >= 0
            # TODO: Update the end index here
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
        obj._temps = [t.copy() for t in self.temps]
        obj._required_params = self.copy_required_params()
        obj._hag = self.hag
        obj._ops = []
        obj._op_map = {}
        obj._global_op_map = {}
        obj._cdlt_id = self._cdlt_id
        obj._instance_id = Codelet.codelet_instance_id
        obj._domain_tiling = deepcopy(self._domain_tiling)
        obj._tile_levels = deepcopy(self._tile_levels)
        obj._domain_loop_map = deepcopy(self._domain_loop_map)
        obj._derived_fps = deepcopy(self._derived_fps)
        obj._op_id_counters = deepcopy(self._op_id_counters)
        obj._id_counter = self._id_counter
        obj._loop_ctxt_level = self._loop_ctxt_level
        obj._compilation_params = deepcopy(self._compilation_params)
        obj._loop_param_map = deepcopy(self._loop_param_map)
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
        if len(self.domain_loop_map) == 0 or 0 not in self.domain_loop_map:
            return []
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
            input_str = ", ".join([f"{i.name}{i.shape_list}" for i in self.inputs])
            out_str = ", ".join([f"{o.name}{o.shape_list}" for o in self.outputs])
            operand_str = f"inputs={input_str}\n" \
                          f"outputs={out_str}\n"
            op_str = f"CODELET:\t{self.op_name}{self.instance_id}\n"
            op_str += operand_str
            for o in self.ops:
                ostr = f"\t" * (o.loop_level + 1)
                ostr += f"{o.emit(output_type)}\n"
                op_str += ostr
        elif output_type == "operations_idx":
            input_str = ", ".join([f"{i.name}{i.shape_list}" for i in self.inputs])
            out_str = ", ".join([f"{o.name}{o.shape_list}" for o in self.outputs])
            operand_str = f"inputs={input_str}\n" \
                          f"outputs={out_str}\n"
            op_str = f"// CODELET:\t{self.op_name}{self.instance_id}\n"
            op_str += operand_str
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
            op_str['instr_len'] = self.num_instr
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
                    instr_list = f"\n".join(instr_list) + "\n"
                    op_str += instr_list
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


    def get_operand_loc_index(self, opname: str, loc, output_idx=False) -> int:
        if output_idx:
            return 0
        elif opname not in self.all_oploc_indices[loc]:
            raise RuntimeError(f"Couldnt find opname: {opname} in all op indices:\n"
                               f"Storage: {self.all_oploc_indices[loc]}")
        else:
            return self.all_oploc_indices[loc].index(opname)


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
                            f"Value: {value}\n"
                            f"Value type: {type(value)}")

    def set_required_param(self, key: str, value: int):

        value = value.value if isinstance(value, FlexParam) else value
        if key not in self.required_params:
            raise KeyError(f"Key {key} for updating param does not exist:\n"
                           f"All Keys: {self.required_params.keys()}\n"
                           f"Updated value: {value}")


        # TODO: Check back on this
        # if self.required_params[key].is_set() and self.required_params[key].value != value and\
        #         not isinstance(self.required_params[key].value, LambdaType):
        #     raise RuntimeError(f"Param {key} has already been set:\n"
        #                        f"Previous value: {self.required_params[key]}\n"
        #                        f"New value: {value}")
        if value is None:
            raise RuntimeError(f"Cannot self None value for required parameter:\n"
                               f"Value: {value}\n"
                               f"Key: {key}")
        self.required_params[key].value = value

    def is_loop_node_target(self, loop, hag_node):
        scoped_ops = self.loop_scope(loop.op_str)
        # for o in self.ops:
        for o in scoped_ops:
            if o.op_type == 'compute' and o.target == hag_node and loop.loop_level <= o.loop_level:
                return True
            elif o.op_type == 'transfer' and hag_node in o.path and loop.loop_level <= o.loop_level:
                return True
        return False

    def loop_scope(self, loopname):
        loop = self.op_map[loopname]
        start = self.ops.index(loop) + 1

        idx = start + 1
        end = None
        while idx < len(self.ops):
            o = self.ops[idx]
            if isinstance(o, LoopEnd) and o.loop_name == loopname:
                end = idx
                break
            idx += 1
        if end is None:
            raise RuntimeError(f"Uable to find loop end for {loopname}")
        scoped_ops = self.ops[start: end + 1]
        return scoped_ops

    def is_direct_loop_dep(self, loop, hag_node):
        for o in self.ops:
            if o.op_type == 'compute' and o.target == hag_node and loop.op_str in o.dependencies:
                return True
            elif o.op_type == 'transfer' and hag_node in o.path and loop.op_str in o.dependencies:
                return True
        return False

    def compute_loop_deps(self, compute_op):
        deps = self.op_map[compute_op].dependencies
        filtered_deps = []
        for d in deps:
            if self.op_map[d].op_type == "loop":
                dep_names = [l.op_str for l in self.loop_scope(d)]
                if compute_op in dep_names:
                    filtered_deps.append(d)
        return filtered_deps

    def compute_node_loops(self, node_name):
        compute_ops = self.get_ops_by_type('compute')
        loops = []
        for c in compute_ops:
            if c.target == node_name:
                loops += self.compute_loop_deps(c.op_str)
        return list(set(loops))

    def loop_node_compute(self, loop, node_name):
        scope_ops = self.loop_scope(loop.op_str)

        for c in scope_ops:
            if c.target == node_name:
                return c
        options = {c.op_str: c.target for c in scope_ops if c.op_type == "compute"}
        raise RuntimeError(f"Could not find compute node for {loop.op_str} with target {node_name}\n"
                           f"{options}")

    def loop_compute_op(self, loop, src_op=None, dst_op=None):
        assert not (src_op is not None and dst_op is not None)
        if isinstance(loop, Loop):
            loop = loop.op_str
        scope_ops = self.loop_scope(loop)
        for c in scope_ops:
            if c.op_type == "compute":
                if src_op is not None and src_op in c.sources:
                    return c
                elif dst_op is not None and dst_op in c.dests:
                    return c
                elif dst_op is None and src_op is None:
                    return c

        options = {c.op_str: c.target for c in scope_ops if c.op_type == "compute"}
        raise RuntimeError(f"Could not find compute node for {loop.op_str}. "
                           f"Src op: {src_op}\n"
                           f"Dst op: {dst_op}\n"
                           f"Possible ops:\n"
                           f"{options}\n")


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

    def create_temp_operand(self, shape_list, location,
                            reference_operand: Operand=None,
                            **kwargs):
        name = f"temp{len(self.temps)}"
        if reference_operand:
            supported_dtypes = reference_operand.supported_dtypes
            dependencies = reference_operand.dependencies
        else:
            # TODO: Infer supported dtypes somehow
            supported_dtypes = None
            dependencies = []

        # TODO: Fix check for node existance
        temp_op = Operand(name, supported_dtypes,
                          shape_list,
                          data_path=[location],
                          dependencies=dependencies,
                          **kwargs)
        self._temps.append(temp_op)
        return temp_op

    def configure(self, start_end, target_name, **kwargs):
        if 'index' in kwargs:
            assert isinstance(kwargs['index'], int)
        cfg = Configure(start_end, target_name,
                        add_codelet=False, **kwargs)
        self.add_op(cfg)
        return cfg

    def compute(self, op_name, sources, dests, **kwargs):
        comp = Compute(op_name, sources, dests,
                        add_codelet=False, **kwargs)
        self.add_op(comp)
        return comp

    def transfer(self, operand, path, sizes=None, **kwargs):
        xfer = Transfer(operand, path, sizes=sizes,
                        add_codelet=False, **kwargs)
        self.add_op(xfer)
        return xfer

    def loop(self, start, end=None, stride=1, offset=0,**kwargs):
        loop_op = Loop(start, end=end, stride=stride, offset=offset, add_codelet=False, **kwargs)
        self.add_op(loop_op)
        return loop_op

    def end_loop(self, loop_name, **kwargs):
        end_loop_op = LoopEnd(loop_name, add_codelet=False, **kwargs)
        self.add_op(end_loop_op)
        return end_loop_op

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

    def set_dim_values(self, node: pm.Node, operand: Operand):

        if not operand.is_instantiated():
            if len(operand.permutation) == len(node.shape):
                perm_map = {s: operand.permutation[s] for s in range(len(node.shape))}
            else:
                perm_map = {s: s for s in range(len(node.shape))}
            for j, s_ in enumerate(node.shape):
                key = operand.shape_list[j]
                s = node.shape[perm_map[j]]
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

    def uses_hag_node(self, node_name: str) -> bool:
        assert self.hag.has_node(node_name), f"Invalid node {node_name} for HAG"
        for o in self.ops:
            if o.op_type == "compute" and node_name == o.target:
                return True
            elif o.op_type == "transfer" and node_name in o.path:
                return True
            elif o.op_type == "config" and node_name in o.target:
                return True
        return False

    def get_tile_level(self, node_name: str):
        for i in self.tile_levels.keys():
            if node_name in self.tile_levels[i]:
                return i
        raise KeyError(f"Unable to find tile level for node {node_name}\n"
                       f"Tile keys: {self.tile_levels}")

    def get_loop_scope(self, loop_name: str):
        op_names = []
        loop = self.op_map[loop_name]
        op_idx = self.ops.index(loop) + 1
        target_level = loop.loop_level + 1
        if not USE_LOOP_END:
            while op_idx < len(self.ops) and self.ops[op_idx].loop_level >= target_level:
                if self.ops[op_idx].loop_level == target_level:
                    op_names.append(self.ops[op_idx].op_str)
                op_idx += 1
        else:
            while op_idx < len(self.ops):
                if self.ops[op_idx].op_type == "loop_end" and self.ops[op_idx].loop_name == loop_name:
                    break
                if self.ops[op_idx].loop_level == target_level:
                    op_names.append(self.ops[op_idx].op_str)
                op_idx += 1
        return op_names

    def get_combined_loop_scope(self, loop_name: str):
        op_names = []
        loop = self.op_map[loop_name]
        op_idx = self.ops.index(loop) + 1
        target_level = loop.loop_level + 1
        while op_idx < len(self.ops) and self.ops[op_idx].loop_level >= target_level:
            if self.ops[op_idx].loop_level >= target_level:
                op_names.append(self.ops[op_idx].op_str)
            op_idx += 1
        return op_names

    def get_operation_scopes(self):
        operation_scopes = {}
        for l in self.get_ops_by_type("loop"):
            ops = self.get_loop_scope(l.op_str)
            for o in ops:
                operation_scopes[o] = l.op_str
        return operation_scopes

    def get_max_loop_dep_level(self, op: Operation):
        all_deps = op.dependencies
        loop_levels = []
        for dep in all_deps:

            dep_idx = self.ops.index(self.op_map[dep])
            if self.ops[dep_idx].op_type == "loop":
                loop_levels.append(self.ops[dep_idx].loop_level + 1)

        if len(loop_levels) == 0:
            return op.loop_level
        else:
            return max(loop_levels)

    def get_loop_end(self, start_loop_name):
        for o in self.ops:
            if o.op_type == "loop_end" and o.loop_name == start_loop_name:
                return o
        raise RuntimeError(f"Unable to find end loop op for {start_loop_name}")

    def get_max_loop_dep(self, op: Operation):
        all_deps = op.dependencies
        if op.loop_level == 0:
            return None
        max_loop_level = -1
        scopes = self.get_operation_scopes()

        max_level_name = scopes[op.op_str]

        for dep in all_deps:
            dep_idx = self.ops.index(self.op_map[dep])
            if self.ops[dep_idx].op_type == "loop" and self.ops[dep_idx].loop_level > max_loop_level:
                max_loop_level = self.ops[dep_idx].loop_level
                max_level_name = self.ops[dep_idx].op_str
        return max_level_name


    def get_tile_splits(self, node_name: str):
        level = self.get_tile_level(node_name)
        return self.domain_tiling[level]

    def get_node_shape_map(self, op_tmplt: Operand, node: pm.Node) -> Dict[str, Dict]:
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

    def set_op_node_name(self, node: pm.Node, operand: Operand):
        if operand.node_name is not None and operand.node_name != node.name:
            raise RuntimeError(f"Name already set to different value for operand:\n"
                               f"Previous name: {operand.node_name}\n"
                               f"New name: {node.name}")
        operand.set_node_name(node.name)

    def instantiate_operands(self, node: pm.Node):

        for i, operand in enumerate(self.inputs):
            n = node.inputs[i]
            for rp_key in operand.required_params:
                if rp_key not in self.required_params:
                    self.add_required_param(rp_key)

            # self.set_dim_values(n, operand)
            # self.set_dtype(n, operand)
            self.set_op_node_name(n, operand)
            operand.node_name = n.name
            operand.operand_type = n.__class__.__name__

        for i, operand in enumerate(self.outputs):
            n = node.outputs[i]
            for rp_key in operand.required_params:
                if rp_key not in self.required_params:
                    self.add_required_param(rp_key)
            # self.set_dim_values(n, operand)
            # self.set_dtype(n, operand)
            self.set_op_node_name(n, operand)
            operand.node_name = n.name
            operand.operand_type = n.__class__.__name__

        # for t in self.temps:
        #     if not t.is_instantiated():
        #         for rp_key in t.required_params:
        #             if rp_key not in self.required_params:
        #                 self.add_required_param(rp_key)
        #
        #         for k in t.shape_list:
        #             if k not in self.required_params or not self.required_params[k].is_set():
        #                 raise RuntimeError(f"Shape {k} for operand {t.name} not found or not set.")
        #             t.update_shape_symbols(k, self.required_params[k].value)

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

    def all_dependencies(self, leaf_deps: List[str]):
        deps = leaf_deps.copy()
        for l in leaf_deps:
            deps += self.all_dependencies(self.op_map[l].dependencies)
        return list(set(deps))

    def is_fusion(self) -> bool:
        return len(set([c.target for c in self.ops if c.op_type == "compute"])) > 1

    def inner_stride(self, operand, loop, loop_idx):

        loop_tile_level = self.get_loop_tile_level(loop.loop_level)
        loop_param = self.loop_param_map[loop.op_str]
        level_tile_size = self.param_tiling[loop_tile_level]
        tgt_tile_size = np.prod(list(self.param_tiling[loop_tile_level + 1].values()))
        operand_tile_size = tuple([level_tile_size[s] for s in operand.shape_list])
        stride = loop.stride * np.prod(operand_tile_size[operand.shape_list.index(loop_param) + 1:], dtype=np.int32)
        stride = np.ceil(stride/tgt_tile_size).astype(np.int32)
        return stride

    def inner_iter(self, operand, loop, loop_idx):
        loop_tile_level = self.get_loop_tile_level(loop.loop_level)
        loop_param = self.loop_param_map[loop.op_str]
        tgt_tile_size = np.prod(list(self.param_tiling[loop_tile_level + 1].values()))

        rel_idx = operand.shape_list.index(loop_param) - len(operand.shape_list)
        loop_iters = loop.iter_count
        if abs(rel_idx) <=1:
            loop_iters = np.ceil(loop_iters/tgt_tile_size).astype(np.int32)
        return loop_iters

    def get_loop_tile_level(self, loop_id):
        tile_levels = len(self.domain_tiling) - 1
        loops_per_level = self.num_loops // tile_levels
        return loop_id // loops_per_level

    @property
    def num_loops(self):
        return len(self.loop_param_map)

    @property
    def derived_params(self):
        return self._derived_fps

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

        for o in self.ops:
            if o.op_type == "compute":
                deps = self.compute_loop_deps(o.op_str)
                for operand in o.operands:
                    dms = operand.get_op_accesses(o.op_str)
                    for dm in dms:
                        names = [str(i) for i in dm.symbol_str_map.keys()]
                        symbol_deps = {self.loop_param_map[d]: d for d in deps}
                        for i in names:
                            if i in self.loop_param_map and symbol_deps[self.loop_param_map[i]] != i:

                                tgt_name = symbol_deps[self.loop_param_map[i]]
                                symbol = self.op_map[tgt_name].param_symbols[tgt_name]
                                dm.update_offset_map(self.loop_param_map[i], symbol)



        for operand in self.operands:
            operand.evaluate_operand(node, hag, self)

    def get_operand_by_node_name(self, node_name):
        for o in self.operands:
            if o.node_name == node_name:
                return o
        raise KeyError(f"Unable to find operand with node name {node_name}\n"
                       f"Operand node names: {[o.node_name for o in self.operands]}")

    def get_operand_shapes(self):
        shape_dims = {}
        operands = (self.inputs + self.outputs)
        for o in operands:
            shape_dims.update(o.shape_symbols)

        return shape_dims





