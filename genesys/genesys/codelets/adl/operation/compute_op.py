from . import Operation, Operand, IndexedOperandTemplate
from typing import List, Union


class Compute(Operation):

    def __init__(self, op_name: str,
                 sources: List[Union[Operand, IndexedOperandTemplate]],
                 dests: List[Union[Operand, IndexedOperandTemplate]],
                 target: str=None,
                 add_codelet=True,
                 **kwargs):
        self._op_name = op_name
        self._sources = []
        self._dests = []
        self._operand_indices = []
        self.oploc_memo = {}
        req_params = {}
        assert target is not None

        # TODO: Need to figure out if these need to be added
        # TODO: Remove these checks during copy
        dependencies = []


        super(Compute, self).__init__('compute', req_params,
                                      target=target,
                                      add_codelet=add_codelet,
                                      dependencies=dependencies,
                                      **kwargs)

        for s_call in sources:
            if isinstance(s_call, IndexedOperandTemplate):
                self._operand_indices += s_call.atomic_loop_offsets
            s = s_call.add_compute_access(target, self.op_str, "source")

            self._dependencies += [dep for dep in s.dependencies if dep not in dependencies and dep != self.op_str]
            self._sources.append(s)


        for d_call in dests:
            if isinstance(d_call, IndexedOperandTemplate):
                self._operand_indices += d_call.atomic_loop_offsets
            d = d_call.add_compute_access(target, self.op_str, "dest")
            self._dependencies += [dep for dep in d.dependencies if dep not in dependencies and dep != self.op_str]
            d.dependencies.append(self.op_str)
            self._dests.append(d)

            if "temp" in d.name and all("transfer" not in dep for dep in d.dependencies):
                for s in self.sources:
                    if s.shape_list == d.shape_list:
                        loop_deps = [dep for dep in s.dependencies if "loop" in dep]
                        d.dependencies = list(set(d.dependencies + loop_deps))
                        break
        self._operand_indices = list(set(self._operand_indices))

    @property
    def operand_indices(self) -> List[str]:
        return self._operand_indices

    def source_names(self):
        return [s.name for s in self.sources]

    def dest_names(self):
        return [d.name for d in self.dests]

    @property
    def sources(self):
        return self._sources

    @property
    def dests(self):
        return self._dests

    @property
    def operands(self):
        return self._sources + self._dests

    @property
    def unique_operands(self):
        ops = []
        for o in self.operands:
            if o not in ops:
                ops.append(o)
        return ops

    @property
    def op_name(self):
        return self._op_name

    @property
    def num_loop_dependencies(self):
        count = 0
        for d in self.dependencies:
            if 'loop' in d:
                count += 1
        return count


    @property
    def unique_operand_locations(self) -> List[str]:
        return list(sorted(list(set([self.get_operand_location(o.name) for o in self.operands]))))

    @property
    def operands_by_unique_location(self):
        keys = []
        operands = []
        for i, o in enumerate(self.operands):
            loc = self.get_operand_location(o.name)
            idx = o.get_mem_index(loc)
            key = (loc, idx)
            if key not in keys:
                keys.append(key)
                operands.append(o)
        return operands

    @property
    def source_locations(self):
        return list(sorted(list(set([self.get_operand_location(o.name) for o in self.sources]))))

    def update_operand_indices(self, dep_map):
        for i, idx in enumerate(self.operand_indices):
            if idx in dep_map:
                self._operand_indices[i] = dep_map[idx]

    def get_operand(self, name):
        op = None
        for o in self.operands:
            if o.name == name:
                op = o
                break
        assert op is not None
        return op


    def get_operand_location(self, operand_name: str) -> str:

        op = self.get_operand(operand_name)
        location = None
        for a in op.data_moves:
            if a.op_name == self.op_str:
                if a.src_node == self.target:
                    location = a.dst_node
                else:
                    assert a.dst_node == self.target
                    location = a.src_node
                break
        assert location is not None
        return location

    def get_src_movement(self, src_name):
        source = None
        for s in self.sources:
            if s.name == src_name:
                source = s
                break
        # TODO: Add message
        if source is None:
            raise KeyError(f"Cannot find source for {src_name}")
        accesses = source.get_op_accesses(self.op_str)
        for a in accesses:
            if a.dst_node == self.target:
                return a
        raise KeyError(f"Cannot find access for {self.target}")

    def get_dest_movement(self, dest_name):
        dest = None
        for d in self.dests:
            if d.name == dest_name:
                dest = d
                break
        # TODO: Add message
        if dest is None:
            raise KeyError
        accesses = dest.get_op_accesses(self.op_str)
        for a in accesses:
            if a.src_node == self.target:
                return a
        raise KeyError

    def get_offset(self, op_name):
        if op_name in self.dest_names():
            return self.get_dst_offset(op_name)
        else:
            return self.get_src_offset(op_name)

    def get_offset_loops(self, op_name):
        offsets = self.get_offset(op_name)
        names = []
        for o in offsets:
            names.append(f'loop{o.loop_id}')
        return names

    def get_dst_offset(self, dst_name, placeholder=None):
        return self.get_dest_movement(dst_name).domain_offsets()

    def get_src_offset(self, src_name, placeholder=None):
        return self.get_src_movement(src_name).domain_offsets()

    def op_type_params(self):
        op_params = [f"OP: {self.op_name}", f"SRC: {self.sources}", f"DST: {self.dests}"]
        return op_params

    def op_type_args_copy(self, cdlt):
        sources = [cdlt.get_operand(s.name) for s in self.sources]
        dests = [cdlt.get_operand(d.name) for d in self.dests]

        return (self.op_name, sources, dests)

    def evaluate_parameters(self, node, hag, cdlt):
        pass
        # for d in self.sources:
        #     print(d.operand.tiling)
        #     print(d.operand.evaluated_tiling)
        #     print()
        # for d in self.dests:
        #     print(d.operand.tiling)
        #     print(d.operand.evaluated_tiling)
        # for s in self.sources:
        #     path_key = (s.data_source, self.target)
        #     src_shape, dst_shape = get_transfer_dim_sizes(s, path_key)
        #
        # for d in self.dests:
        #     path_key = (self.target)
        #     src_shape, dst_shape = get_transfer_dim_sizes(d, path_key)

    def emit(self, output_type):
        # TODO: Add template
        if output_type == "operations":
            source_names = [s.name for s in self.sources]
            dst_names = [d.name for d in self.dests]
            op_str = f"{self.op_str}: {self.target}-{self.op_name}({source_names})->{dst_names}"
        elif output_type == "json":
            sources = [s.name for s in self.sources]
            dests = [d.name for d in self.dests]
            op_str = {"op_type": self.op_type,
                      "op_id": self.global_op_id,
                      "operation_name": self.op_name,
                      "target": self.target,
                      "sources": sources,
                      "destinations": dests
                      }
        else:
            op_str = []
            for ft in self.instructions:
                ft_out = ft.emit(output_type)
                op_str += ft_out
        return op_str

    def copy(self, cdlt, op_name=None, sources=None, dests=None, **kwargs):
        obj = super(Compute, self).copy(cdlt, **kwargs)

        obj._op_name = op_name or self.op_name
        obj._sources = sources or [cdlt.get_operand(s.name) for s in self.sources]
        obj._dests = dests or [cdlt.get_operand(d.name) for d in self.dests]
        return obj

