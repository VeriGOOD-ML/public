from typing import Dict
from .codelets.reference_impls.ref_op import OperandData
from codelets.codelet_impl import Codelet
from codelets.compiler.program import CodeletProgram
import argparse
import numpy as np
import os
from pathlib import Path
import json
from .genesys import get_arch
BENCH_BASE_ADDR = {"INSTR": 0, "OBUF": 0, "BBUF": 4096, "WBUF": 24576, "IBUF": 4259840}



def save_array(path, data):
    with open(path, 'w') as f:
        f.write('\n'.join([str(i) for i in data.flatten().tolist()]))

OUTPUT_TYPES = ["arch_cfg", "operations_idx", "json", "string_final", "decimal", "binary"]
OUT_DIR = Path(f"{Path(__file__).parent}/../../tools/compilation_output")

class DataGen(object):
    def __init__(self, program,
                 single_codelets=True,
                 dir_ext=None,
                 identifier=None,
                 output_types=None,
                 generate_data=False,
                 verbose=False,
                 store_partials=False,
                 store_whole_program=False,
                 propagate_outputs=False):
        self.store_whole_program = store_whole_program
        self._storage_info = {}
        self._propagate_outputs = propagate_outputs
        self._program = program
        self._inouts = {"inputs": [], "outputs": []}
        self._value_dict : Dict[str, Dict[str, OperandData]] = {"inputs": {},
                  "intermediate": {},
                  "outputs": {}}
        self._store_partials = store_partials
        self._single_codelets = single_codelets
        self._verbose = verbose
        self._generate_data = generate_data
        self.output_types = output_types
        if self.output_types is not None:
            assert all([o in OUTPUT_TYPES for o in output_types])
        else:
            self.output_types = OUTPUT_TYPES

        output_dir = f"{OUT_DIR}/{program.name}"
        if dir_ext:
            output_dir = f"{output_dir}_{dir_ext}"
        self.output_dir = f"{output_dir}_{identifier}"
        self.arch_cfg = get_arch(None, self.program.hag.meta_cfg, None)

        if not Path(self.output_dir).exists():
            try:
                os.makedirs(self.output_dir)
            except OSError as e:
                print(f"Creation of directory {self.output_dir} failed:\n {e}")
            else:
                print(f"Successfully created of directory {self.output_dir}")

    @property
    def single_codelets(self):
        return self._single_codelets

    @property
    def storage_info(self):
        return self._storage_info

    @property
    def propagate_outputs(self):
        return self._propagate_outputs

    @property
    def program(self) -> CodeletProgram:
        return self._program

    @property
    def verbose(self):
        return self._verbose

    @property
    def generate_data(self):
        return self._generate_data

    @property
    def value_dict(self):
        return self._value_dict

    @property
    def inouts(self):
        return self._inouts

    def initialize_value_dict(self, cdlt):
        inouts = {"inputs": [], "outputs": []}
        for i in cdlt.inputs:
            if i.node_name in self.value_dict['outputs']:
                operand = self.value_dict['outputs'].pop(i.node_name)
                assert isinstance(operand, OperandData), f"Not operand: {operand.name}"
                assert operand.data.shape == i.shape, "Operand and input shapes are not equal:\n" \
                                                      f"Data: {operand.data.shape}\n" \
                                                      f"Operand: {i.shape}"
                inouts['inputs'].append(operand)
                self.value_dict['intermediate'][i.node_name] = operand
            elif i.node_name in self.value_dict['intermediate']:
                operand = self.value_dict['intermediate'][i.node_name]
                assert operand.data.shape == i.shape, "Operand and input shapes are not equal:\n" \
                                                      f"Data: {operand.data.shape}\n" \
                                                      f"Operand: {i.shape}"
                inouts['inputs'].append(operand)
        return inouts


    def store_inputs(self, base_path, inouts):
        # for n, i in self.value_dict['inputs'].items():
        for inp in inouts['inputs']:
            # node_name = i.node_name.replace("/", "_")
            if inp.node_name in self.value_dict['inputs']:
                i = self.value_dict['inputs'][inp.node_name]
            elif inp.node_name in self.value_dict['intermediate']:
                i = self.value_dict['intermediate'][inp.node_name]
            else:
                raise RuntimeError(f"No value found found for cdlt operand: {inp.node_name}\n" \
                                                   f"UID: {inp.node_name}\n" \
                                                   f"Value inputs: {list(self.value_dict['inputs'].keys())}\n"
                                   f"Value intermediates: {list(self.value_dict['intermediate'].keys())}\n")

            assert isinstance(i, OperandData)
            assert isinstance(i.data, np.ndarray)

            node_name = i.node_name

            assert node_name in self.storage_info, f"No storage info found for cdlt operand: {node_name}\n" \
                                                   f"UID: {i.node_name}\n" \
                                                   f"Storage keys: {list(self.storage_info.keys())}"

            if Path(f"{base_path}/{node_name}").exists():
                save_array(f'{base_path}/{node_name}/{node_name}.txt', i.data)
                self.storage_info[node_name]['path'] = f'{base_path}/{node_name}/'
            else:
                save_array(f'{base_path}/{node_name}.txt', i.data)
                self.storage_info[node_name]['path'] = f'{base_path}/{node_name}.txt'

    def initialize_storage(self, cdlt, inouts):
        formatted = []
        assert all([isinstance(i, OperandData) for i in inouts['inputs']])
        assert all([isinstance(o, OperandData) for o in inouts['outputs']])
        for i in inouts['inputs']:
            if i.fmt is None and i.node_name not in self.value_dict['inputs'] and \
                    i.node_name not in self.value_dict['intermediate']:
                self.value_dict['inputs'][i.node_name] = i
                node_name = i.node_name
                self.storage_info[node_name] = {"cdlt": cdlt.cdlt_uid,
                                                "path": None,
                                                'cdlt_name': i.idx.name,
                                                'operand_type': 'input'}

            elif i.fmt is not None:
                formatted.append(i)
        return formatted

    def store_formatted(self, formatted, base_path):
        for f in formatted:
            if f.node_name in self.value_dict['inputs']:
                assert f.fmt is not None
                node_name = f.node_name
                if not Path(f"{base_path}/{node_name}").exists():
                    os.makedirs(f"{base_path}/{node_name}")
                save_array(f'{base_path}/{node_name}/{node_name}_{f.fmt}.txt', f.data)

    def store_outputs(self, cdlt, inouts, base_path):

        for idx in range(len(inouts['outputs'])):
            o = inouts['outputs'][idx]
            if o.fmt is None:
                node_name = o.node_name
                assert isinstance(o, OperandData)
                assert isinstance(o.data, np.ndarray)
                if o.data.dtype != np.int64:
                    # o.data = o.data.astype(np.int64)
                    o = o._replace(data=o.data.astype(np.int64))
                    inouts['outputs'][idx] = o

                self.value_dict['outputs'][node_name] = o
                self.storage_info[node_name] = {"cdlt": cdlt.cdlt_uid,
                                             "path": None,
                                             'cdlt_name': o.idx.name,
                                             'operand_type': 'output'}


        for o in inouts['outputs']:

            node_name = o.node_name
            assert node_name in self.storage_info, f"No storage info found for output cdlt operand: {node_name}\n" \
                                                   f"UID: {o.node_name}\n" \
                                                   f"Storage keys: {list(self.storage_info.keys())}"

            if Path(f"{base_path}/{node_name}").exists():
                save_array(f'{base_path}/{node_name}/{node_name}.txt', o.data)
                self.storage_info[node_name]['path'] = f'{base_path}/{node_name}/'
            else:
                save_array(f'{base_path}/{node_name}.txt', o.data)
                self.storage_info[node_name]['path'] = f'{base_path}/{node_name}.txt'

    def generate_cdlt_data(self, cdlt: Codelet, base_path):
        inouts = self.initialize_value_dict(cdlt)
        opgen = self.program.metadata['GENESYS_IMPLS'][cdlt.op_name](cdlt, self.program)
        inouts = opgen.compute_outputs(inouts)
        formatted = self.initialize_storage(cdlt, inouts)
        self.store_inputs(base_path, inouts)
        self.store_outputs(cdlt, inouts, base_path)
        self.store_formatted(formatted, base_path)


        with open(f"{base_path}/data_locations.json", "w") as outf:
            outf.write(json.dumps(self.storage_info, indent=2))

        if self.single_codelets:
            self.storage_info.clear()
            self.reset_value_dict()

        assert all([isinstance(i.data, np.ndarray) for i in self.value_dict['inputs'].values()])
        assert all([isinstance(o.data, np.ndarray) for o in self.value_dict['outputs'].values()])
        assert all([isinstance(i.data, np.ndarray) for i in self.value_dict['intermediate'].values()])

    def reset_value_dict(self):
        self._value_dict: Dict[str, Dict[str, OperandData]] = {"inputs": {},
                                                               "intermediate": {},
                                                               "outputs": {}}

    def generate_codelet_data(self):
        for layer_id, cdlt in enumerate(self.program.codelets):
            if self.verbose:
                print(f"Storing codelet {cdlt.cdlt_uid}")
            output_location = f"{self.output_dir}/layer{layer_id}_{cdlt.cdlt_uid}"
            if not Path(output_location).exists():
                try:
                    os.makedirs(output_location)
                except OSError as e:
                    raise RuntimeError(f"Creation of directory {output_location} failed:\n {e}")

            if 'operations_idx' in self.output_types:
                otype = 'operations_idx'
                ext = 'txt'
                res = self.program.emit_codelet_as_program(cdlt.instance_id, otype)
                with open(f"{output_location}/{cdlt.cdlt_uid}_{otype}.{ext}", "w") as outfile:
                    outfile.write(res)

            if 'string_final' in self.output_types:
                otype = 'string_final'
                ext = 'txt'
                res = self.program.emit_codelet_as_program(cdlt.instance_id, otype)
                with open(f"{output_location}/{cdlt.cdlt_uid}_{otype}.{ext}", "w") as outfile:
                    outfile.write(res)

            if 'decimal' in self.output_types:
                otype = 'decimal'
                ext = 'txt'
                res = self.program.emit_codelet_as_program(cdlt.instance_id, otype)
                with open(f"{output_location}/{cdlt.cdlt_uid}_{otype}.{ext}", "w") as outfile:
                    outfile.write(res)

            if 'binary' in self.output_types:
                otype = 'binary'
                ext = 'txt'
                res = self.program.emit_codelet_as_program(cdlt.instance_id, otype)
                with open(f"{output_location}/{cdlt.cdlt_uid}_{otype}.{ext}", "w") as outfile:
                    outfile.write(res)

            if 'json' in self.output_types:
                otype = 'json'
                ext = 'json'
                res = self.program.emit_codelet_as_program(cdlt.instance_id, otype)
                res = json.dumps(res, indent=2)
                with open(f"{output_location}/{cdlt.cdlt_uid}_{otype}.{ext}", "w") as outfile:
                    outfile.write(res)

            if self.generate_data:
                base_path = f"{output_location}/data"
                if not Path(base_path).exists():
                    try:
                        os.makedirs(base_path)
                    except OSError as e:
                        raise RuntimeError(f"Creation of directory {output_location} failed:\n {e}")

                self.generate_cdlt_data(cdlt, base_path)
                if self.verbose:
                    print(f"Generating data to be stored in {base_path}")




    def store_program(self):
        if self.verbose:
            print(f"Storing program {self.program.name}")
        output_location = f"{self.output_dir}/program"
        if not Path(output_location).exists():
            try:
                os.makedirs(output_location)
            except OSError as e:
                raise RuntimeError(f"Creation of directory {output_location} failed:\n {e}")

        if 'operations_idx' in self.output_types:
            otype = 'operations_idx'
            ext = 'txt'
            res = self.program.emit(otype)
            with open(f"{output_location}/{self.program.name}_{otype}.{ext}", "w") as outfile:
                outfile.write(res)

        if 'string_final' in self.output_types:
            otype = 'string_final'
            ext = 'txt'
            res = self.program.emit(otype)
            with open(f"{output_location}/{self.program.name}_{otype}.{ext}", "w") as outfile:
                outfile.write(res)

        if 'decimal' in self.output_types:
            otype = 'decimal'
            ext = 'txt'
            res = self.program.emit(otype)
            with open(f"{output_location}/{self.program.name}_{otype}.{ext}", "w") as outfile:
                outfile.write(res)

        if 'binary' in self.output_types:
            otype = 'binary'
            ext = 'txt'
            res = self.program.emit(otype)
            with open(f"{output_location}/{self.program.name}_{otype}.{ext}", "w") as outfile:
                outfile.write(res)

        if 'json' in self.output_types:
            otype = 'json'
            ext = 'json'
            res = self.program.emit(otype)
            res = json.dumps(res, indent=2)
            with open(f"{output_location}/{self.program.name}_{otype}.{ext}", "w") as outfile:
                outfile.write(res)

        if self.generate_data:
            base_path = f"{output_location}/data"
            if not Path(base_path).exists():
                try:
                    os.makedirs(base_path)
                except OSError as e:
                    raise RuntimeError(f"Creation of directory {output_location} failed:\n {e}")
            self.generate_whole_program_data(base_path)

    def generate_whole_program_data(self, base_path):
        pass

    def generate(self):
        if 'arch_cfg' in self.output_types:
            assert self.arch_cfg is not None
            self.arch_cfg['IBUF_END'] = int(BENCH_BASE_ADDR['IBUF'] + np.prod(self.program.codelets[0].inputs[0].shape))
            res = json.dumps(self.arch_cfg, indent=2)
            with open(f"{self.output_dir}/{self.program.name}_arch_cfg.json", "w") as outfile:
                outfile.write(res)

        self.generate_codelet_data()

        if self.store_whole_program:
            self.store_program()