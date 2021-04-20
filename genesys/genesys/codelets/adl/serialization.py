import json
from typing import Union, Any
from . import ArchitectureNode, ComputeNode, CommunicationNode, StorageNode, \
    OperandTemplate, Datatype, Instruction
from codelets.codelet_impl import Codelet
import linecache
from typing import List, Dict
from jsonschema import validate
from pathlib import Path
import numpy as np
import sympy
import math

# Need to store
_next_id = 0
def exec_with_source(src: str, globals: Dict[str, Any]):
    global _next_id
    key = f'<eval_with_key_{_next_id}>'
    _next_id += 1
    _eval_cache[key] = [line + '\n' for line in src.splitlines()]
    # exec(compile(src, key, 'exec'), globals)
    return eval(compile(src, key, 'eval'))

_eval_cache : Dict[str, List[str]] = {}
_orig_getlines = linecache.getlines
def patched_getline(*args, **kwargs):
    if args[0] in _eval_cache:
        return _eval_cache[args[0]]
    return _orig_getlines(*args, **kwargs)

def _lambda_from_src(src: str):
    gbls: Dict[str, Any] = {'inf': math.inf, 'nan': math.nan}
    return exec_with_source(src, gbls)

linecache.getlines = patched_getline


BaseArchNode = Union[ComputeNode, CommunicationNode, StorageNode]
CWD = Path(f"{__file__}").parent
JSON_SCHEMA_PATH = f"{CWD}/adl_schema.json"

def get_compute_blob(node):
    blob = {"field_name": node.field_name, "dimensions": node.dimensions}
    capabilities = []
    for name, c in node.primitives.items():
        cap_blob = {}
        cap_blob['field_name'] = c.field_name
        cap_blob['input_dimension_names'] = c.input_dimension_names
        cap_blob['output_dimension_names'] = c.output_dimension_names
        cap_blob['input_dtypes'] = c.input_dtypes
        cap_blob['input_components'] = c.input_components
        cap_blob['output_dtypes'] = c.output_dtypes
        cap_blob['output_components'] = c.output_components
        cap_blob['op_params'] = c.op_params
        cap_blob['latency'] = c.latency
        cap_blob['capability_sequence'] = []
        # TODO: Add capability_sequence
        # for sc in c.capability_sequence:
        #     cap_blob['capability_sequence'].append()
        capabilities.append(cap_blob)
    blob['primitives'] = capabilities
    return blob

def get_communication_blob(node):
    blob = {"field_name": node.field_name, "comm_type": node.get_comm_type(), "latency": node.get_latency()}
    blob['bandwidth'] = node.get_bw()
    return blob

def get_storage_blob(node):
    blob = {"field_name": node.field_name, "read_bw": node.read_bw, "write_bw": node.write_bw}
    blob["access_type"] = node.access_type
    blob["size"] = node.size
    blob["input_ports"] = node.input_ports
    blob["output_ports"] = node.output_ports
    blob["buffering_scheme"] = node.buffering_scheme
    blob["latency"] = node.latency
    return blob

def generate_hw_cfg(graph, save_path):
    json_graph = {}
    json_graph[graph.field_name] = {"compute": [], "storage": [], "communication": []}
    compute_blobs = []
    storage_blobs = []
    comm_blobs = []
    for name, node in graph._all_subgraph_nodes.items():
        if isinstance(node, ComputeNode):
            compute_blobs.append(get_compute_blob(node))
        elif isinstance(node, StorageNode):
            storage_blobs.append(get_storage_blob(node))
        elif isinstance(node, CommunicationNode):
            comm_blobs.append(get_communication_blob(node))
    json_graph[graph.field_name]["compute"] = compute_blobs
    json_graph[graph.field_name]["storage"] = storage_blobs
    json_graph[graph.field_name]["communication"] = comm_blobs

    with open(f"{save_path}", "w") as file:
        json.dump(json_graph, file, indent=4)

def serialize_graph(graph: ArchitectureNode, save_path, validate_graph=False):
    json_graph = {}
    json_graph['graph'] = graph.to_json()
    if validate_graph:
        validate_schema(json_graph)
    with open(f"{save_path}", "w") as file:
        json.dump(json_graph, file, indent=4)

    return json_graph

def deserialize_graph(filepath, validate_load=False):
    with open(filepath, "r") as graph_path:
        json_graph = json.load(graph_path)

    if validate_load:
        validate_schema(json_graph)
    if json_graph['graph']['node_type'] not in NODE_INITIALIZERS:
        raise ValueError(f"Unable to parse json object node type: {json_graph['graph']['node_type']}")
    else:
        graph = _deserialize_node(json_graph['graph'])

    return graph

def _deserialize_node(node_object):
    args = [node_object['field_name']]
    subgraph_nodes = [_deserialize_node(n) for n in node_object['subgraph']['nodes']]

    kwargs = {}
    kwargs['index'] = node_object['node_id']

    for k, v in node_object['attributes'].items():
        if k == "primitives":
            kwargs[k] = _deserialize_capabilities(v)
        elif k == "codelets":
            assert "primitives" in kwargs
            kwargs[k] = _deserialize_codelets(kwargs['primitives'], v)
        else:
            kwargs[k] = v

    node = NODE_INITIALIZERS[node_object['node_type']](*tuple(args), **kwargs)

    for sn in subgraph_nodes:
        node.add_subgraph_node(sn)

    for e in node_object['subgraph']['edges']:
        node.add_subgraph_edge(e['src'], e['dest'], **e['attributes'])

    return node

# add extra_params
def _deserialize_capabilities(capability_list: List[Dict]):
    caps = []
    for c in capability_list:
        kwargs = {}
        kwargs['target'] = c['target']
        kwargs['latency'] = c['latency']
        kwargs['source'] = _deserialize_operands(c['source'])
        for k, v in c['extra_params'].items():
            kwargs[k] = v
        cap = Instruction(c['target_name'], c['opcode'], c['opcode_width'], **kwargs)
        caps.append(cap)
    return caps

# Add compoennts, index size to source
def _deserialize_operands(op_list):
    operands = []
    # for o in op_list:
    #     if o['field_name'] == "null":
    #         op = NullOperand(o['bitwidth'], 0)
    #     else:
    #         args = (o['field_name'], o['field_type'], _deserialize_dtypes(o['supported_dtypes']), o['bitwidth'])
    #         kwargs = {}
    #         kwargs['value_names'] = o['possible_values']
    #         kwargs['index_size'] = o['index_size']
    #         kwargs['components'] = o['components']
    #         op = Operand(*args, **kwargs)
    #     operands.append(op)
    return operands

def _deserialize_dtypes(dtypes):
    return [DTYPE_MAP[dt] for dt in dtypes]

def _deserialize_codelets(capabilities: List[Instruction], codelet_list):
    cap_map = {c.name: c for c in capabilities}
    codelets = []
    for cdlt_obj in codelet_list:
        inputs = [OperandTemplate.from_json(ipt_obj) for ipt_obj in cdlt_obj['source']]
        outputs = [OperandTemplate.from_json(opt_obj) for opt_obj in cdlt_obj['dest']]

        args = (cdlt_obj['codelet_name'],)
        kwargs = {}
        kwargs['source'] = inputs
        kwargs['dest'] = outputs
        kwargs['latency'] = cdlt_obj['latency']
        op_params = {}
        for k, v in cdlt_obj['op_params'].items():
            if v['type'] == 'function':
                op_params[k] = _lambda_from_src(v['value'])
            else:
                op_params[k] = v['value']
        kwargs['op_params'] = op_params
        templates = []
        for cap_obj in cdlt_obj['capability_sequence']:
            assert cap_obj['target_name'] in cap_map
            cap_tmp = cap_map[cap_obj['target_name']].create_template()
            for of in cap_obj['op_fields']:
                if of['field_name'] == "null":
                    continue
                if isinstance(of['value'], str):
                    cap_tmp.set_field_by_name(of['field_name'], of['value'])
                elif isinstance(of['value'], int):
                    cap_tmp.set_field_value(of['field_name'], of['value'])
                elif of['value'] is not None:
                    raise TypeError(f"Invalid type value for field {of['field_name']}: {type(of['value'])}")
            templates.append(cap_tmp)
        kwargs['capability_sequence'] = templates
        cdlt = Codelet(*args, **kwargs)
        codelets.append(cdlt)
    return codelets

def validate_schema(json_graph):
    with open(JSON_SCHEMA_PATH, "r") as schema_file:
        schema = json.load(schema_file)
    validate(json_graph, schema=schema)



NODE_INITIALIZERS = {
    "ComputeNode": ComputeNode,
    "StorageNode": StorageNode,
    "CommunicationNode": CommunicationNode
}

DTYPE_MAP = {
    "FXP8": Datatype(type="FXP", bitwidth=8),
    "FXP16": Datatype(type="FXP", bitwidth=16),
    "FXP32": Datatype(type="FXP", bitwidth=32),
    "FP16": Datatype(type="FP", bitwidth=16),
    "FP32": Datatype(type="FP", bitwidth=32),

}
