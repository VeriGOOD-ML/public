
from typing import Dict
from numbers import Integral
from sympy import Basic, Idx, Integer
from itertools import tee


def get_transfer_dim_sizes(operand, path_key):
    src_shape = None
    dst_shape = None
    for i, p in enumerate(operand.evaluated_tiling[:-1]):
        # TODO: FIx this to validate against path key
        if p[0] == path_key[0] and operand.evaluated_tiling[i + 1][0] == path_key[1]:
            src_shape = p[1]
            dst_shape = operand.evaluated_tiling[i + 1][1]
            break

    # if src_shape is None or not all([isinstance(i, Integral) for i in src_shape]) or len(src_shape) == 0:
    #     raise RuntimeError
    #
    # if dst_shape is None or not all([isinstance(i, Integral) for i in dst_shape]) or len(dst_shape) == 0:
    #     raise RuntimeError

    return src_shape, dst_shape

def evaluate_offset(expr: Basic, values: Dict[str, int]):
    for f_sym in list(expr.free_symbols):
        if str(f_sym) in values:
            expr = expr.subs(f_sym, values[str(f_sym)])
    if not isinstance(expr, (Integer, Integral)):
        raise TypeError(f"Unable to compute domain domain_offsets because offset is not an integer:"
                        f"Offset: {expr}\tType: {type(expr)}")
    return int(expr)


def size_from_offsets(cdlt, domain_offsets):
    sizes = []

    for o in domain_offsets:
        if isinstance(o, Basic):
            indices = list(o.atoms(Idx))
            others = [i for i in list(o.free_symbols) if i not in indices]
            max_vals = {str(i): cdlt.op_map[str(i)].stride - 1 for i in indices}
            max_vals.update({str(i): cdlt.required_params[str(i)].value for i in others})
            size = evaluate_offset(o, max_vals) + 1
        else:
            size = o
        sizes.append(size)
    return sizes


def size_from_extent(cdlt, domain_offsets):
    sizes = []

    for o in domain_offsets:
        if isinstance(o, Basic):
            indices = list(o.atoms(Idx))
            others = [i for i in list(o.free_symbols) if i not in indices]
            max_vals = {str(i): cdlt.op_map[str(i)].end - 1 for i in indices}
            max_vals.update({str(i): cdlt.required_params[str(i)].value for i in others})
            size = evaluate_offset(o, max_vals) + 1
        else:
            size = o
        sizes.append(size)
    return sizes

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)