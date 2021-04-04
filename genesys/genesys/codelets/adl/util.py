from types import FunctionType
import inspect
import copy
from operator import mul
import itertools
import polymath as pm
from functools import reduce
from dataclasses import dataclass, field
from typing import Dict, List, Any
import sympy

@dataclass
class ParamFunc:
    func_header: str
    func_body: str
    arg_names: List[str]
    value: Any = field(default=None)

    @property
    def function_str(self):
        arg_str = ", ".join(self.arg_names)
        return f"{self.func_header}({arg_str}):\n\t{self.func_body}"

    def get_param_fnc(self):
        param_fnc_code = compile(self.function_str, "<string>", "exec")
        param_fnc = FunctionType(param_fnc_code.co_consts[1], globals(), self.func_header)
        return param_fnc, self.function_str

    def run_param_fnc(self, *args, **kwargs):
        param_fnc, param_func_str = self.get_param_fnc()

        try:
            result = param_fnc(*args, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Error while trying to execute param func:\n"
                               f"Func: {param_func_str}\n"
                               f"Error: {e}")
        return result

    def set_value_from_param_fn(self, *args):
        self.value = self.run_param_fnc(*args)


def lambda_to_str(fn):
    fn_str = str(inspect.getsourcelines(fn)[0])
    fn_str = fn_str.strip("['\\n']").split(" = ")[1]
    return fn_str

def is_lambda(param) -> bool:
    if isinstance(param, FunctionType) and param.__code__.co_name == "<lambda>":
        return True
    return False

def loop_tile_with_hint(tile_permutations, loop_extent, num_level, loop_hint):
    # TODO support more than 1 level of para hint
    for level in range(num_level):
        if loop_hint[level] != None:
            loop_hint_level = level
            break

    blocking_hint = 1 if loop_hint[loop_hint_level][1] == None else loop_hint[loop_hint_level][1]
    assert loop_hint[loop_hint_level][2]
    para_hint = loop_hint[loop_hint_level][2]
    # para_hint = 1 if loop_hint[loop_hint_level][2] == None else loop_hint[loop_hint_level][2]
    blocking_factor = blocking_hint * para_hint

    pre_tile_permutations = []
    if loop_hint_level == 0:
        pre_tile_permutations.append([])
    else:
        for sub_extent in factors((loop_extent + blocking_factor - 1) // blocking_factor):
            recursive_tile(pre_tile_permutations, [], sub_extent, 0, loop_hint_level)

    for pre_tile in pre_tile_permutations:
        # TODO support not fixed blocking hint
        if loop_hint[loop_hint_level][1]:
            pre_tile.append(blocking_factor)
            blocking_accum = reduce(mul, pre_tile, 1)
            recursive_tile(tile_permutations, pre_tile, (loop_extent + blocking_accum - 1) // blocking_accum,
                           loop_hint_level + 1, num_level)
        else:
            blocking_accum = reduce(mul, pre_tile, 1)
            for i in factors((loop_extent + blocking_accum - 1) // blocking_accum):
                if i >= para_hint:
                    new_pre_tile = copy.copy(pre_tile)
                    new_pre_tile.append(i)
                    new_blocking_accum = blocking_accum * i
                    recursive_tile(tile_permutations, new_pre_tile,
                                   (loop_extent + new_blocking_accum - 1) // new_blocking_accum, loop_hint_level + 1,
                                   num_level)

def factors(n):
    return set(reduce(list.__add__,
                      ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)))

def recursive_tile(tile_permutations, curr_loop_tile, n, curr_level, num_level):
    if curr_level == num_level - 1:
        curr_loop_tile.append(n)
        tile_permutations.append(curr_loop_tile)
        return

    for i in factors(n):
        new_loop_tile = copy.copy(curr_loop_tile)
        new_loop_tile.append(i)
        recursive_tile(tile_permutations, new_loop_tile, n // i, curr_level + 1, num_level)

def tile_perms(dimensions, num_levels, schedule=None):

    hint = schedule.schedule_hint if schedule else None

    all_tile_permutations = []
    for i, d in enumerate(dimensions):
        loop_hint = hint[i] if hint and i in hint else None
        all_tile_permutations.append(loop_tile(d, num_levels, loop_hint))
    tiles = list(itertools.product(*all_tile_permutations))
    # for tile in itertools.product(*all_tile_permutations):
        # TODO here the generated is a list of lists, not a list of tuples
        # if cost_model.valid_blocking_size(resource, dummy_mapping_point, layer):
        # yield list(tile)
    return tiles


def loop_tile(loop_extent, num_level, loop_hint=None):
    tile_permutations = []
    if not loop_hint:
        recursive_tile(tile_permutations, [], loop_extent, 0, num_level)
    else:
        loop_tile_with_hint(tile_permutations, loop_extent, num_level, loop_hint)

    return tile_permutations

def get_node_location(node: pm.Node) -> str:
    if "component_location" not in node.added_attrs:
        node.add_attribute("component_location", "DRAM")
        location = "DRAM"
    else:
        location = node.kwargs["component_location"]
    return location

# TODO: Fix this
def get_node_dtype(node, hag):
    return "fxp32"

def get_lambda_source(lambda_func):
    import ast
    import os
    """Return the source of a (short) lambda function.
    If it's impossible to obtain, returns None.
    """
    try:
        source_lines, _ = inspect.getsourcelines(lambda_func)
    except (IOError, TypeError):
        print(f"IO/Type Error")
        return None

    # skip `def`-ed functions and long lambdas
    if len(source_lines) != 1:
        return None

    source_text = os.linesep.join(source_lines).strip()

    # find the AST node of a lambda definition
    # so we can locate it in the source code
    source_ast = ast.parse(source_text)
    lambda_node = next((node for node in ast.walk(source_ast)
                        if isinstance(node, ast.Lambda)), None)
    if lambda_node is None:  # could be a single line `def fn(x): ...`
        return None

    # HACK: Since we can (and most likely will) get source lines
    # where lambdas are just a part of bigger expressions, they will have
    # some trailing junk after their definition.
    #
    # Unfortunately, AST nodes only keep their _starting_ domain_offsets
    # from the original source, so we have to determine the end ourselves.
    # We do that by gradually shaving extra junk from after the definition.
    lambda_text = source_text[lambda_node.col_offset:]
    lambda_body_text = source_text[lambda_node.body.col_offset:]
    min_length = len('lambda:_')  # shortest possible lambda expression
    while len(lambda_text) > min_length:
        try:
            # What's annoying is that sometimes the junk even parses,
            # but results in a *different* lambda. You'd probably have to
            # be deliberately malicious to exploit it but here's one way:
            #
            #     bloop = lambda x: False, lambda x: True
            #     get_short_lamnda_source(bloop[0])
            #
            # Ideally, we'd just keep shaving until we get the same code,
            # but that most likely won't happen because we can't replicate
            # the exact closure environment.
            code = compile(lambda_body_text, '<unused filename>', 'eval')

            # Thus the next best thing is to assume some divergence due
            # to e.g. LOAD_GLOBAL in original code being LOAD_FAST in
            # the one compiled above, or vice versa.
            # But the resulting code should at least be the same *length*
            # if otherwise the same operations are performed in it.
            if len(code.co_code) == len(lambda_func.__code__.co_code):
                return lambda_text
        except SyntaxError:
            pass
        lambda_text = lambda_text[:-1]
        lambda_body_text = lambda_body_text[:-1]

    return None

def split(expr, variables):
    """Split affine, linear, and nonlinear part of expr w.r.t. variables."""
    if isinstance(expr, float):
        return expr, 0, 0

    input_is_list = True
    if not isinstance(variables, list):
        input_is_list = False
        variables = [variables]

    # See <https://github.com/sympy/sympy/issues/11475> on why we need expand() here.
    expr = expr.expand()

    # Get the affine part by removing all terms with any of the variables.
    affine = expr
    for var in variables:
        affine = affine.coeff(var, n=0)

    # Extract the linear coefficients by extracting the affine parts of the derivatives.
    linear = []
    for var in variables:
        d = sympy.diff(expr, var)
        for var2 in variables:
            d = d.coeff(var2, n=0)
        linear.append(d)

    # The rest is nonlinear
    nonlinear = expr - affine
    for var, coeff in zip(variables, linear):
        nonlinear -= var * coeff
    nonlinear = sympy.simplify(nonlinear)

    if not input_is_list:
        assert len(linear) == 1
        linear = linear[0]

    return affine, linear, nonlinear