from codelets.templates.codelet_template import CodeletTemplate
from codelets.templates.operand_template import IndexOperandTemplate
from examples.genesys import FXP_CONFIGS
from fxpmath import Fxp

CAST_FUNC = lambda x, dtype: Fxp(x, **FXP_CONFIGS[dtype]).val.item()

def range_from_cfg(cfg, as_int=True):
    if cfg['signed']:
        upper_val = (1 << (cfg['n_word'] - 1)) - 1
        lower_val = -upper_val - 1
    else:
        upper_val = (1 << cfg['n_word']) - 1
        lower_val = 0

    if not as_int:
        upper_val = upper_val / 2.0 ** cfg['n_frac']
        lower_val = lower_val / 2.0 ** cfg['n_frac']

    return (lower_val, upper_val)


def create_immediate_with_operand(cdlt: CodeletTemplate, key: str, value, simd_size, cast_float_to_fxp=False):
    # idx = len(cdlt.temps)
    dtype = None
    if isinstance(value, float) and cast_float_to_fxp:
        value = CAST_FUNC(value, 'FXP32')
        dtype = 'FXP32'
    dummy_value = cdlt.dummy_op(key, value, dtype=dtype)
    op = cdlt.create_temp_operand([simd_size], "IMM", name=key)
    cdlt.configure("start", "IMM", immediate_value=dummy_value)
    return op


def add_sys_array_cast(cdlt):
    pass

def add_scale_op(cdlt: CodeletTemplate, input_op, output_op, multiplier_op, shift_op, indices):
    assert not isinstance(input_op, IndexOperandTemplate)
    assert not isinstance(output_op, IndexOperandTemplate)
    cdlt.compute("MUL", [input_op[indices], multiplier_op], [output_op[indices]], target="SIMD")
    cdlt.compute("RSHIFT", [output_op[indices], shift_op], [output_op[indices]], target="SIMD")

def add_scale_and_cast_op(cdlt: CodeletTemplate, input_op, output_op, multiplier_op, shift_op, indices):
    assert not isinstance(input_op, IndexOperandTemplate)
    assert not isinstance(output_op, IndexOperandTemplate)
    cdlt.compute("MUL", [input_op[indices], multiplier_op], [input_op[indices]], target="SIMD")
    cdlt.compute("RSHIFT", [input_op[indices], shift_op], [input_op[indices]], target="SIMD")
    cdlt.compute("32FXP_8FXP", [input_op[indices]], [output_op[indices]], target="SIMD")


def add_quantization(cdlt, operand):
    pass