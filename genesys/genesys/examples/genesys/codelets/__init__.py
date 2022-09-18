
from .util import range_from_cfg, create_immediate_with_operand, add_quantization,\
    add_scale_op, add_sys_array_cast, add_scale_and_cast_op
from .arch_constraints import add_simd_constraint, add_conv_constraints,\
    add_gemm_constraints, add_simd_tile_constraint, add_flex_simd_constraints, add_multi_simd_constraint

# SW Impl
from .reference_impls.fusion_layers import load_fusion_impl, load_unquant_fusion_impl
from .reference_impls.gradients import load_gradient_impls
from .reference_impls.binary import load_binary_impls
from .reference_impls.unary import load_unary_impls
from .reference_impls.dnn import load_dnn_impls
from .reference_impls.transform import load_transform_impls
from .reference_impls.systolic_array import load_sa_impls
from .reference_impls.reduction import load_reduce_impls

# Codelets
from .fusion_layers import load_fusion_cdlts, load_fusion_op_info
from .unquantized_fusion_layers import load_unquant_fusion_op_info, load_unquant_fusion_cdlts
from .gradients import load_gradient_cdlts
from .binary import load_binary_cdlts
from .unary import load_unary_cdlts
from .dnn import load_dnn_cdlts
from .transform import load_transform_cdlts
from .systolic_array import load_sa_cdlts
from .reduction import load_reduce_cdlts

def load_impls_cdlts(cfg):


    if cfg['ALL_QUANT_OFF']:
        unquant_num = len(load_unquant_fusion_impl(cfg))
        print(f"Number fusion layers: {unquant_num}")
        GENESYS_IMPLS = {
            **load_unquant_fusion_impl(cfg),
            **load_gradient_impls(cfg),
            **load_binary_impls(cfg),
            **load_unary_impls(cfg),
            **load_dnn_impls(cfg),
            **load_transform_impls(cfg),
            **load_sa_impls(cfg),
            **load_reduce_impls(cfg)
        }
        GENESYS_CODELETS = {
            **load_unquant_fusion_cdlts(cfg),
            **load_gradient_cdlts(cfg),
            **load_binary_cdlts(cfg),
            **load_unary_cdlts(cfg),
            **load_dnn_cdlts(cfg),
            **load_transform_cdlts(cfg),
            **load_sa_cdlts(cfg),
            **load_reduce_cdlts(cfg)
        }

    else:
        GENESYS_IMPLS = {
            **load_fusion_impl(cfg),
            **load_gradient_impls(cfg),
            **load_binary_impls(cfg),
            **load_unary_impls(cfg),
            **load_dnn_impls(cfg),
            **load_transform_impls(cfg),
            **load_sa_impls(cfg),
            **load_reduce_impls(cfg)
        }
        GENESYS_CODELETS = {
            **load_fusion_cdlts(cfg),
            **load_gradient_cdlts(cfg),
            **load_binary_cdlts(cfg),
            **load_unary_cdlts(cfg),
            **load_dnn_cdlts(cfg),
            **load_transform_cdlts(cfg),
            **load_sa_cdlts(cfg),
            **load_reduce_cdlts(cfg)
        }
    for k in GENESYS_CODELETS.keys():
        if k not in GENESYS_IMPLS:
            raise RuntimeError(f"Not all codelets have a software implementation: {k}")
    return GENESYS_CODELETS, GENESYS_IMPLS

SPLIT_INFO = {}
SPLIT_INFO['depthwise_conv_bias'] = ('bias_add', 3, [('depthwise_conv', 3, ([0, 1],
                                                                        {'stride': 'stride', 'pad': 'pad',
                                                                         'groups': 'groups', 'dilation': 'dilation'})),2],)
