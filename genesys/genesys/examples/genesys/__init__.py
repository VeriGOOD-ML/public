from codelets import Datatype
from fxpmath import Fxp
OP_LOCATIONS = {"OBUF": 0, "IBUF": 1, "VMEM": 2, "IMM": 3, "EXTMEM": 4}
SIMD_OP_READ_NS = ["OBUF", "VMEM", "IMM"]
SIMD_OP_WRITE_NS = ["IBUF", "VMEM", "IMM"]
SIMD_NS = ["OBUF", "IBUF", "VMEM", "IMM"]

OP_DTYPES = [Datatype(type='FXP', bitwidth=8, fractional=4, exp=4),
             Datatype(type='FXP', bitwidth=16, fractional=8, exp=8),
             Datatype(type='FXP', bitwidth=32, fractional=16, exp=16),
             Datatype(type='FP', bitwidth=16), Datatype(type='FP', bitwidth=32), Datatype(type='FXP', bitwidth=4)]

DTYPE_MAP = {}
DTYPE_MAP['FXP32'] = Datatype(type='FXP', bitwidth=32)
DTYPE_MAP['FXP16'] = Datatype(type='FXP', bitwidth=16)
DTYPE_MAP['FXP8'] = Datatype(type='FXP', bitwidth=8)
DTYPE_MAP['FXP4'] = Datatype(type='FXP', bitwidth=4)
DTYPE_MAP['FP32'] = Datatype(type='FXP', bitwidth=32)
DTYPE_MAP['FP16'] = Datatype(type='FXP', bitwidth=16)

# GENESYS_DTYPES = {}
# GENESYS_DTYPES['SIMD'] = 'FXP32'
# GENESYS_DTYPES['SYSTOLIC_ARRAY'] = {}
# GENESYS_DTYPES['SYSTOLIC_ARRAY']['inp_weight'] = 'FXP8'
# GENESYS_DTYPES['SYSTOLIC_ARRAY']['bias_out'] = 'FXP32'

FXP_CONFIGS = {
    "FXP32": {"signed": True, "n_int": 15, "n_frac": 16, "overflow": "saturate", "n_word": 32},
    "FXP8": {"signed": True, "n_int": 3, "n_frac": 4, "overflow": "saturate", "n_word": 8},
    "FXP16": {"signed": True, "n_int": 7, "n_frac": 8, "overflow": "saturate", "n_word": 16},
    "FXP4": {"signed": True, "n_int": 1, "n_frac": 2, "overflow": "saturate", "n_word": 4},
}
QUANT_SCALE = 0.6
SIGN_SHIFT = 30
QUANT_SCALE = Fxp(QUANT_SCALE, **FXP_CONFIGS['FXP32']).val.item()
SIGN_SHIFT = Fxp(SIGN_SHIFT, **FXP_CONFIGS['FXP32']).val.item()

BIT = 1
BYTE = 8

# Start of cfg options moved to cfg file
# GENESYS_CFG = {}
# GENESYS_CFG['DATA_WIDTH'] = DTYPE_MAP[GENESYS_DTYPES['SYSTOLIC_ARRAY']['inp_weight']].bits()
# GENESYS_CFG['WGT_WIDTH'] = DTYPE_MAP[GENESYS_DTYPES['SYSTOLIC_ARRAY']['inp_weight']].bits()
# GENESYS_CFG['BIAS_WIDTH'] = DTYPE_MAP[GENESYS_DTYPES['SYSTOLIC_ARRAY']['bias_out']].bits()
# GENESYS_CFG['ACC_WIDTH'] = DTYPE_MAP[GENESYS_DTYPES['SYSTOLIC_ARRAY']['bias_out']].bits()
# GENESYS_CFG['INSTR_WIDTH'] = 32
#
#
# PAPER_CFG1 = False
# NON_ASIC_32CFG = False
# TINY_CFG = False
# SMALL_CFG = False
# MED_CFG = False
# ASIC_CONFIG = False
# PAPER_CFG2 = False
# CUSTOM_CFG = 32
# DEMO_CFG = False
#
# ## Quantization
# USE_QUANTIZATION = False
#
#
# SW_PIPELINE_TEST = False
# ADDR_GEN_TEST = False
#
#
# SYS_TILE_CONSTR = False
# FUSION_CONSTRAINTS = False
# ALL_QUANT_OFF = True
#
# if SW_PIPELINE_TEST:
#     USE_QUANTIZATION = False
#     ALL_QUANT_OFF = True
#     FUSION_CONSTRAINTS = False
#
# if ALL_QUANT_OFF:
#     USE_QUANTIZATION = False
#


### End quantization
## SHIYU CONFIG
# if isinstance(CUSTOM_CFG, int):
#     bw_factor = 1
#     mem_factor = 1
#     cfg_size = CUSTOM_CFG
#     multiplier = 128 // cfg_size
#
#     GENESYS_CFG['ARRAY_N'] = cfg_size
#     GENESYS_CFG['ARRAY_M'] = cfg_size
#     GENESYS_CFG['PARAM_BUF_CHANNEL_BW'] = 512 // bw_factor // BIT
#     GENESYS_CFG['IBUF_CHANNEL_BW'] = 512 // bw_factor // BIT
#     GENESYS_CFG['OBUF_CHANNEL_BW'] = 512 // bw_factor // BIT
#     GENESYS_CFG['INSTR_CHANNEL_BW'] = 512 // bw_factor // BIT
#     GENESYS_CFG['SIMD_CHANNEL_BW'] = 512 // bw_factor // BIT
#     GENESYS_CFG['IBUF_DEPTH'] = 1024*multiplier
#     GENESYS_CFG['WBUF_DEPTH'] = 64*multiplier*multiplier
#     GENESYS_CFG['OBUF_DEPTH'] = 256*multiplier
#     GENESYS_CFG['BBUF_DEPTH'] = 128*multiplier
#
# elif ASIC_CONFIG:
#     GENESYS_CFG['ARRAY_N'] = 32
#     GENESYS_CFG['ARRAY_M'] = 32
#     GENESYS_CFG['PARAM_BUF_CHANNEL_BW'] = 256 // BIT
#     GENESYS_CFG['IBUF_CHANNEL_BW'] = 256 // BIT
#     GENESYS_CFG['OBUF_CHANNEL_BW'] = 256 // BIT
#     GENESYS_CFG['INSTR_CHANNEL_BW'] = 256 // BIT
#     GENESYS_CFG['SIMD_CHANNEL_BW'] = 256 // BIT
#     GENESYS_CFG['IBUF_DEPTH'] = int(1048576 / (GENESYS_CFG['ARRAY_M']*GENESYS_CFG['DATA_WIDTH']))
#     GENESYS_CFG['WBUF_DEPTH'] = int(2097152 / (GENESYS_CFG['ARRAY_M']*GENESYS_CFG['ARRAY_N']*GENESYS_CFG['ACC_WIDTH']))
#     GENESYS_CFG['OBUF_DEPTH'] = int(2097152*2 / (GENESYS_CFG['ARRAY_N']*GENESYS_CFG['ACC_WIDTH']))
#     GENESYS_CFG['BBUF_DEPTH'] = int(32768 / (GENESYS_CFG['ARRAY_M']*GENESYS_CFG['ACC_WIDTH']))
# elif SMALL_CFG:
#     factor = 8
#     GENESYS_CFG['ARRAY_N'] = 8
#     GENESYS_CFG['ARRAY_M'] = 8
#     GENESYS_CFG['PARAM_BUF_CHANNEL_BW'] = 512 // BIT
#     GENESYS_CFG['IBUF_CHANNEL_BW'] = 512 // BIT
#     GENESYS_CFG['OBUF_CHANNEL_BW'] = 512 // BIT
#     GENESYS_CFG['INSTR_CHANNEL_BW'] = 512 // BIT
#     GENESYS_CFG['SIMD_CHANNEL_BW'] = 512 // BIT
#     GENESYS_CFG['IBUF_DEPTH'] = 2048*factor
#     GENESYS_CFG['WBUF_DEPTH'] = 4096*factor*factor
#     GENESYS_CFG['OBUF_DEPTH'] = 2048*factor
#     GENESYS_CFG['BBUF_DEPTH'] = 1024*factor
# elif TINY_CFG:
#     factor = 8
#     GENESYS_CFG['ARRAY_N'] = 4
#     GENESYS_CFG['ARRAY_M'] = 4
#     GENESYS_CFG['PARAM_BUF_CHANNEL_BW'] = 512 // BIT
#     GENESYS_CFG['IBUF_CHANNEL_BW'] = 512 // BIT
#     GENESYS_CFG['OBUF_CHANNEL_BW'] = 512 // BIT
#     GENESYS_CFG['INSTR_CHANNEL_BW'] = 512 // BIT
#     GENESYS_CFG['SIMD_CHANNEL_BW'] = 512 // BIT
#     GENESYS_CFG['IBUF_DEPTH'] = 2048 * factor
#     GENESYS_CFG['WBUF_DEPTH'] = 4096 * factor * factor
#     GENESYS_CFG['OBUF_DEPTH'] = 2048 * factor
#     GENESYS_CFG['BBUF_DEPTH'] = 1024 * factor
# elif MED_CFG:
#     factor = 8
#     GENESYS_CFG['ARRAY_N'] = 16
#     GENESYS_CFG['ARRAY_M'] = 16
#     GENESYS_CFG['PARAM_BUF_CHANNEL_BW'] = 512 // BIT
#     GENESYS_CFG['IBUF_CHANNEL_BW'] = 512 // BIT
#     GENESYS_CFG['OBUF_CHANNEL_BW'] = 512 // BIT
#     GENESYS_CFG['INSTR_CHANNEL_BW'] = 512 // BIT
#     GENESYS_CFG['SIMD_CHANNEL_BW'] = 512 // BIT
#     GENESYS_CFG['IBUF_DEPTH'] = 2048*factor
#     GENESYS_CFG['WBUF_DEPTH'] = 4096*factor*factor
#     GENESYS_CFG['OBUF_DEPTH'] = 2048*factor
#     GENESYS_CFG['BBUF_DEPTH'] = 1024*factor
# elif NON_ASIC_32CFG:
#     factor = 1
#     GENESYS_CFG['ARRAY_N'] = 32
#     GENESYS_CFG['ARRAY_M'] = 32
#     GENESYS_CFG['PARAM_BUF_CHANNEL_BW'] = 512 // BIT
#     GENESYS_CFG['IBUF_CHANNEL_BW'] = 512 // BIT
#     GENESYS_CFG['OBUF_CHANNEL_BW'] = 512 // BIT
#     GENESYS_CFG['INSTR_CHANNEL_BW'] = 512 // BIT
#     GENESYS_CFG['SIMD_CHANNEL_BW'] = 512 // BIT
#     GENESYS_CFG['IBUF_DEPTH'] = 2048*factor
#     GENESYS_CFG['WBUF_DEPTH'] = 4096*factor*factor
#     GENESYS_CFG['OBUF_DEPTH'] = 2048*factor
#     GENESYS_CFG['BBUF_DEPTH'] = 1024*factor
# elif PAPER_CFG1:
#     GENESYS_CFG['ARRAY_N'] = 16
#     GENESYS_CFG['ARRAY_M'] = 16
#     GENESYS_CFG['PARAM_BUF_CHANNEL_BW'] = 512 // BIT
#     GENESYS_CFG['IBUF_CHANNEL_BW'] = 512 // BIT
#     GENESYS_CFG['OBUF_CHANNEL_BW'] = 512 // BIT
#     GENESYS_CFG['INSTR_CHANNEL_BW'] = 512 // BIT
#     GENESYS_CFG['SIMD_CHANNEL_BW'] = 512 // BIT
#     GENESYS_CFG['IBUF_DEPTH'] = 4096
#     GENESYS_CFG['WBUF_DEPTH'] = 1024
#     GENESYS_CFG['OBUF_DEPTH'] = 1024
#     GENESYS_CFG['BBUF_DEPTH'] = 128
# elif PAPER_CFG2:
#     bw_factor = 1
#     mem_factor = 1
#     cfg_size = 32
#
#     GENESYS_CFG['ARRAY_N'] = cfg_size
#     GENESYS_CFG['ARRAY_M'] = cfg_size
#     GENESYS_CFG['PARAM_BUF_CHANNEL_BW'] = 512
#     GENESYS_CFG['IBUF_CHANNEL_BW'] = 512
#     GENESYS_CFG['OBUF_CHANNEL_BW'] = 512
#     GENESYS_CFG['INSTR_CHANNEL_BW'] = 512
#     GENESYS_CFG['SIMD_CHANNEL_BW'] = 512
#
#     GENESYS_CFG['IBUF_DEPTH'] = 2048
#     GENESYS_CFG['WBUF_DEPTH'] = 2048
#     GENESYS_CFG['OBUF_DEPTH'] = 2048
#     GENESYS_CFG['BBUF_DEPTH'] = 1024
# elif DEMO_CFG:
#
#     cfg_size = 32
#     ALL_QUANT_OFF = True
#     GENESYS_CFG['ARRAY_N'] = cfg_size
#     GENESYS_CFG['ARRAY_M'] = cfg_size
#     GENESYS_CFG['PARAM_BUF_CHANNEL_BW'] = 512
#     GENESYS_CFG['IBUF_CHANNEL_BW'] = 512
#     GENESYS_CFG['OBUF_CHANNEL_BW'] = 512
#     GENESYS_CFG['INSTR_CHANNEL_BW'] = 512
#     GENESYS_CFG['SIMD_CHANNEL_BW'] = 512
#
#     GENESYS_CFG['IBUF_DEPTH'] = 2048
#     GENESYS_CFG['WBUF_DEPTH'] = 2048
#     GENESYS_CFG['OBUF_DEPTH'] = 2048
#     GENESYS_CFG['BBUF_DEPTH'] = 1024
# else:
#     ## DEFAULT CONFIG
#     GENESYS_CFG['ARRAY_N'] = 64
#     GENESYS_CFG['ARRAY_M'] = 64
#     GENESYS_CFG['PARAM_BUF_CHANNEL_BW'] = 512 // BIT
#     GENESYS_CFG['IBUF_CHANNEL_BW'] = 512 // BIT
#     GENESYS_CFG['OBUF_CHANNEL_BW'] = 512 // BIT
#     GENESYS_CFG['INSTR_CHANNEL_BW'] = 512 // BIT
#     GENESYS_CFG['SIMD_CHANNEL_BW'] = 512 // BIT
#     GENESYS_CFG['IBUF_DEPTH'] = 2048
#     GENESYS_CFG['WBUF_DEPTH'] = 4096
#     GENESYS_CFG['OBUF_DEPTH'] = 2048
#     GENESYS_CFG['BBUF_DEPTH'] = 1024

    ## END DEFAULT CONFIG


    ### Smaller test config 2

### End smaller test config 2

# GENESYS_CFG['SIMD_WIDTH'] = GENESYS_CFG['ARRAY_M']
# GENESYS_CFG['INSTR_DEPTH'] = 1024
# GENESYS_CFG['IMM_DEPTH'] = 64
# GENESYS_CFG['DRAM_DEPTH'] = 10000000
# GENESYS_CFG['VMEM_DEPTH'] = GENESYS_CFG['OBUF_DEPTH'] // 2 - 1
#
# GENESYS_CFG['VMEM_BANKS'] = GENESYS_CFG['SIMD_WIDTH']
# GENESYS_CFG['INSTR_BANKS'] = 1
#
# GENESYS_CFG['DRAM_WIDTH'] = 8
# GENESYS_CFG['DRAM_BANKS'] = GENESYS_CFG['SIMD_CHANNEL_BW'] // GENESYS_CFG['DRAM_WIDTH']
# end of cfg options moved to cfg file

SIMD_OPCODE_BITWIDTH = 4
SIMD_FNCODE_BITWIDTH = 4
NS_BITWIDTH = 5
NS_IDX_BITWIDTH = 3



from .genesys import define_genesys, compile_genesys, compile_genesys_layer, get_transformed_srdfg, \
    compile_extracted_genesys_layer, get_arch
