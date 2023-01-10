## This file contains hardware parameters and the input and output data objects for the results

import logging
import math
import numpy as np

class HardwareObject(object):
    # This object class stores all the hardware parameters
    def __init__(self, Hardware_config, Mode):
        # this object takes the Hardware parameters as input
        #print(Hardware_config)
        
        ################################# Systolic array parameters
        self.SysArray_row = Hardware_config['ARRAY_N']
        self.SysArray_col = Hardware_config['ARRAY_M']

        #total size in kB
        self.Size_WBUF = Hardware_config['WBUF_CAPACITY_BITS']/(8 * 1024)     
        self.Size_IBUF = Hardware_config['IBUF_CAPACITY_BITS']/(8 * 1024)  
        self.Size_OBUF = 2 * Hardware_config['OBUF_CAPACITY_BITS']/(8 * 1024)  # the hardware config file provides param for OBUF1, hence multiply by 2
        self.Size_BBUF = Hardware_config['BBUF_CAPACITY_BITS']/(8 * 1024)

        #Hardware bitwidth. this part is used to set SIMD ofmap bitwidth depending on whether the SIMD layer is before a conv/FC or not
        #when the SIMD layer is followed by a conv/fc layer the output of that SIMD layer is quantized to match the activation bitwidth of next Sys array layer
        self.SAbw_ifmap = Hardware_config['ACT_DATA_WIDTH']
        self.SAbw_filter = Hardware_config['WGT_DATA_WIDTH']
        self.SAbw_bias = Hardware_config['BIAS_WIDTH']
        self.SAbw_psum = Hardware_config['ACC_WIDTH']
        self.SAbw_ofmap = self.SAbw_psum
        #self.SIMD_bw = 32   # current SIMD is 32 bit

        # off-chip interface, Maximum WBUF bandwidth is limited by number of PE rows * bw_filter
        max_WBUF_BW = self.SysArray_row * self.SAbw_filter
        self.RBW_DRAM_to_WBUF = min(max_WBUF_BW, Hardware_config['PARAMBUF_AXI_DATA_WIDTH'])   # in bit/cycle, bias is also loaded through the same AXI interface
        self.RBW_DRAM_to_IBUF = Hardware_config['IBUF_AXI_DATA_WIDTH']
        self.RBW_DRAM_to_OBUF = Hardware_config['OBUF_AXI_DATA_WIDTH']
        self.WBW_OBUF_to_DRAM = Hardware_config['OBUF_AXI_DATA_WIDTH']
        # In the current implementation, all buffers of the Systolic array is double buffered
        self.Buffering_scheme_SA = "double"

        ################################### SIMD array parameters
        self.SIMD_Ins_size = 32  # size of SIMD instruction in bits
        self.SIMD_address = 16  # size of the addresses for the SIMD buffers, in bits

        #total size in kB
        self.Size_VMEM1 = Hardware_config['SIMD_VMEM_CAPACITY_BITS']/(8 * 1024)
        self.Size_VMEM2 = Hardware_config['SIMD_VMEM_CAPACITY_BITS']/(8 * 1024)
        self.Size_IMM = 0.125
        self.Size_InsMem = (math.pow(2, Hardware_config['SIMD_INST_MEM_ADDR_WIDTH'])) * self.SIMD_Ins_size / (8 * 1024)
        #print("Size of SIMD InsMem:", self.Size_InsMem)

        #VMEM is single buffer, IMM and InsMem are also single buffered, 
        self.Buffering_scheme_VMEM = "single"  

        #Using one bandwidth parameter for both VMEM1 & VMEM2, both VMEM uses the same AXI to perform data transaction with the DRAM
        self.RBw_DRAM_to_VMEM = Hardware_config['SIMD_AXI_DATA_WIDTH'] # in bit/cycle
        self.WBw_VMEM_to_DRAM = Hardware_config['SIMD_AXI_DATA_WIDTH'] 
        #self.RBw_DRAM_to_IMM = 64  # IMM does not read data from DRAM, data in IMM comes with the instruction
        #self.RBw_DRAM_to_InsMem = 64  # Instruction memory of SIMD does not read instructions from DRAM. It comes from the on-chip global buffer for instructions

        ################################## Latency parameters for some of the SIMD arithmetic operation which take more than one cycle
        #all other arithmatic operations except the ones below require a single cycle to complete
        #The number of cycles for these operations are obtained from DesignWare IPs for the target clock period (800 ps)
        if Mode == "inference":
            # 32 bit fixed point implementation for inference
            self.div_cycles = 9   
            self.exp_cycles = 3
            self.inv_sqrt_cycles = 13
        else:
            # 32 bit floating point implementation for training
            self.div_cycles = 5   
            self.exp_cycles = 3
            self.inv_sqrt_cycles = 11

        self.cond_move_cycles = 1   # Nos of cycles for the conditional move operation


# All access results are in bits
class SAResult_Inflayer(object):
    # This object stores the simulation outputs for the layers which are executed on the systolic array
    # The result is for each layer
    def __init__(self):
        # initializing all the result of a layer with zeros
        self.SRAM_access = {}
        self.SRAM_access['filter'] = 0
        self.SRAM_access['ifmap'] = 0
        self.SRAM_access['psum'] = 0
        self.SRAM_access['bias'] = 0

        self.DRAM_access = {}
        self.DRAM_access['filter'] = 0
        self.DRAM_access['ifmap'] = 0
        self.DRAM_access['ofmap'] = 0
        self.DRAM_access['psum'] = 0
        self.DRAM_access['bias'] = 0

        self.cycles = {}
        self.cycles['compute'] = 0
        self.cycles['DRAM_Stall'] = 0
        self.cycles['total'] = 0

        self.pipe_register_access = 0

        self.arithmetic = {}
        self.arithmetic['mac'] = 0

        

# All access results are in bits
class SIMDResult_Inflayer(object):
    # This object stores the simulation outputs for the layers which are executed on the SIMD array
    # The result is for each layer
    def __init__(self):
        # initializing all the result of a layer with zeros

        # for systolic array, there is dedicated buffer for each data type. Hence for SA using buffer-wise or data type wise parameterization is same
        # For SIMD array, used buffer wise parameterization instead of data-wise since there is not dedicated buffer for each data type
        # The parameterization may grow as new layers are implemented depending on the need
        self.SRAM_access = {}
        self.SRAM_access['VMEM'] = 0
        self.SRAM_access['IMM'] = 0
        self.SRAM_access['OBUF'] = 0
        self.SRAM_access['IBUF'] = 0
        self.SRAM_access['InsMem'] = 0

        self.DRAM_access = {}
        self.DRAM_access['filter'] = 0
        self.DRAM_access['ifmap'] = 0
        self.DRAM_access['ofmap'] = 0
        self.DRAM_access['intermediate'] = 0      # there might be intermediate data that goes to DRAM for SIMD layers (i.e., softmax)

        self.cycles = {}
        self.cycles['compute'] = 0
        self.cycles['DRAM_stall'] = 0
        self.cycles['total'] = 0

        self.pipe_reg_access = 0
        self.indextbl_access = 0
        self.addrgen_add = 0 #add operation for address generation. bit width of address may be different from operands (i.e. 16 vs 32), hence keeping seperate param

        ## parameters for different arithmatic operations (will grow)
        self.arithmetic = {}
        self.arithmetic['max'] = 0
        self.arithmetic['add'] = 0
        self.arithmetic['sub'] = 0
        self.arithmetic['mul'] = 0
        self.arithmetic['div'] = 0
        self.arithmetic['exp'] = 0 
        self.arithmetic['inv_sqrt'] = 0  
        self.arithmetic['op_ScmnBN'] = 0   #this is for the SIMD common function & Batch Norm functions during training; the output is given directly as op count (may modify later)
        self.arithmetic['CondMove'] = 0

        





















            




        