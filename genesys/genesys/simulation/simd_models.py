# This file contains the functions for the data access and cycle count models for the Layers executed on the SIMD Array

import logging
import math
import numpy as np
from data_objects import HardwareObject, SAResult_Inflayer, SIMDResult_Inflayer
from layer_object import LayerObject

#Current Quantization Assumption: for now assuming quantization is free and happen inside the ALU before writing ofmap back to VMEM

def other_access_SIMDhigh(Hardware_param, LayerObj, SIMDResult_inflayer, comp_cycles_ideal):
    #This function computes the number of accesses from the pipeline registers, index tables, and instruction memory
    #for all class-high SIMD layers (ReLU, Pool, Element-wise add of two operands)

    #unpacking the parameters
    Layer_name = LayerObj.Layer_name
    SIMD_Ins_size = Hardware_param.SIMD_Ins_size
    SIMD_address = Hardware_param.SIMD_address
    bw_ifmap = LayerObj.bw_ifmap # using 32 bit input
    SysArray_col = Hardware_param.SysArray_col  #of SIMD lane

    #Access for IF stage
    Nos_of_IF = comp_cycles_ideal  #number of instruction fetched
    InsMem_access = Nos_of_IF * SIMD_Ins_size    # in bit

    #Access for IF/ID pipeline register
    IFID_reg_access = SIMD_Ins_size * Nos_of_IF * 2  # 2 for read and write

    #NOTE: FOR SOFTMAX STEP1 IS ONE OPERAND WHILE STEP2,3 IS TWO OPERAND, STILL KEEPING THIS IN TWO OP CATEGORY FOR SIMPLICITY
    # determining layer category
    two_op_layer = ["ReLU", "ElemAdd", "MaxPool", "AvgPool", "Softmax", "ROIAlignPool"]  # layers for which there are two operands in the instruction
    one_op_layer = ["Sigmoid"]                                # layers for which there are one operand in the instruction

    #for RoIAlign, although there will be some IMM access, modeled all access count as VMEM for simplicity
    imm_read_layer = ["ReLU"]                                 # layers for which one operand is read from IMM, for ReLU zero operand is read from IMM

    # determining whetehr the layer has one operand vs. two operand instructions
    for layername in two_op_layer:
        if Layer_name == layername:
            operand_no = 2    #of operand
            break

    for layername in one_op_layer:
        if Layer_name == layername:
            operand_no = 1
            break

    #print("operand_no:", operand_no)

    # determining whether tha layer reads one operand from IMM or not,  
    immop_no = 0   #of operand read from imm, by default set to zero
    for layername in imm_read_layer:
        if Layer_name == layername:
            immop_no = 1   #of operand read from imm
            break
    #print("immop_no:", immop_no)

    #Access for the Index tables (index tables are implemented as registers in GeneSys)
    offset_size = SIMD_address
    stride_size = SIMD_address
    
    IT_read_access = (operand_no + 1) * (offset_size + stride_size) * Nos_of_IF
    IT_write_access = (operand_no + 1) * offset_size * Nos_of_IF
    indextable_access = IT_read_access + IT_write_access

    #Access for ID/AG pipeline register
    IDAG_reg_access = (SIMD_Ins_size + (operand_no + 1) * (offset_size + stride_size)) * Nos_of_IF * 2   # 2 for read and write

    # Addition operation for address generation
    addrgen_add = (operand_no + 1) * Nos_of_IF   #of addition operation to generate the address

    #Access for AG/RD pipeline register 
    AGRD_reg_access = (SIMD_Ins_size + (operand_no + 1) * SIMD_address) * Nos_of_IF * 2  # 2 for read and write

    #Access for RD/Ex pipeline register, for the first ALU
    RDExR1_reg_access = ((operand_no + 1 - immop_no) * SIMD_address + (3 + 3 + 3 + 4 + 4)) * Nos_of_IF * 2

    #Access for the R2 to Rn pipeline registers, for ALU2 to ALU last-1 
    bw_ALUout = bw_ifmap  #using bw_ifmap for the bitwidth if ALU output since quantization has not happened yet
    # in the following equation, using SysArray_col and that is fine since SRAM tiles will be padded for the corner cases
    R2Rn_reg_access = ((operand_no + 1 - immop_no) * SIMD_address + (3 + 3 + 3 + 4 + 4) + bw_ALUout + (bw_ALUout * immop_no)) * (SysArray_col - 1) * Nos_of_IF * 2

    #Access for the Rend (i.e., last) pipeline register
    Rend_reg_access = (SIMD_address + 3 + bw_ALUout) * Nos_of_IF * 2

    '''
    print("InsMem_access:", InsMem_access)
    print("IFID_reg_access:", IFID_reg_access)
    print("IDAG_reg_access:", IDAG_reg_access)
    print("AGRD_reg_access:", AGRD_reg_access)
    print("RDExR1_reg_access:", RDExR1_reg_access)
    print("R2Rn_reg_access:", R2Rn_reg_access)
    print("Rend_reg_access:", Rend_reg_access)
    print("indextable_access:", indextable_access)
    print("addrgen_add:", addrgen_add)
    '''

    SIMDResult_inflayer.SRAM_access['InsMem'] = InsMem_access
    SIMDResult_inflayer.pipe_reg_access = IFID_reg_access + IDAG_reg_access + AGRD_reg_access + RDExR1_reg_access + R2Rn_reg_access + Rend_reg_access
    SIMDResult_inflayer.indextbl_access = indextable_access
    SIMDResult_inflayer.addrgen_add = addrgen_add


def relu_access_model(Hardware_param, LayerObj, SIMDResult_inflayer):
    # data access model for ReLU layer
    ####### For layers where IC = OC in the function code, used OC parameters and IC paramters are ignored
    ####### For example, for ReLU, IC, DTile_ic, STile_ic etc is not used in their model's code

    bw_ifmap = LayerObj.bw_ifmap 
    bw_ofmap = LayerObj.bw_ofmap  

    OW = LayerObj.OW
    OH = LayerObj.OH
    OC = LayerObj.OC
    Batch = LayerObj.Batch

    DTile_ow = LayerObj.DTile_ow
    DTile_oh = LayerObj.DTile_oh
    DTile_oc = LayerObj.DTile_oc
    DTile_batch = LayerObj.DTile_batch

    Stile_ow = LayerObj.Stile_ow
    Stile_oh = LayerObj.Stile_oh
    Stile_oc = LayerObj.Stile_oc
    Stile_batch = LayerObj.Stile_batch

    fusion_status = LayerObj.fusion_status

    #Loop order does not matter for ReLU layer
    ############ DRAM Access
    if (fusion_status == "NoFusion"):
        ifmap_access_DRAM = (DTile_oh * DTile_ow * DTile_oc * DTile_batch) * (OH/DTile_oh) * (OW/DTile_ow) * (OC/DTile_oc) * (Batch/DTile_batch) * bw_ifmap
        ofmap_access_DRAM = (DTile_oh * DTile_ow * DTile_oc * DTile_batch) * (OH/DTile_oh) * (OW/DTile_ow) * (OC/DTile_oc) * (Batch/DTile_batch) * bw_ofmap

        #print("ifmap_access_DRAM:", ifmap_access_DRAM)
        #print("ofmap_access_DRAM:", ofmap_access_DRAM)
        SIMDResult_inflayer.DRAM_access['ifmap'] = ifmap_access_DRAM
        SIMDResult_inflayer.DRAM_access['ofmap'] = ofmap_access_DRAM

    else:
        print("model for fusion do not exist yet")


    ########## On-chip SRAM Access
    ## Access for VMEM
    if (fusion_status == "NoFusion"):
        ifmap_access_VMEM = (Stile_oh * Stile_ow * Stile_oc * Stile_batch) * (DTile_oh/Stile_oh) * (DTile_ow/Stile_ow) * (DTile_oc/Stile_oc) * (DTile_batch/Stile_batch) \
                                        * (OH/DTile_oh) * (OW/DTile_ow) * (OC/DTile_oc) * (Batch/DTile_batch) * bw_ifmap

        ifmap_access_OBUF = 0  # for fusion-1, the ifmap will be accessed from OBUF instead of VMEM, hence keepig this parameter

        # for now assuming quantization is free and happen inside the ALU before writing ofmap back to VMEM
        ofmap_access_VMEM = (Stile_oh * Stile_ow * Stile_oc * Stile_batch) * (DTile_oh/Stile_oh) * (DTile_ow/Stile_ow) * (DTile_oc/Stile_oc) * (DTile_batch/Stile_batch) \
                                        * (OH/DTile_oh) * (OW/DTile_ow) * (OC/DTile_oc) * (Batch/DTile_batch) * bw_ofmap

        #print("ifmap_access_VMEM:", ifmap_access_VMEM)
        #print("ofmap_access_VMEM:", ofmap_access_VMEM)
        SIMDResult_inflayer.SRAM_access['VMEM'] = ifmap_access_VMEM + ofmap_access_VMEM
    
    else:       
        print("model for fusion do not exist yet")

    ## Access for IMM (for ReLU, the zero operand is read from IMM)
    ifmap_access_IMM = (Stile_oh * Stile_ow * Stile_batch) * (DTile_oh/Stile_oh) * (DTile_ow/Stile_ow) * math.ceil((DTile_oc/Stile_oc)) * (DTile_batch/Stile_batch) \
                                        * (OH/DTile_oh) * (OW/DTile_ow) * (OC/DTile_oc) * (Batch/DTile_batch) * bw_ifmap
    #print("ifmap_access_IMM:", ifmap_access_IMM)
    SIMDResult_inflayer.SRAM_access['IMM'] = ifmap_access_IMM

    ## Number of max operations, 
    Nos_of_max = (Stile_oh * Stile_ow * Stile_oc * Stile_batch) * (DTile_oh/Stile_oh) * (DTile_ow/Stile_ow) * (DTile_oc/Stile_oc) * (DTile_batch/Stile_batch) \
                                        * (OH/DTile_oh) * (OW/DTile_ow) * (OC/DTile_oc) * (Batch/DTile_batch)
    SIMDResult_inflayer.arithmetic["max"] = Nos_of_max


def relu_cycle_model(Hardware_param, LayerObj, SIMDResult_inflayer):
    #compute cycle and DRAM stall cycle count model for the ReLU layer

    bw_ifmap = LayerObj.bw_ifmap 
    bw_ofmap = LayerObj.bw_ofmap 

    SysArray_col = Hardware_param.SysArray_col
    RBw_DRAM_to_VMEM = Hardware_param.RBw_DRAM_to_VMEM
    WBw_VMEM_to_DRAM = Hardware_param.WBw_VMEM_to_DRAM

    OW = LayerObj.OW
    OH = LayerObj.OH
    OC = LayerObj.OC
    Batch = LayerObj.Batch

    DTile_ow = LayerObj.DTile_ow
    DTile_oh = LayerObj.DTile_oh
    DTile_oc = LayerObj.DTile_oc
    DTile_batch = LayerObj.DTile_batch

    Stile_ow = LayerObj.Stile_ow
    Stile_oh = LayerObj.Stile_oh
    Stile_oc = LayerObj.Stile_oc
    Stile_batch = LayerObj.Stile_batch

    fusion_status = LayerObj.fusion_status

    # #of cycles to compute one tile, k operations per cycle, compute cycles do not depend on loop order or fusion
    cycle_oneTile = ((Stile_oh * Stile_ow * Stile_oc * Stile_batch) / SysArray_col)  * (DTile_oh/Stile_oh) * (DTile_ow/Stile_ow) * (DTile_oc/Stile_oc) * (DTile_batch/Stile_batch) 
    #print("cycle_oneTile:", cycle_oneTile)

    #of cycles to fill the pipeline for a tile, there are 6 pipeline stages
    pipe_overhead_tile = (6 - 1) + (SysArray_col - 1)  #using PE col, 

    #of cycles to compute one tile including the pipeline setup operhead, need this variable to compute DRAM stall cycles
    ComputeTile_cycles = cycle_oneTile + pipe_overhead_tile

    #for now omitting the use of any ceil since DRAM tile size will be integer multiple of loops, 
    Number_of_Tile = (OH/DTile_oh) * (OW/DTile_ow) * (OC/DTile_oc) * (Batch/DTile_batch)
    compute_cycles = math.ceil((cycle_oneTile + pipe_overhead_tile) * Number_of_Tile)   # giving the outer ceil to avoid fraction cycle numbers

    ######## model for the DRAM stall cycles, depends on fusion etc
    if (fusion_status == "NoFusion"): #Model for the version where there is no fusion
        if (Hardware_param.Buffering_scheme_VMEM == "single"):
            #of cycles required to load/store each tile of each kind of data, both VMEM shares the same AXI 
            ifmapTile_load_cycles = math.ceil((DTile_ow * DTile_oh * DTile_oc * DTile_batch * bw_ifmap) / RBw_DRAM_to_VMEM)
            ofmapTile_store_cycles = math.ceil((DTile_ow * DTile_oh * DTile_oc * DTile_batch * bw_ofmap) / WBw_VMEM_to_DRAM)

            # VMEM is single buffered, so there is no overlap with computation.
            ifmap_stall_cycles = ifmapTile_load_cycles * Number_of_Tile
            ofmap_stall_cycles = ofmapTile_store_cycles * Number_of_Tile
            DRAM_stall_cycles = ifmap_stall_cycles + ofmap_stall_cycles
        else:
            print("Model do not exist for VMEM non single buffer scheme now")
    else:
        print("model for fusion do not exist yet")

    #print("compute_cycles:", compute_cycles)
    #print("DRAM_stall_cycles:", DRAM_stall_cycles)
    SIMDResult_inflayer.cycles['compute'] = compute_cycles
    SIMDResult_inflayer.cycles['DRAM_stall'] = DRAM_stall_cycles
    SIMDResult_inflayer.cycles['total'] = compute_cycles + DRAM_stall_cycles
    
    # other accesses which happens at every compute cycle
    # using a common function for all SIMD class high layers to calculate the access from pipeline registers, instruction memory, and index tables
    comp_cycles_ideal = math.ceil(cycle_oneTile * Number_of_Tile)   #of total compute cycles without pipeline overhead
    #print("comp_cycles_ideal:", comp_cycles_ideal)
    other_access_SIMDhigh(Hardware_param, LayerObj, SIMDResult_inflayer, comp_cycles_ideal)

# ElemAdd and ReLU is very similar. Despite keeping them in seperate functions since fusion will affect them differently
def elemadd_access_model(Hardware_param, LayerObj, SIMDResult_inflayer):
    # data access model for element wise addition layer

    bw_ifmap = LayerObj.bw_ifmap 
    bw_ofmap = LayerObj.bw_ofmap 

    OW = LayerObj.OW
    OH = LayerObj.OH
    OC = LayerObj.OC
    Batch = LayerObj.Batch

    DTile_ow = LayerObj.DTile_ow
    DTile_oh = LayerObj.DTile_oh
    DTile_oc = LayerObj.DTile_oc
    DTile_batch = LayerObj.DTile_batch

    Stile_ow = LayerObj.Stile_ow
    Stile_oh = LayerObj.Stile_oh
    Stile_oc = LayerObj.Stile_oc
    Stile_batch = LayerObj.Stile_batch

    fusion_status = LayerObj.fusion_status

    #Loop order does not matter for ElemAdd layer
    ############ DRAM Access
    if (fusion_status == "NoFusion"):
        ifmap_access_DRAM_1 = (DTile_oh * DTile_ow * DTile_oc * DTile_batch) * (OH/DTile_oh) * (OW/DTile_ow) * (OC/DTile_oc) * (Batch/DTile_batch) * bw_ifmap
        ifmap_access_DRAM_2 = (DTile_oh * DTile_ow * DTile_oc * DTile_batch) * (OH/DTile_oh) * (OW/DTile_ow) * (OC/DTile_oc) * (Batch/DTile_batch) * bw_ifmap
        # readig both tensor from the DRAM
        ifmap_access_DRAM = ifmap_access_DRAM_1 + ifmap_access_DRAM_2

        ofmap_access_DRAM = (DTile_oh * DTile_ow * DTile_oc * DTile_batch) * (OH/DTile_oh) * (OW/DTile_ow) * (OC/DTile_oc) * (Batch/DTile_batch) * bw_ofmap

        #print("ifmap_access_DRAM:", ifmap_access_DRAM)
        #print("ofmap_access_DRAM:", ofmap_access_DRAM)
        SIMDResult_inflayer.DRAM_access['ifmap'] = ifmap_access_DRAM
        SIMDResult_inflayer.DRAM_access['ofmap'] = ofmap_access_DRAM
    else:
        print("model for fusion do not exist yet")


    ########## On-chip SRAM Access
    ## Access for VMEM
    if (fusion_status == "NoFusion"):
        # for two input tensors, hence multiplying with 2
        ifmap_access_VMEM = (Stile_oh * Stile_ow * Stile_oc * Stile_batch) * (DTile_oh/Stile_oh) * (DTile_ow/Stile_ow) * (DTile_oc/Stile_oc) * (DTile_batch/Stile_batch) \
                                        * (OH/DTile_oh) * (OW/DTile_ow) * (OC/DTile_oc) * (Batch/DTile_batch) * 2 * bw_ifmap

        ifmap_access_OBUF = 0  # for fusion-1, one input tensor will be accessed from OBUF instead of VMEM, hence keepig this parameter

        # for now assuming quantization is free and happen inside the ALU before writing ofmap back to VMEM
        ofmap_access_VMEM = (Stile_oh * Stile_ow * Stile_oc * Stile_batch) * (DTile_oh/Stile_oh) * (DTile_ow/Stile_ow) * (DTile_oc/Stile_oc) * (DTile_batch/Stile_batch) \
                                        * (OH/DTile_oh) * (OW/DTile_ow) * (OC/DTile_oc) * (Batch/DTile_batch) * bw_ofmap

        #print("ifmap_access_VMEM:", ifmap_access_VMEM)
        #print("ofmap_access_VMEM:", ofmap_access_VMEM)
        SIMDResult_inflayer.SRAM_access['VMEM'] = ifmap_access_VMEM + ofmap_access_VMEM
    else:
        print("model for fusion do not exist yet")


    #There is no IMM access for ElemAdd layer

    ## Number of add operations, 
    Nos_of_add = (Stile_oh * Stile_ow * Stile_oc * Stile_batch) * (DTile_oh/Stile_oh) * (DTile_ow/Stile_ow) * (DTile_oc/Stile_oc) * (DTile_batch/Stile_batch) \
                                        * (OH/DTile_oh) * (OW/DTile_ow) * (OC/DTile_oc) * (Batch/DTile_batch)
    #print("Nos_of_add:", Nos_of_add)
    SIMDResult_inflayer.arithmetic["add"] = Nos_of_add


def elemadd_cycle_model(Hardware_param, LayerObj, SIMDResult_inflayer):
    #compute cycle and DRAM stall cycle count model for the ElemAdd (element wise addition of two tensors) layer

    bw_ifmap = LayerObj.bw_ifmap 
    bw_ofmap = LayerObj.bw_ofmap 

    SysArray_col = Hardware_param.SysArray_col
    RBw_DRAM_to_VMEM = Hardware_param.RBw_DRAM_to_VMEM
    WBw_VMEM_to_DRAM = Hardware_param.WBw_VMEM_to_DRAM

    OW = LayerObj.OW
    OH = LayerObj.OH
    OC = LayerObj.OC
    Batch = LayerObj.Batch

    DTile_ow = LayerObj.DTile_ow
    DTile_oh = LayerObj.DTile_oh
    DTile_oc = LayerObj.DTile_oc
    DTile_batch = LayerObj.DTile_batch

    Stile_ow = LayerObj.Stile_ow
    Stile_oh = LayerObj.Stile_oh
    Stile_oc = LayerObj.Stile_oc
    Stile_batch = LayerObj.Stile_batch

    fusion_status = LayerObj.fusion_status

    # #of cycles to compute one tile, k operations per cycle, compute cycles do not depend on loop order or fusion
    cycle_oneTile = ((Stile_oh * Stile_ow * Stile_oc * Stile_batch) / SysArray_col) * (DTile_oh/Stile_oh) * (DTile_ow/Stile_ow) * (DTile_oc/Stile_oc) * (DTile_batch/Stile_batch)
    #print("cycle_oneTile:", cycle_oneTile)

    #of cycles to fill the pipeline for a tile, there are 6 pipeline stages
    pipe_overhead_tile = (6 - 1) + (SysArray_col - 1)  #using PE col, 
    #of cycles to compute one tile including the pipeline setup operhead, need this variable to compute DRAM stall cycles
    ComputeTile_cycles = cycle_oneTile + pipe_overhead_tile

    #for now omitting the use of any ceil since DRAM tile size will be integer multiple of loops, 
    Number_of_Tile = (OH/DTile_oh) * (OW/DTile_ow) * (OC/DTile_oc) * (Batch/DTile_batch)
    compute_cycles = math.ceil((cycle_oneTile + pipe_overhead_tile) * Number_of_Tile)   # giving the outer ceil to avoid fraction cycle numbers

    ######## model for the DRAM stall cycles, depends on fusion etc
    if (fusion_status == "NoFusion"): #Model for the version where there is no fusion
        if (Hardware_param.Buffering_scheme_VMEM == "single"):
            #of cycles required to load/store each tile of each kind of data, both VMEM share the same AXI
            ifmapTile_load_cycles_1 = math.ceil((DTile_ow * DTile_oh * DTile_oc * DTile_batch * bw_ifmap) / RBw_DRAM_to_VMEM)
            ifmapTile_load_cycles_2 = math.ceil((DTile_ow * DTile_oh * DTile_oc * DTile_batch * bw_ifmap) / RBw_DRAM_to_VMEM)
            ofmapTile_store_cycles = math.ceil((DTile_ow * DTile_oh * DTile_oc * DTile_batch * bw_ofmap) / WBw_VMEM_to_DRAM)

            # VMEM is single buffered, so there is no overlap with computation.
            ifmap_stall_cycles_1 = ifmapTile_load_cycles_1 * Number_of_Tile
            ifmap_stall_cycles_2 = ifmapTile_load_cycles_2 * Number_of_Tile
            ofmap_stall_cycles = ofmapTile_store_cycles * Number_of_Tile
            DRAM_stall_cycles = ifmap_stall_cycles_1 + ifmap_stall_cycles_2 + ofmap_stall_cycles
        else:
            print("Model do not exist for VMEM non single buffer scheme now")
    else:
        print("model for fusion do not exist yet")

    #print("compute_cycles:", compute_cycles)
    #print("DRAM_stall_cycles:", DRAM_stall_cycles)
    SIMDResult_inflayer.cycles['compute'] = compute_cycles
    SIMDResult_inflayer.cycles['DRAM_stall'] = DRAM_stall_cycles
    SIMDResult_inflayer.cycles['total'] = compute_cycles + DRAM_stall_cycles
    
    # other accesses which happens at every compute cycle
    # using a common function for all SIMD class high layers to calculate the access from pipeline registers, instruction memory, and index tables
    comp_cycles_ideal = math.ceil(cycle_oneTile * Number_of_Tile)   #of total compute cycles without pipeline overhead
    #print("comp_cycles_ideal:", comp_cycles_ideal)
    other_access_SIMDhigh(Hardware_param, LayerObj, SIMDResult_inflayer, comp_cycles_ideal)


def pool_access_model(Hardware_param, LayerObj, SIMDResult_inflayer):
    # this is the data access model for Average pool, Max pool, and global average pool layers

    bw_ifmap = LayerObj.bw_ifmap 
    bw_psum = LayerObj.bw_psum
    bw_ofmap = LayerObj.bw_ofmap  
    Layer_name = LayerObj.Layer_name

    OW = LayerObj.OW
    OH = LayerObj.OH
    OC = LayerObj.OC
    KW = LayerObj.KW
    KH = LayerObj.KH
    Batch = LayerObj.Batch

    DTile_ow = LayerObj.DTile_ow
    DTile_oh = LayerObj.DTile_oh
    DTile_oc = LayerObj.DTile_oc
    DTile_kw = LayerObj.DTile_kw
    DTile_kh = LayerObj.DTile_kh
    #DTile_ic = LayerObj.DTile_ic
    DTile_iw = LayerObj.DTile_iw
    DTile_ih = LayerObj.DTile_ih 
    DTile_batch = LayerObj.DTile_batch

    Stile_ow = LayerObj.Stile_ow
    Stile_oh = LayerObj.Stile_oh
    Stile_oc = LayerObj.Stile_oc
    Stile_kw = LayerObj.Stile_kw
    Stile_kh = LayerObj.Stile_kh
    #Stile_ic = LayerObj.Stile_ic 
    Stile_iw = LayerObj.Stile_iw 
    Stile_ih = LayerObj.Stile_ih 
    Stile_batch = LayerObj.Stile_batch

    #Loop_order = LayerObj.Loop_order
    fusion_status = LayerObj.fusion_status

    #Invalid tiling: in pooling KH and KW diemnsion should not be tiled for DRAM tiles. completely unnecessary and in-efficient 
    if (KW/DTile_kw) != 1 or (KH/DTile_kh) != 1:
        print("Invalid DRAM tiling for kernel height and width of pool layer")
        return

    #### DRAM access model: loop order does not matter since there is no data reuse
    if (fusion_status == "NoFusion"):
        ifmap_access_DRAM = (DTile_ih * DTile_iw * DTile_oc * DTile_batch) * (OH/DTile_oh) * (OW/DTile_ow) * (OC/DTile_oc) * (Batch/DTile_batch) * bw_ifmap
        ofmap_access_DRAM = (DTile_oh * DTile_ow * DTile_oc * DTile_batch) * (OH/DTile_oh) * (OW/DTile_ow) * (OC/DTile_oc) * (Batch/DTile_batch) * bw_ofmap

        #print("ifmap_access_DRAM:", ifmap_access_DRAM)
        #print("ofmap_access_DRAM:", ofmap_access_DRAM)
        SIMDResult_inflayer.DRAM_access['ifmap'] = ifmap_access_DRAM
        SIMDResult_inflayer.DRAM_access['ofmap'] = ofmap_access_DRAM

    else:
        print("model for fusion do not exist yet")
 
    ########## On-chip SRAM Access: loop order does not matter since there is no local register for the ALUs
    ## Access for VMEM: 
    DRAM_loop_multiplier = (OH/DTile_oh) * (OW/DTile_ow) * (OC/DTile_oc) * (Batch/DTile_batch)
    if (fusion_status == "NoFusion"):
        # read two input operands from the VMEM, hence multiplying with 2
        ifmap_access_VMEM = (Stile_ih * Stile_iw * Stile_oc * Stile_batch) * (DTile_oh/Stile_oh) * (DTile_ow/Stile_ow) * (DTile_oc/Stile_oc) * (DTile_batch/Stile_batch) \
                                                                           * (DTile_kh/Stile_kh) * (DTile_kw/Stile_kw) * DRAM_loop_multiplier * 2 * bw_ifmap
        ifmap_access_OBUF = 0  # for fusion-1, one input tensor will be accessed from OBUF instead of VMEM, hence keepig this parameter

        # intermediate data + ofmap access
        ofpsm_access_VMEM = (Stile_oh * Stile_ow * Stile_oc * Stile_batch) * (DTile_oh/Stile_oh) * (DTile_ow/Stile_ow) * (DTile_oc/Stile_oc) * (DTile_batch/Stile_batch) \
                                                                           * (DTile_kh/Stile_kh) * (DTile_kw/Stile_kw) * DRAM_loop_multiplier
        # intermediate data access
        psum_access_VMEM = (ofpsm_access_VMEM - (OH * OW * OC * Batch)) * bw_psum  # intermediate data is 32 bit (before quantization)
        # final ofmap access: for now assuming quantization is free and happen inside the ALU before writing ofmap back to VMEM 
        ofmap_access_VMEM = (OH * OW * OC * Batch) * bw_ofmap   
                                                                                  
        div_access_VMEM = 0  # additional VMEM access due to the division operation in Average Pool
        if Layer_name == "AvgPool":
            div_access_VMEM = (Stile_oh * Stile_ow * Stile_oc * Stile_batch) * (DTile_oh/Stile_oh) * (DTile_ow/Stile_ow) * (DTile_oc/Stile_oc) \
                                                   * (DTile_batch/Stile_batch) * DRAM_loop_multiplier * (2 + 1) * bw_psum
            #using bw_ifmap here casue used bw_ofmap in the above ofmap_access_VMEM equation which represents the final ofmap write to VMEM regardless of Avg or Max pool
            #while this div_access_VMEM mimics the intermediate data access

        #print("ifmap_access_VMEM:", ifmap_access_VMEM)
        #print("psum_access_VMEM:", psum_access_VMEM)
        #print("ofmap_access_VMEM:", ofmap_access_VMEM)
        #print("div_access_VMEM:", div_access_VMEM)
        SIMDResult_inflayer.SRAM_access['VMEM'] = ifmap_access_VMEM + psum_access_VMEM + ofmap_access_VMEM + div_access_VMEM

    else:
        print("model for fusion do not exist yet")

    # Nos of arithmetic operations:
    if Layer_name == "MaxPool":
        Nos_of_max = (Stile_oh * Stile_ow * Stile_oc * Stile_batch) * (DTile_oh/Stile_oh) * (DTile_ow/Stile_ow) * (DTile_oc/Stile_oc) * (DTile_batch/Stile_batch) \
                                                                           * (DTile_kh/Stile_kh) * (DTile_kw/Stile_kw) * DRAM_loop_multiplier
        #print("Nos_of_max:", Nos_of_max)
        SIMDResult_inflayer.arithmetic["max"] = Nos_of_max

    if Layer_name == "AvgPool":
        Nos_of_add = (Stile_oh * Stile_ow * Stile_oc * Stile_batch) * (DTile_oh/Stile_oh) * (DTile_ow/Stile_ow) * (DTile_oc/Stile_oc) * (DTile_batch/Stile_batch) \
                                                                           * (DTile_kh/Stile_kh) * (DTile_kw/Stile_kw) * DRAM_loop_multiplier
        Nos_of_div = (Stile_oh * Stile_ow * Stile_oc * Stile_batch) * (DTile_oh/Stile_oh) * (DTile_ow/Stile_ow) * (DTile_oc/Stile_oc) * (DTile_batch/Stile_batch) \
                                                                           * DRAM_loop_multiplier
        #print("Nos_of_add:", Nos_of_add)
        #print("Nos_of_div:", Nos_of_div)
        SIMDResult_inflayer.arithmetic["add"] = Nos_of_add
        SIMDResult_inflayer.arithmetic["div"] = Nos_of_div

def pool_cycle_model(Hardware_param, LayerObj, SIMDResult_inflayer):
    #compute cycle and DRAM stall cycle count model for Max Pool, Average Pool (also Global Average Pool) Layer

    bw_ifmap = LayerObj.bw_ifmap 
    bw_ofmap = LayerObj.bw_ofmap 
    Layer_name = LayerObj.Layer_name
    SysArray_col = Hardware_param.SysArray_col
    RBw_DRAM_to_VMEM = Hardware_param.RBw_DRAM_to_VMEM
    WBw_VMEM_to_DRAM = Hardware_param.WBw_VMEM_to_DRAM
    div_cycles = Hardware_param.div_cycles

    OW = LayerObj.OW
    OH = LayerObj.OH
    OC = LayerObj.OC
    KW = LayerObj.KW
    KH = LayerObj.KH
    Batch = LayerObj.Batch

    DTile_ow = LayerObj.DTile_ow
    DTile_oh = LayerObj.DTile_oh
    DTile_oc = LayerObj.DTile_oc
    DTile_kw = LayerObj.DTile_kw
    DTile_kh = LayerObj.DTile_kh
    #DTile_ic = LayerObj.DTile_ic
    DTile_iw = LayerObj.DTile_iw
    DTile_ih = LayerObj.DTile_ih 
    DTile_batch = LayerObj.DTile_batch

    Stile_ow = LayerObj.Stile_ow
    Stile_oh = LayerObj.Stile_oh
    Stile_oc = LayerObj.Stile_oc
    Stile_kw = LayerObj.Stile_kw
    Stile_kh = LayerObj.Stile_kh
    #Stile_ic = LayerObj.Stile_ic 
    #Stile_iw = LayerObj.Stile_iw 
    #Stile_ih = LayerObj.Stile_ih 
    Stile_batch = LayerObj.Stile_batch

    Loop_order = LayerObj.Loop_order
    fusion_status = LayerObj.fusion_status

    #Invalid tiling: in pooling KH and KW diemnsion should not be tiled for DRAM tiles. completely unnecessary and inefficient 
    if (KW/DTile_kw) != 1 or (KH/DTile_kh) != 1:
        print("Invalid DRAM tiling for kernel height and width of pool layer")
        return

    ### Placing conditions where stall will be trigered due to not having the data forwarding scheme in the SIMD pipeline of GeneSys. Not modeling this limitation, 
    # can be avoided such situations by scheduling/tiling, Hence placing warning
    if Loop_order[0] == 'kw' or Loop_order[0] == 'kh':
        print("WARNING: Model for additional SIMD stall cycles due to data dependency in GeneSys do not exist for the input loop order for the pool layer")

    # determining the number of operatios within a tile before which kh/kw loops come in a SIMD ALU
    Loop_dict = {'n': DTile_batch, 'oc': DTile_oc, 'ow': DTile_ow, 'oh': DTile_oh}
    Id = 0
    for key in Loop_order:
        if (key == 'kh' or key == 'kw'):
            break
        Id = Id + 1
    
    #print("Index of first kernel loop:", Id)
    #print(Loop_order[0:Id])  # this gives values from 0 to Id-1

    Loop_ops = 1    # Loop_ops: number of operations within a tile before which kh/kw loop comes in a SIMD ALU
    for key in Loop_order[0:Id]:
        Loop_ops = Loop_ops * Loop_dict[key]

    #print("Loop_ops:", Loop_ops)  
    if Loop_ops < 3:
        print("WARNING: Model for additional SIMD stall cycles due to data dependency in GeneSys do not exist for the input loop order for the pool layer")


    ########## compute cycle model
    cycle_oneTile_cmn = ((Stile_oh * Stile_ow * Stile_oc * Stile_batch) / SysArray_col) * (DTile_oh/Stile_oh) * (DTile_ow/Stile_ow) * (DTile_oc/Stile_oc) * (DTile_batch/Stile_batch) \
                                                                           * (DTile_kh/Stile_kh) * (DTile_kw/Stile_kw) 

    cy_oneTile_div = 0
    cycle_oneTile_div = 0  #additional cycles due to the division operation in Average Pool
    if Layer_name == "AvgPool":
        cy_oneTile_div = ((Stile_oh * Stile_ow * Stile_oc * Stile_batch) / SysArray_col) * (DTile_oh/Stile_oh) * (DTile_ow/Stile_ow) * (DTile_oc/Stile_oc) * (DTile_batch/Stile_batch)
        cycle_oneTile_div = cy_oneTile_div * div_cycles  # division operation takes > 1 cycles 

    cycle_oneTile = cycle_oneTile_cmn + cycle_oneTile_div  #of cycles for one tile
    #print("cycle_oneTile:", cycle_oneTile)

    #of cycles to fill the pipeline for a tile, there are 6 pipeline stages
    pipe_overhead_tile = (6 - 1) + (SysArray_col - 1)  #using PE col, 

    #of cycles to compute one tile including the pipeline setup operhead, need this variable to compute DRAM stall cycles
    ComputeTile_cycles = cycle_oneTile + pipe_overhead_tile

    #for now omitting the use of any ceil since DRAM tile size will be integer multiple of loops, 
    Number_of_Tile = (OH/DTile_oh) * (OW/DTile_ow) * (OC/DTile_oc) * (Batch/DTile_batch)
    compute_cycles = math.ceil((cycle_oneTile + pipe_overhead_tile) * Number_of_Tile)   # giving the outer ceil to avoid fraction cycle numbers


    ######## model for the DRAM stall cycles, depends on fusion etc
    if (fusion_status == "NoFusion"): #Model for the version where there is no fusion
        if (Hardware_param.Buffering_scheme_VMEM == "single"):
            #of cycles required to load/store each tile of each kind of data, both VMEM shares the same AXI 
            ifmapTile_load_cycles = math.ceil((DTile_iw * DTile_ih * DTile_oc * DTile_batch * bw_ifmap) / RBw_DRAM_to_VMEM)
            ofmapTile_store_cycles = math.ceil((DTile_ow * DTile_oh * DTile_oc * DTile_batch * bw_ofmap) / WBw_VMEM_to_DRAM)

            # VMEM is single buffered, so there is no overlap with computation.
            ifmap_stall_cycles = ifmapTile_load_cycles * Number_of_Tile
            ofmap_stall_cycles = ofmapTile_store_cycles * Number_of_Tile
            DRAM_stall_cycles = ifmap_stall_cycles + ofmap_stall_cycles
        else:
            print("Model do not exist for VMEM non single buffer scheme now")
    else:
        print("model for fusion do not exist yet")

    #print("compute_cycles:", compute_cycles)
    #print("DRAM_stall_cycles:", DRAM_stall_cycles)
    SIMDResult_inflayer.cycles['compute'] = compute_cycles
    SIMDResult_inflayer.cycles['DRAM_stall'] = DRAM_stall_cycles
    SIMDResult_inflayer.cycles['total'] = compute_cycles + DRAM_stall_cycles

    # other accesses which happens at every compute cycle
    # using a common function for all SIMD class high layers to calculate the access from pipeline registers, instruction memory, and index tables
    #of total compute cycles without pipeline overhead and counting one cycle for each div. IF should happen once for each div operation 
    comp_cycles_ideal = math.ceil((cycle_oneTile_cmn + cy_oneTile_div) * Number_of_Tile) 
    #print("comp_cycles_ideal:", comp_cycles_ideal)
    other_access_SIMDhigh(Hardware_param, LayerObj, SIMDResult_inflayer, comp_cycles_ideal)


def softmax_access_model(Hardware_param, LayerObj, SIMDResult_inflayer):
    #data access model for softmax layer from high level description 

    bw_ifmap = LayerObj.bw_ifmap 
    bw_psum = LayerObj.bw_psum # intermediate data
    bw_ofmap = LayerObj.bw_ofmap 

    OW = LayerObj.OW
    OH = LayerObj.OH
    OC = LayerObj.OC
    Batch = LayerObj.Batch

    DTile_ow = LayerObj.DTile_ow
    DTile_oh = LayerObj.DTile_oh
    DTile_oc = LayerObj.DTile_oc
    DTile_batch = LayerObj.DTile_batch

    Stile_ow = LayerObj.Stile_ow
    Stile_oh = LayerObj.Stile_oh
    Stile_oc = LayerObj.Stile_oc
    Stile_batch = LayerObj.Stile_batch

    fusion_status = LayerObj.fusion_status
    #Loop order does not matter for softmax layer

    ############ DRAM Access
    if (fusion_status == "NoFusion"):
        # This access is for step1: e^x, ifmap access is read access, and psum access is write access
        ifmap_access_DRAM = (DTile_oh * DTile_ow * DTile_oc * DTile_batch) * (OH/DTile_oh) * (OW/DTile_ow) * (OC/DTile_oc) * (Batch/DTile_batch) * bw_ifmap
        psum_access_DRAM1 = (DTile_oh * DTile_ow * DTile_oc * DTile_batch) * (OH/DTile_oh) * (OW/DTile_ow) * (OC/DTile_oc) * (Batch/DTile_batch) * bw_psum

        # There is no DRAM access for step2: reduction, in my currect scheduling, 

        #This access is for step3: division; psum access is read access, and ofmap access is write access
        psum_access_DRAM2 = (DTile_oh * DTile_ow * DTile_oc * DTile_batch) * (OH/DTile_oh) * (OW/DTile_ow) * (OC/DTile_oc) * (Batch/DTile_batch) * bw_psum
        ofmap_access_DRAM = (DTile_oh * DTile_ow * DTile_oc * DTile_batch) * (OH/DTile_oh) * (OW/DTile_ow) * (OC/DTile_oc) * (Batch/DTile_batch) * bw_ofmap

        psum_access_DRAM = psum_access_DRAM1 + psum_access_DRAM2

        SIMDResult_inflayer.DRAM_access['ifmap'] = ifmap_access_DRAM
        SIMDResult_inflayer.DRAM_access['ofmap'] = ofmap_access_DRAM
        SIMDResult_inflayer.DRAM_access['intermediate'] = psum_access_DRAM

    else:
        print("model for fusion do not exist yet")

    ########## On-chip SRAM Access
    ## Access for VMEM: In the current mapping, Batch is mapped across different SIMD lane, 
    # Also using math.ceil for batch diemnsion assuming zero padding when Batch = 1, uisng the ceil fix only here cause it will always happen for Batch = 1
    #print("Stile_batch:", Stile_batch, "Stile_oc:", Stile_oc)
    if (fusion_status == "NoFusion"):
        ##### Step1: e^x, ifmap access is read access and psum access is write access
        ifmap_access_VMEM = (Stile_oh * Stile_ow * Stile_oc * Stile_batch) * (DTile_oh/Stile_oh) * (DTile_ow/Stile_ow) * (DTile_oc/Stile_oc)\
                                * math.ceil((DTile_batch/Stile_batch)) * (OH/DTile_oh) * (OW/DTile_ow) * (OC/DTile_oc) * (Batch/DTile_batch) * bw_ifmap
        ifmap_access_OBUF = 0  # for fusion-1, the ifmap will be accessed from OBUF instead of VMEM, hence keepig this parameter
        psum_access_VMEM1 = (Stile_oh * Stile_ow * Stile_oc * Stile_batch) * (DTile_oh/Stile_oh) * (DTile_ow/Stile_ow) * (DTile_oc/Stile_oc)\
                                * math.ceil((DTile_batch/Stile_batch)) * (OH/DTile_oh) * (OW/DTile_ow) * (OC/DTile_oc) * (Batch/DTile_batch) * bw_psum

        ###### Step2: reduction, equation is same as psum_access_VMEM1, two operand read and one operand write for each addition of the reduction process
        psum_access_VMEM2 = psum_access_VMEM1 * (2 + 1)

        ###### Step3: Division, for each division, two operand read access and one operand write access
        psum_access_VMEM3 = psum_access_VMEM1 * 2
        # for now assuming quantization is free and happen inside the ALU before writing ofmap back to VMEM
        ofmap_access_VMEM = (Stile_oh * Stile_ow * Stile_oc * Stile_batch) * (DTile_oh/Stile_oh) * (DTile_ow/Stile_ow) * (DTile_oc/Stile_oc) \
                                    * math.ceil((DTile_batch/Stile_batch)) * (OH/DTile_oh) * (OW/DTile_ow) * (OC/DTile_oc) * (Batch/DTile_batch) * bw_ofmap

        psum_access_VMEM = psum_access_VMEM1 + psum_access_VMEM2 + psum_access_VMEM3

        #print("ifmap_access_VMEM:", ifmap_access_VMEM)
        #print("ofmap_access_VMEM:", ofmap_access_VMEM)
        SIMDResult_inflayer.SRAM_access['VMEM'] = ifmap_access_VMEM + psum_access_VMEM + ofmap_access_VMEM
        # for this division operation one operand is constant. chances are this constant will be read from IMM and passes through the pipeline registers,
        # for simplicity now considering access from VMEM and not from IMM
    
    else:       
        print("model for fusion do not exist yet")

    ## Number of arithmetic operations, using math.ceil for batch diemnsion assuming zero padding when Batch = 1
    Nos_of_exp = (Stile_oh * Stile_ow * Stile_oc * Stile_batch) * (DTile_oh/Stile_oh) * (DTile_ow/Stile_ow) * (DTile_oc/Stile_oc) * math.ceil((DTile_batch/Stile_batch)) \
                                        * (OH/DTile_oh) * (OW/DTile_ow) * (OC/DTile_oc) * (Batch/DTile_batch)
    Nos_of_add = Nos_of_exp
    Nos_of_div = Nos_of_exp

    SIMDResult_inflayer.arithmetic["exp"] = Nos_of_exp
    SIMDResult_inflayer.arithmetic["add"] = Nos_of_add
    SIMDResult_inflayer.arithmetic["div"] = Nos_of_div

def softmax_cycle_model(Hardware_param, LayerObj, SIMDResult_inflayer):
    #compute cycle and DRAM stall cycle count model for the softmax layer

    bw_ifmap = LayerObj.bw_ifmap 
    bw_psum = LayerObj.bw_psum
    bw_ofmap = LayerObj.bw_ofmap 
    div_cycles = Hardware_param.div_cycles
    exp_cycles = Hardware_param.exp_cycles

    SysArray_col = Hardware_param.SysArray_col
    RBw_DRAM_to_VMEM = Hardware_param.RBw_DRAM_to_VMEM
    WBw_VMEM_to_DRAM = Hardware_param.WBw_VMEM_to_DRAM

    OW = LayerObj.OW
    OH = LayerObj.OH
    OC = LayerObj.OC
    Batch = LayerObj.Batch

    DTile_ow = LayerObj.DTile_ow
    DTile_oh = LayerObj.DTile_oh
    DTile_oc = LayerObj.DTile_oc
    DTile_batch = LayerObj.DTile_batch

    Stile_ow = LayerObj.Stile_ow
    Stile_oh = LayerObj.Stile_oh
    Stile_oc = LayerObj.Stile_oc
    Stile_batch = LayerObj.Stile_batch

    fusion_status = LayerObj.fusion_status

    #Common cycle count equation for one tile, k operations per cycle, compute cycles do not depend on loop order or fusion
    cycle_oneTile_eqn = (Stile_oh * Stile_ow * Stile_oc) * (DTile_oh/Stile_oh) * (DTile_ow/Stile_ow) * (DTile_oc/Stile_oc) * math.ceil((DTile_batch/Stile_batch))
    #of cycles to fill the pipeline for a tile, there are 6 pipeline stages
    pipe_overhead_tile = (6 - 1) + (SysArray_col - 1)  #using PE col, 

    # need to add the pipeline overhead for all three steps because there is DRAM load/store in between each step of computation
    ##Step1: exponential
    cycle_oneTile_exp = cycle_oneTile_eqn * exp_cycles  #exp takes more than one cycles to compute
    #of cycles to compute one tile including the pipeline setup operhead, need this variable to compute DRAM stall cycles
    ComputeTile_cycles_exp = cycle_oneTile_exp + pipe_overhead_tile

    ##Step2: addition to reduce
    cycle_oneTile_add = cycle_oneTile_eqn
    ComputeTile_cycles_add = cycle_oneTile_add + pipe_overhead_tile

    ##Step3: division
    cycle_oneTile_div = cycle_oneTile_eqn * div_cycles
    ComputeTile_cycles_div = cycle_oneTile_div + pipe_overhead_tile

    #for now omitting the use of any ceil since DRAM tile size will be integer multiple of loops, 
    Number_of_Tile = (OH/DTile_oh) * (OW/DTile_ow) * (OC/DTile_oc) * (Batch/DTile_batch)
    compute_cycles = math.ceil((ComputeTile_cycles_exp + ComputeTile_cycles_add + ComputeTile_cycles_div) * Number_of_Tile)  #giving outer ceil to avoid fraction cycles

    ######## model for the DRAM stall cycles, depends on fusion etc
    if (fusion_status == "NoFusion"): #Model for the version where there is no fusion
        if (Hardware_param.Buffering_scheme_VMEM == "single"):
            #of cycles required to load/store each tile of each kind of data, both VMEM shares the same AXI 
            data_tile_eqn = DTile_ow * DTile_oh * DTile_oc * DTile_batch
            ifmapTile_load_cycles = math.ceil((data_tile_eqn * bw_ifmap) / RBw_DRAM_to_VMEM)
            psum_store_cycles = math.ceil((data_tile_eqn * bw_psum) / WBw_VMEM_to_DRAM)
            psum_load_cycles = math.ceil((data_tile_eqn * bw_psum) / RBw_DRAM_to_VMEM)
            ofmapTile_store_cycles = math.ceil((data_tile_eqn * bw_ofmap) / WBw_VMEM_to_DRAM)

            # VMEM is single buffered, so there is no overlap with computation.
            ifmap_stall_cycles = ifmapTile_load_cycles * Number_of_Tile
            psum_stall_cycles = (psum_store_cycles + psum_load_cycles) * Number_of_Tile
            ofmap_stall_cycles = ofmapTile_store_cycles * Number_of_Tile
            DRAM_stall_cycles = ifmap_stall_cycles + psum_stall_cycles + ofmap_stall_cycles
        else:
            print("Model do not exist for VMEM non single buffer scheme now")
    else:
        print("model for fusion do not exist yet")

    #print("compute_cycles:", compute_cycles)
    #print("DRAM_stall_cycles:", DRAM_stall_cycles)
    SIMDResult_inflayer.cycles['compute'] = compute_cycles
    SIMDResult_inflayer.cycles['DRAM_stall'] = DRAM_stall_cycles
    SIMDResult_inflayer.cycles['total'] = compute_cycles + DRAM_stall_cycles

    # other accesses which happens at every compute cycle
    # using a common function for all SIMD class high layers to calculate the access from pipeline registers, instruction memory, and index tables
    #of total compute cycles without pipeline overhead and counting one cycle for each div & exp. IF should happen once for each div/exp operation, 
    comp_cycles_ideal = math.ceil(cycle_oneTile_eqn * 3 * Number_of_Tile)  # 3 for the three steps, #of total compute cycles without pipeline overhead
    #print("comp_cycles_ideal:", comp_cycles_ideal)
    other_access_SIMDhigh(Hardware_param, LayerObj, SIMDResult_inflayer, comp_cycles_ideal)
    #NOTE: FOR SOFTMAX STEP1 IS ONE OPERAND WHILE STEP2,3 IS TWO OPERAND, PUTTING SOFTMAX IN TWO OPERAND CATEGORY FOR SIMPLICITY FOR NOW


def roialignpool_access_model(Hardware_param, LayerObj, SIMDResult_inflayer):
    # data access model for RoiAlignPool layer
    ## the Level projection part specific for FPN backbone is omitting now, later will add that either in the same function or as a seperate operation

    bw_ifmap = LayerObj.bw_ifmap 
    bw_psum = LayerObj.bw_psum
    bw_ofmap = LayerObj.bw_ofmap  

    OW = LayerObj.OW
    OH = LayerObj.OH
    OC = LayerObj.OC
    Batch = LayerObj.Batch
    RoI = LayerObj.RoI

    DTile_ow = LayerObj.DTile_ow
    DTile_oh = LayerObj.DTile_oh
    DTile_oc = LayerObj.DTile_oc

    Stile_ow = LayerObj.Stile_ow
    Stile_oh = LayerObj.Stile_oh
    Stile_oc = LayerObj.Stile_oc

    fusion_status = LayerObj.fusion_status

    box_coordinate = 4  #there are 4 co-ordinates for each RoI
    pixel_per_ofmap = 4 # Q11, Q12, Q21, Q22--> there are four pixels to load for the bilinear interpolation for each ofmap location

    #Loop order does not matter for RoiAlignPool layer
    #All computation is batch serialized and RoI serialized
    ############ DRAM Access
    if (fusion_status == "NoFusion"):
        #step 1:
        ifmap_access_DRAM1 = (box_coordinate + 1) * RoI * Batch * bw_ifmap
        #step 7:
        ifmap_access_DRAM2 = pixel_per_ofmap * (DTile_ow * DTile_oh * DTile_oc) * (OW/DTile_ow) * (OH/DTile_oh) * (OC/DTile_oc) * RoI * Batch * bw_ifmap
        ifmap_access_DRAM = ifmap_access_DRAM1 + ifmap_access_DRAM2
        #print("ifmap_access_DRAM1:", ifmap_access_DRAM1, "ifmap_access_DRAM2:", ifmap_access_DRAM2)

        # step 9:
        ofmap_access_DRAM = (DTile_ow * DTile_oh * DTile_oc) * (OW/DTile_ow) * (OH/DTile_oh) * (OC/DTile_oc) * RoI * Batch * bw_ofmap

        SIMDResult_inflayer.DRAM_access['ifmap'] = ifmap_access_DRAM
        SIMDResult_inflayer.DRAM_access['ofmap'] = ofmap_access_DRAM
    else:
        print("model for fusion do not exist yet")

    ############## Number of arithmetic operations
    ###Step 2 to 4: Assuming one ALU is being used
    obtain_levelstride_div = 1
    roi_coordinate_div = box_coordinate 
    roi_box_div = box_coordinate - 2
    div_step234 = (obtain_levelstride_div + roi_coordinate_div + roi_box_div) * RoI * Batch
    #print("div_step234:", div_step234)

    ###Step 5, 6, 8:
    # Step 5: determine subbox center
    add_s5 = 4
    mul_s5 = 2 + 2
    #Step 6: obtain the co-ordinate of four neighboring pixels
    add_s6 = 2
    sub_s6 = 2
    div_s6 = 2
    #Step 8: apply bilinear interpolation equation (optimized)
    add_s8 = 3
    sub_s8 = 8
    mul_s8 = 8
    div_s8 = 1
    
    loop_mul_step568 = (Stile_oh * Stile_ow * Stile_oc) * (DTile_oh/Stile_oh) * (DTile_ow/Stile_ow) * (DTile_oc/Stile_oc) \
                                                                                    * (OW/DTile_ow) * (OH/DTile_oh) * (OC/DTile_oc) * RoI * Batch 

    add_step568 = (add_s5 + add_s6 + add_s8) * loop_mul_step568
    sub_step568 = (sub_s6 + sub_s8) * loop_mul_step568
    mul_step568 = (mul_s5 + mul_s8) * loop_mul_step568
    div_step568 = (div_s6 + div_s8) * loop_mul_step568

    Nos_of_add = add_step568
    Nos_of_sub = sub_step568
    Nos_of_mul = mul_step568
    Nos_of_div = div_step568 + div_step234

    SIMDResult_inflayer.arithmetic["add"] = Nos_of_add
    SIMDResult_inflayer.arithmetic["sub"] = Nos_of_sub
    SIMDResult_inflayer.arithmetic["mul"] = Nos_of_mul
    SIMDResult_inflayer.arithmetic["div"] = Nos_of_div

    
    ########## On-chip SRAM Access:
    # Access for VMEM, although there will be some IMM access, keeping all access as VMEM for simplicity
    # Even for fusion, the ifmap has to come from DRAM to VMEM
    if (fusion_status == "NoFusion"):
        # For each arithmetic operation, there is 2 read access and one write access, all are two operand operations
        ofpsm_access_VMEM = (2 + 1) * (Nos_of_add + Nos_of_sub + Nos_of_mul + Nos_of_div)
        #Using bw_psum cause most operations involve input and intermediate data both
        psum_access_VMEM = (ofpsm_access_VMEM - (OW * OH * OC * RoI * Batch)) * bw_psum  #intermediate data is 32 bit (before quantization)
        # final ofmap access: for now assuming quantization is free and happen inside the ALU before writing ofmap back to VMEM 
        ofmap_access_VMEM = (OW * OH * OC * RoI * Batch) * bw_ofmap   #May change depending on quantization decision, FIX LATER

        SIMDResult_inflayer.SRAM_access['VMEM'] = psum_access_VMEM + ofmap_access_VMEM
    else:       
        print("model for fusion do not exist yet")
    

def roialignpool_cycle_model(Hardware_param, LayerObj, SIMDResult_inflayer):
    # cycle count and DRAM stall model for RoiAlignPool layer
    ## the Level projection part specific for FPN backbone is omitting now, later will add that either in the same function or as a seperate operation

    bw_ifmap = LayerObj.bw_ifmap 
    bw_psum = LayerObj.bw_psum
    bw_ofmap = LayerObj.bw_ofmap  

    SysArray_col = Hardware_param.SysArray_col
    RBw_DRAM_to_VMEM = Hardware_param.RBw_DRAM_to_VMEM
    WBw_VMEM_to_DRAM = Hardware_param.WBw_VMEM_to_DRAM
    div_cycles = Hardware_param.div_cycles

    OW = LayerObj.OW
    OH = LayerObj.OH
    OC = LayerObj.OC
    Batch = LayerObj.Batch
    RoI = LayerObj.RoI

    DTile_ow = LayerObj.DTile_ow
    DTile_oh = LayerObj.DTile_oh
    DTile_oc = LayerObj.DTile_oc

    Stile_ow = LayerObj.Stile_ow
    Stile_oh = LayerObj.Stile_oh
    Stile_oc = LayerObj.Stile_oc

    fusion_status = LayerObj.fusion_status

    box_coordinate = 4  #there are 4 co-ordinates for each RoI
    pixel_per_ofmap = 4 # Q11, Q12, Q21, Q22--> there are four pixels to load for the bilinear interpolation for each ofmap location
    
    ############## Number of arithmetic operations (Copying the similar code from access count model)
    ###Step 2 to 4: Assuming one ALU is being used
    obtain_levelstride_div = 1
    roi_coordinate_div = box_coordinate 
    roi_box_div = box_coordinate - 2
    div_step234 = (obtain_levelstride_div + roi_coordinate_div + roi_box_div) * RoI * Batch
    #print("div_step234:", div_step234)

    cycle_div_step234 = math.ceil(div_step234 * div_cycles) # since only one ALU is being used, assuming no setup overhead for this part

    ###Step 5, 6, 8: Using All ALUs, mapping OC dimension across different SIMD ALUs
    # Step 5: determine subbox center
    add_s5 = 4
    mul_s5 = 2 + 2
    #Step 6: obtain the co-ordinate of four neighboring pixels
    add_s6 = 2
    sub_s6 = 2
    div_s6 = 2
    #Step 8: apply bilinear interpolation equation (optimized)
    add_s8 = 3
    sub_s8 = 8
    mul_s8 = 8
    div_s8 = 1

    # These gives the number of arithmetic operation per ofmap location
    add_step568 = (add_s5 + add_s6 + add_s8) 
    sub_step568 = (sub_s6 + sub_s8) 
    mul_step568 = (mul_s5 + mul_s8) 
    div_step568 = (div_s6 + div_s8)

    # This is the multiplier for cycle count to cover one tile per RoI and per batch, k operations per cycle, compute cycles do not depend on loop order or fusion
    cy_oneTile_step568_mul = (Stile_oh * Stile_ow) * (DTile_oh/Stile_oh) * (DTile_ow/Stile_ow) * math.ceil((DTile_oc/Stile_oc))
    #cycle count to compute one tile
    cycle_oneTile_step568 = (add_step568 + sub_step568 + mul_step568 + (div_s8 * div_cycles)) * cy_oneTile_step568_mul
    #print("cycle_oneTile_step568:", cycle_oneTile_step568)

    #of cycles to fill the pipeline for a tile, there are 6 pipeline stages
    pipe_overhead_tile = (6 - 1) + (SysArray_col - 1)  #using PE col, 

    #of cycles to compute one tile including the pipeline setup operhead
    ComputeTile_cycles_step568 = cycle_oneTile_step568 + pipe_overhead_tile

    #for now omitting the use of any ceil since DRAM tile size will be integer multiple of loops, 
    Number_of_Tile = (OH/DTile_oh) * (OW/DTile_ow) * (OC/DTile_oc) * RoI * Batch
    compute_cycles_step568 = math.ceil(ComputeTile_cycles_step568 * Number_of_Tile)   # giving the outer ceil to avoid fraction cycle numbers

    # Total compute cycles across all steps
    compute_cycles = cycle_div_step234 + compute_cycles_step568

    ######## model for the DRAM stall cycles, depends on fusion etc
    if (fusion_status == "NoFusion"): #Model for the version where there is no fusion
        if (Hardware_param.Buffering_scheme_VMEM == "single"):
            #of cycles required to load/store each tile of each kind of data, both VMEM shares the same AXI 

            # load cycles for step1 (utilizing equations from DRAM access model)
            #step 1:
            ifmap_load_cycles_step1 = math.ceil(((box_coordinate + 1) * bw_ifmap)/RBw_DRAM_to_VMEM) # at a time 4+1 data is loaded at step1
            ifmap_stall_cycles_step1 = ifmap_load_cycles_step1 * RoI * Batch    # step1 is repeated for all RoI and Batch            

            #step 7:
            ifmap_load_cycles_step7 = math.ceil((pixel_per_ofmap * bw_ifmap)/RBw_DRAM_to_VMEM) # at a time 4 pixel data is loaded at step7
            ifmap_stall_cycles_step7 = ifmap_load_cycles_step7 * (DTile_ow * DTile_oh * DTile_oc) * (OW/DTile_ow) * (OH/DTile_oh) * (OC/DTile_oc) * RoI * Batch
            
            #step 9:
            ofmapTile_store_cycles = math.ceil((DTile_ow * DTile_oh * DTile_oc * bw_ofmap) / WBw_VMEM_to_DRAM) # at a time one ofmap tile is stored
            ofmap_stall_cycles_step9 = ofmapTile_store_cycles * Number_of_Tile

            # VMEM is single buffered, so there is no overlap with computation.
            DRAM_stall_cycles = ifmap_stall_cycles_step1 + ifmap_stall_cycles_step7 + ofmap_stall_cycles_step9
        else:
            print("Model do not exist for VMEM non single buffer scheme now")
    else:
        print("model for fusion do not exist yet")

    #print("compute_cycles:", compute_cycles)
    #print("DRAM_stall_cycles:", DRAM_stall_cycles)
    SIMDResult_inflayer.cycles['compute'] = compute_cycles
    SIMDResult_inflayer.cycles['DRAM_stall'] = DRAM_stall_cycles
    SIMDResult_inflayer.cycles['total'] = compute_cycles + DRAM_stall_cycles
    
    # other accesses which happens at every compute cycle
    # using a common function for all SIMD class high layers to calculate the access from pipeline registers, instruction memory, and index tables
    #of total compute cycles without pipeline overhead and counting one cycle for each div. IF should happen once for each div operation, 
    #Step568:
    cycle_oneTile_s568_ideal = (add_step568 + sub_step568 + mul_step568 + div_s8) * cy_oneTile_step568_mul
    compute_cycles_s568_ideal = cycle_oneTile_s568_ideal * Number_of_Tile
    #Step568 & Step234
    comp_cycles_ideal = math.ceil(compute_cycles_s568_ideal + div_step234)
    #print("comp_cycles_ideal:", comp_cycles_ideal)
    other_access_SIMDhigh(Hardware_param, LayerObj, SIMDResult_inflayer, comp_cycles_ideal)













    














    



















