#This file contains the additional functions to support training

import logging
import math
import numpy as np
from data_objects import HardwareObject, SAResult_Inflayer, SIMDResult_Inflayer
from layer_object import LayerObject

def common_SIMD_backward_model(Hardware_param, LayerObj, SIMDResult_inflayer):
    # This is a common function for several SIMD operation during the backward pass

    #unpacking the parameters
    CompilerOut_layer = LayerObj.CompilerOut_layer  # the full DNN Spec for the layer
    #print(CompilerOut_layer)
    bw_ifmap = LayerObj.bw_ifmap 
    bw_psum = LayerObj.bw_psum
    bw_ofmap = LayerObj.bw_ofmap  

    SysArray_col = Hardware_param.SysArray_col
    SIMD_Ins_size = Hardware_param.SIMD_Ins_size
    RBw_DRAM_to_VMEM = Hardware_param.RBw_DRAM_to_VMEM
    WBw_VMEM_to_DRAM = Hardware_param.WBw_VMEM_to_DRAM

    layer_op = CompilerOut_layer['operation']
    #print("operation_name:", layer_op)


    ##################Computing DRAM access
    #### INPUT-1
    # there will be at least one input, extracting the dictionary keys for the first input
    input1_keys = list(CompilerOut_layer['inputs'][0]['shape_symbols'].keys())
    #print("input1_keys:", input1_keys)
    DRAM_in1 = CompilerOut_layer['inputs'][0]['tiling']['DRAM'] # total DRAM data for input-1
    #print("DRAM_in1:", DRAM_in1)
    DRAM_access_num_in1 = 1
    for key in input1_keys:
        DRAM_access_num_in1 = DRAM_access_num_in1 * DRAM_in1[key]
    #print("DRAM_access_num_in1:", DRAM_access_num_in1)
    DRAM_access_in1 = DRAM_access_num_in1 * bw_ifmap


    ##### INPUT-2 (if any)
    Nos_of_input = len(CompilerOut_layer['inputs'])
    if Nos_of_input > 1:
        input2_keys = list(CompilerOut_layer['inputs'][1]['shape_symbols'].keys())
        #print("input2_keys:", input2_keys)
        DRAM_in2 = CompilerOut_layer['inputs'][1]['tiling']['DRAM'] # total DRAM data for input-2
        #print("DRAM_in2:", DRAM_in2)
        DRAM_access_num_in2 = 1
        for key in input2_keys:
            DRAM_access_num_in2 = DRAM_access_num_in2 * DRAM_in2[key]
        #print("DRAM_access_num_in2:", DRAM_access_num_in2)
        DRAM_access_in2 = DRAM_access_num_in2 * bw_ifmap
    else:
        DRAM_access_in2 = 0
    #print("DRAM_access_in2:", DRAM_access_in2)
    ifmap_access_DRAM = DRAM_access_in1 + DRAM_access_in2


    ###### Output
    out_keys = list(CompilerOut_layer['outputs'][0]['shape_symbols'].keys())
    #print("out_keys:", out_keys)
    DRAM_out = CompilerOut_layer['outputs'][0]['tiling']['DRAM'] # total DRAM data tile for output
    #print("DRAM_out:", DRAM_out)
    DRAM_access_num_out = 1
    for key in out_keys:
        DRAM_access_num_out = DRAM_access_num_out * DRAM_out[key]
    #print("DRAM_access_num_out:", DRAM_access_num_out)
    ofmap_access_DRAM = DRAM_access_num_out * bw_ofmap

    #print("ifmap_access_DRAM:", ifmap_access_DRAM)
    #print("ofmap_access_DRAM:", ofmap_access_DRAM)
    SIMDResult_inflayer.DRAM_access['ifmap'] = ifmap_access_DRAM
    SIMDResult_inflayer.DRAM_access['ofmap'] = ofmap_access_DRAM


    ################## Computing VMEM access: default: no fusion
    # dictionary to hold the number of computation step within each tile of a layer to produce an output, 
    comp_step_rsum = "None"
    if layer_op == "reduce_sum":
        comp_step_rsum = CompilerOut_layer['inputs'][0]['shape_symbols']['N']
        #print("comp_step_rsum:", comp_step_rsum)

    comp_step_dict = {'sgd4d' : 2, 'sgd2d': 2, 'sgd1d': 2, 'relu_grad': 1, 'elem_add_grad': 1, 'reduce_sum': comp_step_rsum}
    #print("comp_step_dict:", comp_step_dict)

    ifmap_access_VMEM = 2 * DRAM_access_num_out * bw_ifmap * comp_step_dict[layer_op]  # for each computation 2 operand read and one operand write
    psum_access_VMEM = DRAM_access_num_out * bw_psum * (comp_step_dict[layer_op] - 1)
    # assuming quantization is free and happen inside the ALU before writing ofmap back to VMEM
    ofmap_access_VMEM = DRAM_access_num_out * bw_ofmap

    #print("ifmap_access_VMEM:", ifmap_access_VMEM)
    #print("psum_access_VMEM:", psum_access_VMEM)
    #print("ofmap_access_VMEM:", ofmap_access_VMEM)
    SIMDResult_inflayer.SRAM_access['VMEM'] = ifmap_access_VMEM + psum_access_VMEM + ofmap_access_VMEM

    ########################## Computing cycle counts
    tile_keys_out = list(CompilerOut_layer['outputs'][0]['tiling'].keys())
    #print("tile_keys_out:", tile_keys_out)

    for key in tile_keys_out:
        if key == 'VMEM1' or key == 'VMEM2':
            VMEM_key_out = key   # VMEM key can be either VMEM1 or VMEM2, storing that key so that can use
            break
    #print("VMEM_key_out:",VMEM_key_out)

    VMEM_out = CompilerOut_layer['outputs'][0]['tiling'][VMEM_key_out] # output tile volume for VMEM
    #print("VMEM_out:", VMEM_out)

    VMEM_vol_out = 1
    for key in out_keys:
        VMEM_vol_out = VMEM_vol_out * VMEM_out[key]
    #print("VMEM_vol_out:", VMEM_vol_out)

    #of cycles to compute one VMEM tile, k operations per cycle
    cycle_oneTile = math.ceil(VMEM_vol_out/SysArray_col) * comp_step_dict[layer_op]
    #print("cycle_oneTile:", cycle_oneTile)

    #of cycles to fill the pipeline for a tile, there are 6 pipeline stages
    pipe_overhead_tile = (6 - 1) + (SysArray_col - 1)  
    
    Number_of_Tile = 1
    for key in out_keys:
        Number_of_Tile = Number_of_Tile * (DRAM_out[key]/VMEM_out[key])
    #print("Number_of_Tile:", Number_of_Tile)

    compute_cycles = math.ceil((cycle_oneTile + pipe_overhead_tile) * Number_of_Tile)   # giving the outer ceil to avoid fraction cycle numbers

    ################################ Computing DRAM Stall cycles
    #INPUT-1
    tile_keys_in1 = list(CompilerOut_layer['inputs'][0]['tiling'].keys())
    #print("tile_keys_in1:", tile_keys_in1)
    for key in tile_keys_in1:
        if key == 'VMEM1' or key == 'VMEM2':
            VMEM_key_in1 = key   # VMEM key can be either VMEM1 or VMEM2, storing that key so that can use
            break
    #print("VMEM_key_in1:",VMEM_key_in1)
    VMEM_in1 = CompilerOut_layer['inputs'][0]['tiling'][VMEM_key_in1] # input1 tile volume for VMEM
    #print("VMEM_in1:", VMEM_in1)
    VMEM_vol_in1 = 1
    for key in input1_keys:
        VMEM_vol_in1 = VMEM_vol_in1 * VMEM_in1[key]
    #print("VMEM_vol_in1:", VMEM_vol_in1)
    ifmapTile_load_cycles_1 = math.ceil((VMEM_vol_in1 * bw_ifmap) / RBw_DRAM_to_VMEM)

    #INPUT-2 (if any)
    if Nos_of_input > 1:
        tile_keys_in2 = list(CompilerOut_layer['inputs'][1]['tiling'].keys())
        for key in tile_keys_in2:
            if key == 'VMEM1' or key == 'VMEM2':
                VMEM_key_in2 = key   # VMEM key can be either VMEM1 or VMEM2, storing that key so that can use
                break
        #print("VMEM_key_in2:",VMEM_key_in2)
        VMEM_in2 = CompilerOut_layer['inputs'][1]['tiling'][VMEM_key_in2] # input1 tile volume for VMEM
        #print("VMEM_in2:", VMEM_in2)

        VMEM_vol_in2 = 1
        for key in input2_keys:
            VMEM_vol_in2 = VMEM_vol_in2 * VMEM_in2[key]
        #print("VMEM_vol_in2:", VMEM_vol_in2)
        ifmapTile_load_cycles_2 = math.ceil((VMEM_vol_in2 * bw_ifmap) / RBw_DRAM_to_VMEM)
    else:
        ifmapTile_load_cycles_2 = 0

    #Output
    ofmapTile_store_cycles = math.ceil((VMEM_vol_out * bw_ofmap) / WBw_VMEM_to_DRAM)

    # VMEM is single buffered, so there is no overlap with computation.
    ifmap_stall_cycles_1 = ifmapTile_load_cycles_1 * Number_of_Tile
    ifmap_stall_cycles_2 = ifmapTile_load_cycles_2 * Number_of_Tile
    ofmap_stall_cycles = ofmapTile_store_cycles * Number_of_Tile
    DRAM_stall_cycles = ifmap_stall_cycles_1 + ifmap_stall_cycles_2 + ofmap_stall_cycles

    #print("compute_cycles:", compute_cycles)
    #print("DRAM_stall_cycles:", DRAM_stall_cycles)
    SIMDResult_inflayer.cycles['compute'] = compute_cycles
    SIMDResult_inflayer.cycles['DRAM_stall'] = DRAM_stall_cycles
    SIMDResult_inflayer.cycles['total'] = compute_cycles + DRAM_stall_cycles

    ########################## Access to Instruction Memory
    comp_cycles_ideal = math.ceil(cycle_oneTile * Number_of_Tile)   #of total compute cycles without pipeline overhead
    #Access for IF stage
    Nos_of_IF = comp_cycles_ideal  #number of instruction fetched
    InsMem_access = Nos_of_IF * SIMD_Ins_size    # in bit

    SIMDResult_inflayer.SRAM_access['InsMem'] = InsMem_access

    #################### Conputing number of arithmetic Ops for each kind of layer
    Nos_of_op_tile = VMEM_vol_out * comp_step_dict[layer_op]
    Nos_of_op = Nos_of_op_tile * Number_of_Tile

    #print("Nos_of_op_tile:", Nos_of_op_tile, "Nos_of_op:", Nos_of_op)
    SIMDResult_inflayer.arithmetic['op_ScmnBN'] = Nos_of_op


def batch_norm_forward_estimate(Hardware_param, LayerObj, SIMDResult_inflayer):
    # This function analytically estimates the cost for Banch_norm forward layer during training, 

    #unpacking the parameters
    CompilerOut_layer = LayerObj.CompilerOut_layer  # the full DNN spec for the layer
    #print(CompilerOut_layer)
    bw_ifmap = LayerObj.bw_ifmap 
    bw_psum = LayerObj.bw_psum
    #bw_ofmap = LayerObj.bw_ofmap 

    SysArray_col = Hardware_param.SysArray_col
    inv_sqrt_cycles = Hardware_param.inv_sqrt_cycles

    SIMD_Ins_size = Hardware_param.SIMD_Ins_size
    RBw_DRAM_to_VMEM = Hardware_param.RBw_DRAM_to_VMEM
    WBw_VMEM_to_DRAM = Hardware_param.WBw_VMEM_to_DRAM

    #print(CompilerOut_layer['iterable_dimensions'])
    mini_batch_size = CompilerOut_layer['iterable_dimensions']['N']
    IC = CompilerOut_layer['iterable_dimensions']['C']
    Height = CompilerOut_layer['iterable_dimensions']['H']
    Width = CompilerOut_layer['iterable_dimensions']['W']

    effective_mini_batch = mini_batch_size * Height * Width
    #print("effective_mini_batch:", effective_mini_batch)

    #DRAM access count
    #DRAM_access_1 = (2 + 1) * IC * (2 * effective_mini_batch + 4)   # previous equation
    DRAM_access_1 = (2 + 1) * IC * (effective_mini_batch + 2) + (IC * effective_mini_batch * 8) + IC    # updated equation
    DRAM_access_2 = (IC + IC * effective_mini_batch * 2) * 4
    total_DRAM_access = (DRAM_access_1 + DRAM_access_2) * bw_ifmap
    #print("total_DRAM_access:", total_DRAM_access)

    #SRAM access count
    #VMEM_access_1 = (2 + 1) * IC * (2 * effective_mini_batch + 4)  # previous equation
    VMEM_access_1 = (2 + 1) * IC * (4 * effective_mini_batch + 2)   # updated equation
    VMEM_access_2 = ((2 + 1) * IC * effective_mini_batch) * 4
    total_VMEM_access = (VMEM_access_1 + VMEM_access_2) * bw_ifmap
    #print("total_VMEM_access:", total_VMEM_access)

    #Cycle count
    #comp_cycle_1 = (IC * (2 * effective_mini_batch + 3) + (IC * inv_sqrt_cycles)) / SysArray_col   #previous equation
    comp_cycle_1 = (IC * (4 * effective_mini_batch + 1) + (IC * inv_sqrt_cycles)) / SysArray_col    #updated equation  
    comp_cycle_2 = (IC * effective_mini_batch * 4) / SysArray_col   
    compute_cycles = comp_cycle_1 + comp_cycle_2        #omitting pipeline overhead
    #print("compute_cycles:", compute_cycles)

    DRAM_stall_cycles = total_DRAM_access/RBw_DRAM_to_VMEM
    #print("DRAM_stall_cycles:", DRAM_stall_cycles)

    # Nos of Op count: SysArray_col op per cycle (all ops are add, sub, or multiplication, there is also inverse sqrt, assuming op per inv sqrt = inv_sqrt_cycles)
    Nos_of_op = compute_cycles * SysArray_col
    SIMDResult_inflayer.arithmetic['op_ScmnBN'] = Nos_of_op

    # Access to instruction memory
    Nos_of_IF = compute_cycles  #number of instruction fetched (estimate, ideally for each inverse sqrt, one IF fetch)
    InsMem_access = Nos_of_IF * SIMD_Ins_size    # in bit

    SIMDResult_inflayer.SRAM_access['InsMem'] = InsMem_access

    SIMDResult_inflayer.DRAM_access['ifmap'] = total_DRAM_access   # putting all DARM access in ifmap DRAM access for simplification
    SIMDResult_inflayer.SRAM_access['VMEM'] = total_VMEM_access

    SIMDResult_inflayer.cycles['compute'] = compute_cycles
    SIMDResult_inflayer.cycles['DRAM_stall'] = DRAM_stall_cycles
    SIMDResult_inflayer.cycles['total'] = compute_cycles + DRAM_stall_cycles


def batch_norm_backward_estimate(Hardware_param, LayerObj, SIMDResult_inflayer):
    # This function analytically estimates the cost for Banch_norm_backward pass layer, 

    #unpacking the parameters
    CompilerOut_layer = LayerObj.CompilerOut_layer  # the full DNN spec for the layer
    #print(CompilerOut_layer)
    bw_ifmap = LayerObj.bw_ifmap 
    bw_psum = LayerObj.bw_psum
    #bw_ofmap = LayerObj.bw_ofmap 

    SysArray_col = Hardware_param.SysArray_col
    SIMD_Ins_size = Hardware_param.SIMD_Ins_size
    RBw_DRAM_to_VMEM = Hardware_param.RBw_DRAM_to_VMEM
    WBw_VMEM_to_DRAM = Hardware_param.WBw_VMEM_to_DRAM

    mini_batch_size = CompilerOut_layer['iterable_dimensions']['N']
    IC = CompilerOut_layer['iterable_dimensions']['C']
    Height = CompilerOut_layer['iterable_dimensions']['H']
    Width = CompilerOut_layer['iterable_dimensions']['W']

    effective_mini_batch = mini_batch_size * Height * Width
    #print("effective_mini_batch:", effective_mini_batch)

    #DRAM access count
    total_DRAM_access = (2 + 1) * IC * (10 * effective_mini_batch + 2) * bw_ifmap # using 32 bit for all access
    #print("total_DRAM_access:", total_DRAM_access)

    #SRAM access count
    total_VMEM_access = (2 + 1) * IC * (10 * effective_mini_batch + 2) * bw_ifmap
    #print("total_VMEM_access:", total_VMEM_access)

    #cycle counts
    compute_cycles = IC * (10 * effective_mini_batch + 2) / SysArray_col  # omitting pipeline overhead
    #print("compute_cycles:", compute_cycles)

    DRAM_stall_cycles = total_DRAM_access/RBw_DRAM_to_VMEM
    #print("DRAM_stall_cycles:", DRAM_stall_cycles)

    # Nos of Op count: SysArray_col op per cycle (all ops are either add, sub, or multiplication)
    Nos_of_op = IC * (10 * effective_mini_batch + 2)
    SIMDResult_inflayer.arithmetic['op_ScmnBN'] = Nos_of_op

    # Access to instruction memory
    Nos_of_IF = compute_cycles  #number of instruction fetched
    InsMem_access = Nos_of_IF * SIMD_Ins_size    # in bit

    SIMDResult_inflayer.SRAM_access['InsMem'] = InsMem_access

    SIMDResult_inflayer.DRAM_access['ifmap'] = total_DRAM_access   # putting all DARM access in ifmap DRAM access for simplification
    SIMDResult_inflayer.SRAM_access['VMEM'] = total_VMEM_access

    SIMDResult_inflayer.cycles['compute'] = compute_cycles
    SIMDResult_inflayer.cycles['DRAM_stall'] = DRAM_stall_cycles
    SIMDResult_inflayer.cycles['total'] = compute_cycles + DRAM_stall_cycles


def mean_istd_model(Hardware_param, LayerObj, SIMDResult_inflayer):
    #This function generates the cycle count and data access cost to compute the mean and inverse standrad deviation for the BatchNorm layer
    ####### For layers where IC = OC in the function code, used OC parameters and IC paramters are ignored

    #unpacking the parameters
    SysArray_col = Hardware_param.SysArray_col
    RBw_DRAM_to_VMEM = Hardware_param.RBw_DRAM_to_VMEM
    WBw_VMEM_to_DRAM = Hardware_param.WBw_VMEM_to_DRAM
    div_cycles = Hardware_param.div_cycles
    inv_sqrt_cycles = Hardware_param.inv_sqrt_cycles
    SIMD_Ins_size = Hardware_param.SIMD_Ins_size

    bw_ifmap = LayerObj.bw_ifmap 
    bw_psum = LayerObj.bw_psum
    bw_ofmap = LayerObj.bw_ofmap  

    IW = LayerObj.IW
    IH = LayerObj.IH
    OC = LayerObj.OC
    Batch = LayerObj.Batch

    DTile_iw = LayerObj.DTile_iw
    DTile_ih = LayerObj.DTile_ih
    DTile_oc = LayerObj.DTile_oc
    DTile_batch = LayerObj.DTile_batch

    Stile_iw = LayerObj.Stile_iw
    Stile_ih = LayerObj.Stile_ih
    Stile_oc = LayerObj.Stile_oc
    Stile_batch = LayerObj.Stile_batch

    Loop_order = LayerObj.Loop_order
    fusion_status = LayerObj.fusion_status

    assert Loop_order[-1] == 'oc'   # the model works for a loop order where channels are at the outer most loop
    assert fusion_status == "NoFusion"  #the model works for the no fusion case

    # Optimality Flags
    MUL_flag = False      #True = DIV is replaced by MUL by the compiler, else False
    Optimal_flag = True  # True = SDM's proposal of one less DIV/MUL, False = The current implementation by the compiler

    ############ DRAM Access
    ifmap_access_DRAM = (DTile_ih * DTile_iw * DTile_oc * DTile_batch) * (IH/DTile_ih) * (IW/DTile_iw) * (OC/DTile_oc) * (Batch/DTile_batch) * bw_ifmap
    mean_access_DRAM = DTile_oc * (OC/DTile_oc) * bw_ofmap
    istd_access_DRAM = DTile_oc * (OC/DTile_oc) * bw_ofmap
    ofmap_access_DRAM = mean_access_DRAM + istd_access_DRAM

    print("ifmap_access_DRAM:", ifmap_access_DRAM)
    print("ofmap_access_DRAM:", ofmap_access_DRAM)
    SIMDResult_inflayer.DRAM_access['ifmap'] = ifmap_access_DRAM
    SIMDResult_inflayer.DRAM_access['ofmap'] = ofmap_access_DRAM

    #Nos of operations inside the iw, ih, n loops for each type of ops
    Op_inner = (Stile_ih * Stile_iw * Stile_oc * Stile_batch) * (DTile_ih/Stile_ih) * (DTile_iw/Stile_iw) * (DTile_oc/Stile_oc) * (DTile_batch/Stile_batch) \
                                                            * (IH/DTile_ih) * (IW/DTile_iw) * (OC/DTile_oc) * (Batch/DTile_batch)
    #Nos of operations inside the oc loop for each type of ops
    Op_outer = Stile_oc * (DTile_oc/Stile_oc) * (OC/DTile_oc)

    ########## On-chip SRAM Access
    #for the on-chip access using bw_ifmap (32 bits) only since there is a lot of intermediate ops and omitting seperating bw_ofmap parts
    data_access_VMEM_inner = (2 + 1) * Op_inner * (1 + 1 + 1) * bw_ifmap # for ADD, MUL, ADD

    if Optimal_flag == True:
        nos_of_DIV_or_MUL = 2
    else:
        nos_of_DIV_or_MUL = 3

    data_access_VMEM_outer1 = (2 + 1) * Op_outer * (1 + 1 + 1 + 1) * bw_ifmap # for MUL, SUB, ADD, INVSQRT
    data_access_VMEM_outer2 = 2 * Op_outer * nos_of_DIV_or_MUL * bw_ifmap # for the DIV/MUL ops that use one operand from the IMM
    data_access_IMM = Op_outer * nos_of_DIV_or_MUL * bw_ifmap # for the DIV/MUL ops that use one operand from the IMM

    SIMDResult_inflayer.SRAM_access['VMEM'] = data_access_VMEM_inner + data_access_VMEM_outer1 + data_access_VMEM_outer2
    SIMDResult_inflayer.SRAM_access['IMM'] = data_access_IMM

    ############## Compute Cycles
    #of cycles for the operations in the iw, ih, n loops for one tile
    op1_add_cycle = 1
    op2_mul_cycle = 1
    op3_add_cycle = 1 + 2  # 2 extra NOP cycles for read after write dependency
    ops_inner_cycles = op1_add_cycle + op2_mul_cycle + op3_add_cycle

    cycle_oneTile_inner = ops_inner_cycles * ((Stile_ih * Stile_iw * Stile_oc * Stile_batch) / SysArray_col)  * \
                                    (DTile_ih/Stile_ih) * (DTile_iw/Stile_iw) * (DTile_oc/Stile_oc) * (DTile_batch/Stile_batch)  #for ADD, MUL, ADD

    #of cycles to fill the pipeline for a tile, there are 6 pipeline stages
    pipe_overhead_tile = (6 - 1) + (SysArray_col - 1)  

    #of cycles to compute one inner tile including the pipeline setup operhead
    ComputeTile_cycles_inner = cycle_oneTile_inner + pipe_overhead_tile

    #of cycles for the operations in the oc loop for one oc tile
    if MUL_flag == True:
        DIV_or_MUL_cycle = 1
    else:
        DIV_or_MUL_cycle = div_cycles

    if Optimal_flag == True:   # +2 is for the additional NOP cycles due to read-after-write dependency
        op4_divmul_cycle = DIV_or_MUL_cycle
        op5_mul_cycle = 1 + 2
        op6_divmul_cycle = DIV_or_MUL_cycle
        op7_sub_cycle = 1 + 2
        op8_add_cycle = 1 + 2
        op9_invsqrt_cycle = inv_sqrt_cycles + 2
        ops_outer_cycles = op4_divmul_cycle + op5_mul_cycle + op6_divmul_cycle + op7_sub_cycle + op8_add_cycle + op9_invsqrt_cycle
    else:
        op4_mul_cycle = 1
        op5_divmul_cycle = DIV_or_MUL_cycle + 2
        op6_sub_cycle = 1 + 2
        op7_divmul_cycle = DIV_or_MUL_cycle + 2
        op8_add_cycle = 1 + 2
        op9_invsqrt_cycle = inv_sqrt_cycles + 2
        op10_divmul_cycle = DIV_or_MUL_cycle
        ops_outer_cycles = op4_mul_cycle + op5_divmul_cycle + op6_sub_cycle + op7_divmul_cycle + op8_add_cycle + op9_invsqrt_cycle + op10_divmul_cycle

    #of cycles to compute one outer tile: the outer tile does not have pipeline setup overhead since it comes just after an inner compute tile
    ComputeTile_cycles_outer = ops_outer_cycles * (Stile_oc / SysArray_col) * (DTile_oc/Stile_oc) 

    #Total compute cycles
    Nos_of_inner_tile = (IH/DTile_ih) * (IW/DTile_iw) * (Batch/DTile_batch)
    Nos_of_outer_tile = (OC/DTile_oc)
    compute_cycles = math.ceil(((ComputeTile_cycles_inner * Nos_of_inner_tile) + ComputeTile_cycles_outer) * Nos_of_outer_tile) # giving the outer ceil to avoid fraction

    ######## Model for the DRAM stall cycles
    if (Hardware_param.Buffering_scheme_VMEM == "single"):
        #of cycles required to load/store each tile of each kind of data, both VMEM shares the same AXI 
        ifmapTile_load_cycles = math.ceil((DTile_iw * DTile_ih * DTile_oc * DTile_batch * bw_ifmap) / RBw_DRAM_to_VMEM)
        meanTile_store_cycles = math.ceil((DTile_oc * bw_ofmap) / WBw_VMEM_to_DRAM)
        istdTile_store_cycles = math.ceil((DTile_oc * bw_ofmap) / WBw_VMEM_to_DRAM)

        # VMEM is single buffered, so there is no overlap with computation.
        ifmap_stall_cycles = ifmapTile_load_cycles * Nos_of_inner_tile * Nos_of_outer_tile
        MeanIstd_stall_cycles = (meanTile_store_cycles + istdTile_store_cycles) * Nos_of_outer_tile
        DRAM_stall_cycles = ifmap_stall_cycles + MeanIstd_stall_cycles
    else:
        print("Model do not exist for VMEM non single buffer scheme now")

    print("compute_cycles:", compute_cycles)
    print("DRAM_stall_cycles:", DRAM_stall_cycles)
    SIMDResult_inflayer.cycles['compute'] = compute_cycles
    SIMDResult_inflayer.cycles['DRAM_stall'] = DRAM_stall_cycles
    SIMDResult_inflayer.cycles['total'] = compute_cycles + DRAM_stall_cycles

    ###### Number of arithmetic Op counts:
    #ADD, SUB, INVSQRT operations
    Nos_of_add = 2 * Op_inner + 1 * Op_outer
    Nos_of_sub = 1 * Op_outer
    Nos_of_invsqrt = 1 * Op_outer

    # MUL & DIV operations
    mul_inner = 1 * Op_inner
    mul_outer_cmn = 1 * Op_outer
    if MUL_flag == True:
        mul_outer_diff = nos_of_DIV_or_MUL * Op_outer
        div_outer_diff = 0
    else:
        mul_outer_diff = 0
        div_outer_diff = nos_of_DIV_or_MUL * Op_outer

    Nos_of_mul = mul_inner + mul_outer_cmn + mul_outer_diff
    Nos_of_div = div_outer_diff

    SIMDResult_inflayer.arithmetic['add'] = Nos_of_add
    SIMDResult_inflayer.arithmetic['sub'] = Nos_of_sub
    SIMDResult_inflayer.arithmetic['mul'] = Nos_of_mul
    SIMDResult_inflayer.arithmetic['div'] = Nos_of_div
    SIMDResult_inflayer.arithmetic['inv_sqrt'] = Nos_of_invsqrt

    ####### Other Access that happen at every compute cycle: InsMem & PipeReg
    # Access for Instruction Memory
    Nos_of_IF = (Nos_of_add + Nos_of_sub + Nos_of_mul + Nos_of_div + Nos_of_invsqrt)/SysArray_col
    InsMem_access = Nos_of_IF * SIMD_Ins_size
    SIMDResult_inflayer.SRAM_access['InsMem'] = InsMem_access


def batch_norm_forward_model(Hardware_param, LayerObj, SIMDResult_inflayer):
    #This function generates the cycle count and data access cost for the BatchNorm layer given mean and inverse standard deviation is computed
    ####### For layers where IC = OC in the function code, used OC parameters and IC paramters are ignored

    #unpacking the parameters
    SysArray_col = Hardware_param.SysArray_col
    RBw_DRAM_to_VMEM = Hardware_param.RBw_DRAM_to_VMEM
    WBw_VMEM_to_DRAM = Hardware_param.WBw_VMEM_to_DRAM
    SIMD_Ins_size = Hardware_param.SIMD_Ins_size

    bw_ifmap = LayerObj.bw_ifmap 
    bw_psum = LayerObj.bw_psum
    bw_ofmap = LayerObj.bw_ofmap  

    #for BatchNorm, the tiling and dimension of 4D input data and the 4D output data are same
    OW = LayerObj.IW
    OH = LayerObj.IH
    OC = LayerObj.OC
    Batch = LayerObj.Batch

    DTile_ow = LayerObj.DTile_iw
    DTile_oh = LayerObj.DTile_ih
    DTile_oc = LayerObj.DTile_oc
    DTile_batch = LayerObj.DTile_batch

    Stile_ow = LayerObj.Stile_iw
    Stile_oh = LayerObj.Stile_ih
    Stile_oc = LayerObj.Stile_oc
    Stile_batch = LayerObj.Stile_batch

    Loop_order = LayerObj.Loop_order
    fusion_status = LayerObj.fusion_status

    assert Loop_order[-1] == 'oc'   # the model works for a loop order where channels are at the outer most loop
    assert fusion_status == "NoFusion"  #the model works for the no fusion case


    ############ DRAM Access
    ifmap_data_access_DRAM = (DTile_oh * DTile_ow * DTile_oc * DTile_batch) * (OH/DTile_oh) * (OW/DTile_ow) * (OC/DTile_oc) * (Batch/DTile_batch) * bw_ifmap
    mean_access_DRAM = DTile_oc * (OC/DTile_oc) * bw_ifmap
    istd_access_DRAM = DTile_oc * (OC/DTile_oc) * bw_ifmap
    scale_access_DRAM = DTile_oc * (OC/DTile_oc) * bw_ifmap
    offset_access_DRAM = DTile_oc * (OC/DTile_oc) * bw_ifmap
    ifmap_access_DRAM = ifmap_data_access_DRAM + mean_access_DRAM + istd_access_DRAM + scale_access_DRAM + offset_access_DRAM

    ofmap_access_DRAM = (DTile_oh * DTile_ow * DTile_oc * DTile_batch) * (OH/DTile_oh) * (OW/DTile_ow) * (OC/DTile_oc) * (Batch/DTile_batch) * bw_ofmap

    print("ifmap_access_DRAM:", ifmap_access_DRAM)
    print("ofmap_access_DRAM:", ofmap_access_DRAM)
    SIMDResult_inflayer.DRAM_access['ifmap'] = ifmap_access_DRAM
    SIMDResult_inflayer.DRAM_access['ofmap'] = ofmap_access_DRAM

    #### Nos of Ops
    Nos_of_op_each = (Stile_oh * Stile_ow * Stile_oc * Stile_batch) * (DTile_oh/Stile_oh) * (DTile_ow/Stile_ow) * (DTile_oc/Stile_oc) * (DTile_batch/Stile_batch) \
                                                            * (OH/DTile_oh) * (OW/DTile_ow) * (OC/DTile_oc) * (Batch/DTile_batch)
    Nos_of_sub = Nos_of_op_each
    Nos_of_mul1 = Nos_of_op_each
    Nos_of_mul2 = Nos_of_op_each
    Nos_of_add = Nos_of_op_each

    ########## On-chip SRAM Access
    #Current Quantization Assumption: for now assuming quantization is free and happen inside the ALU before writing ofmap back to VMEM
    data_access_VMEM_sub = (2 + 1) * Nos_of_sub * bw_ifmap
    data_access_VMEM_mul1 = (2 + 1) * Nos_of_mul1 * bw_ifmap
    data_access_VMEM_mul2 = (2 + 1) * Nos_of_mul2 * bw_ifmap
    data_access_VMEM_add = Nos_of_add * (2 * bw_ifmap + bw_ofmap) # this is the last operation that produce the output

    SIMDResult_inflayer.SRAM_access['VMEM'] = data_access_VMEM_sub + data_access_VMEM_mul1 + data_access_VMEM_mul2 + data_access_VMEM_add

    ############## Compute Cycles
    #of cycles for the operations
    op1_sub_cycle = 1
    op2_mul_cycle = 1 + 2 #2 extra NOP cycles for read after write dependency
    op3_mul_cycle = 1 + 2  
    op4_add_cycle = 1 + 2
    all_ops_cycles = op1_sub_cycle + op2_mul_cycle + op3_mul_cycle + op4_add_cycle

    cycle_oneTile = all_ops_cycles * ((Stile_oh * Stile_ow * Stile_oc * Stile_batch) / SysArray_col)  * \
                                    (DTile_oh/Stile_oh) * (DTile_ow/Stile_ow) * (DTile_oc/Stile_oc) * (DTile_batch/Stile_batch)  #for SUB, MUL, MUL, ADD

    #of cycles to fill the pipeline for a tile, there are 6 pipeline stages
    pipe_overhead_tile = (6 - 1) + (SysArray_col - 1)  

    #of cycles to compute one tile including the pipeline setup operhead
    ComputeTile_cycles = cycle_oneTile + pipe_overhead_tile

    #Total compute cycles
    Number_of_Tile = (OH/DTile_oh) * (OW/DTile_ow) * (OC/DTile_oc) * (Batch/DTile_batch)
    compute_cycles = math.ceil(ComputeTile_cycles * Number_of_Tile)   # giving the outer ceil to avoid fraction cycle numbers

    ######## Model for the DRAM stall cycles
    if (Hardware_param.Buffering_scheme_VMEM == "single"):
        #of cycles required to load/store each tile of each kind of data, both VMEM shares the same AXI 
        ifmapTile_load_cycles = math.ceil((DTile_ow * DTile_oh * DTile_oc * DTile_batch * bw_ifmap) / RBw_DRAM_to_VMEM)
        meanTile_load_cycles = math.ceil((DTile_oc * bw_ifmap) / RBw_DRAM_to_VMEM)
        istdTile_load_cycles = math.ceil((DTile_oc * bw_ifmap) / RBw_DRAM_to_VMEM)
        scaleTile_load_cycles = math.ceil((DTile_oc * bw_ifmap) / RBw_DRAM_to_VMEM)
        offsetTile_load_cycles = math.ceil((DTile_oc * bw_ifmap) / RBw_DRAM_to_VMEM)
        ofmapTile_store_cycles = math.ceil((DTile_ow * DTile_oh * DTile_oc * DTile_batch * bw_ofmap) / WBw_VMEM_to_DRAM)

        # VMEM is single buffered, so there is no overlap with computation.
        ifmap_stall_cycles = ifmapTile_load_cycles * Number_of_Tile
        MISO_stall_cycles = (meanTile_load_cycles + istdTile_load_cycles + scaleTile_load_cycles + offsetTile_load_cycles) * (OC/DTile_oc)
        ofmap_stall_cycles = ofmapTile_store_cycles * Number_of_Tile
        DRAM_stall_cycles = ifmap_stall_cycles + MISO_stall_cycles + ofmap_stall_cycles
    else:
        print("Model do not exist for VMEM non single buffer scheme now")

    print("compute_cycles:", compute_cycles)
    print("DRAM_stall_cycles:", DRAM_stall_cycles)
    SIMDResult_inflayer.cycles['compute'] = compute_cycles
    SIMDResult_inflayer.cycles['DRAM_stall'] = DRAM_stall_cycles
    SIMDResult_inflayer.cycles['total'] = compute_cycles + DRAM_stall_cycles

    ###### Number of arithmetic Op counts
    SIMDResult_inflayer.arithmetic['sub'] = Nos_of_sub
    SIMDResult_inflayer.arithmetic['mul'] = Nos_of_mul1 + Nos_of_mul2
    SIMDResult_inflayer.arithmetic['add'] = Nos_of_add

    ####### Other Access that happen at every compute cycle: InsMem & PipeReg
    # Access for Instruction Memory
    Nos_of_IF = (Nos_of_sub + Nos_of_mul1 + Nos_of_mul2 + Nos_of_add)/SysArray_col
    InsMem_access = Nos_of_IF * SIMD_Ins_size
    SIMDResult_inflayer.SRAM_access['InsMem'] = InsMem_access



def batch_norm_backward_model(Hardware_param, LayerObj, SIMDResult_inflayer):
    #This function generates the cycle count and data access cost for the BatchNorm_BAckward layer given mean and inverse standard deviation is computed
    ####### For layers where IC = OC in the function code, used OC parameters and IC paramters are ignored

    #unpacking the parameters
    SysArray_col = Hardware_param.SysArray_col
    RBw_DRAM_to_VMEM = Hardware_param.RBw_DRAM_to_VMEM
    WBw_VMEM_to_DRAM = Hardware_param.WBw_VMEM_to_DRAM
    div_cycles = Hardware_param.div_cycles
    SIMD_Ins_size = Hardware_param.SIMD_Ins_size

    bw_ifmap = LayerObj.bw_ifmap 
    bw_psum = LayerObj.bw_psum
    bw_ofmap = LayerObj.bw_ofmap  

    #for BatchNorm_Backward, the tiling and dimension of 4D input data and the 4D output data_grad are same
    OW = LayerObj.IW
    OH = LayerObj.IH
    OC = LayerObj.OC
    Batch = LayerObj.Batch

    DTile_ow = LayerObj.DTile_iw
    DTile_oh = LayerObj.DTile_ih
    DTile_oc = LayerObj.DTile_oc
    DTile_batch = LayerObj.DTile_batch

    Stile_ow = LayerObj.Stile_iw
    Stile_oh = LayerObj.Stile_ih
    Stile_oc = LayerObj.Stile_oc
    Stile_batch = LayerObj.Stile_batch

    Loop_order = LayerObj.Loop_order
    fusion_status = LayerObj.fusion_status

    assert Loop_order[-1] == 'oc'   # the model works for a loop order where channels are at the outer most loop
    assert fusion_status == "NoFusion"  #the model works for the no fusion case

    MUL_flag = False # flag to determine whether the DIV is replaced by MUL or not. True = DIV is replaced by MUL

    ######## DRAM Access
    #Part-1: scale-grad and offset-grad computation; Part-2: data-grad computation
    access4D_DRAM = (DTile_oh * DTile_ow * DTile_oc * DTile_batch) * (OH/DTile_oh) * (OW/DTile_ow) * (OC/DTile_oc) * (Batch/DTile_batch)
    access1D_DRAM = DTile_oc * (OC/DTile_oc)

    data_access_DRAM = access4D_DRAM * bw_ifmap
    prev_grad_access_DRAM = access4D_DRAM * bw_ifmap * 2 # once during part-1 and once during part-2
    istd_access_DRAM = access1D_DRAM * bw_ifmap
    mean_access_DRAM = access1D_DRAM * bw_ifmap
    scale_access_DRAM = access1D_DRAM * bw_ifmap

    data_norm_access_DRAM = access4D_DRAM * bw_ifmap * 2  # write access at the end of part-1 and read access at the beginning of part-2s
    
    scale_grad_access_DRAM = access1D_DRAM * bw_ofmap
    offset_grad_access_DRAM = access1D_DRAM * bw_ofmap
    data_grad_access_DRAM = access4D_DRAM * bw_ofmap

    ifmap_access_DRAM = data_access_DRAM + prev_grad_access_DRAM + istd_access_DRAM + mean_access_DRAM + scale_access_DRAM + data_norm_access_DRAM
    ofmap_access_DRAM = scale_grad_access_DRAM + offset_grad_access_DRAM + data_grad_access_DRAM

    SIMDResult_inflayer.DRAM_access['ifmap'] = ifmap_access_DRAM
    SIMDResult_inflayer.DRAM_access['ofmap'] = ofmap_access_DRAM

    ###### Op Count
    Op_inner = (Stile_oh * Stile_ow * Stile_oc * Stile_batch) * (DTile_oh/Stile_oh) * (DTile_ow/Stile_ow) * (DTile_oc/Stile_oc) * (DTile_batch/Stile_batch) \
                                                            * (OH/DTile_oh) * (OW/DTile_ow) * (OC/DTile_oc) * (Batch/DTile_batch)
    Op_outer = Stile_oc * (DTile_oc/Stile_oc) * (OC/DTile_oc) 
    
    #for Part-1
    sub_p1 = Op_inner
    mul1_p1 = Op_inner
    mul2_p1 = Op_inner
    add1_p1 = Op_inner
    add2_p1 = Op_inner

    #for Part-2
    #outer c loop
    mul1_p2_outer = Op_outer
    divmul_p2_outer = Op_outer

    mul2_p2_inner = Op_inner
    mul3_p2_inner = Op_inner
    sub1_p2_inner = Op_inner
    sub2_p2_inner = Op_inner
    mul4_p2_inner = Op_inner

    ########## On-Chip SRAM Access
    #for Part-1: using bw_ifmap for all since the output from part-1 (data-grad and offset-grad) are used in Part-2. Hence, they are before quantization (i.e, 32 bit)
    data_access_VMEM_p1 = (2 + 1) * (sub_p1 + mul1_p1 + mul2_p1 + add1_p1 + add2_p1) * bw_ifmap

    #for Part-2:
    data_access_VMEM_p2_1 = (2 + 1) * (mul1_p2_outer + divmul_p2_outer + mul2_p2_inner + mul3_p2_inner + sub1_p2_inner + sub2_p2_inner) * bw_ifmap
    data_access_VMEM_p2_2 = mul4_p2_inner * (2 * bw_ifmap + bw_ofmap)  # this is the last operation that produce the output

    SIMDResult_inflayer.SRAM_access['VMEM'] = data_access_VMEM_p1 + data_access_VMEM_p2_1 + data_access_VMEM_p2_2


    ######### Compute Cycles
    #of cycles for the operations: Part-1
    op1_p1_sub_cycle = 1
    op2_p1_mul_cycle = 1 + 2 #2 extra NOP cycles for read after write dependency
    op3_p1_mul_cycles = 1 + 2 
    op4_p1_add_cycle = 1 + 2
    op5_p1_add_cycle = 1

    #of cycles for the operations: Part-2
    op1_p2_mul_cycle = 1
    if MUL_flag == True:
        op2_p2_divmul_cycle = 1 + 2
    else:
        op2_p2_divmul_cycle = div_cycles + 2
    op3_p2_mul_cycle = 1 
    op4_p2_mul_cycle = 1 
    op5_p2_sub_cycle = 1 + 2
    op6_p2_sub_cycle = 1 + 2
    op7_p2_mul_cycle = 1 + 2

    #of cycles to fill the pipeline for a tile, there are 6 pipeline stages
    pipe_overhead_tile = (6 - 1) + (SysArray_col - 1)  

    # Formulation for Part-1 over a single tile
    all_ops_cycles_p1 = op1_p1_sub_cycle + op2_p1_mul_cycle + op3_p1_mul_cycles + op4_p1_add_cycle + op5_p1_add_cycle
    cycle_oneTile_p1 = all_ops_cycles_p1 * ((Stile_oh * Stile_ow * Stile_oc * Stile_batch) / SysArray_col)  * \
                                    (DTile_oh/Stile_oh) * (DTile_ow/Stile_ow) * (DTile_oc/Stile_oc) * (DTile_batch/Stile_batch)  
    
    ComputeTile_cycles_p1 = cycle_oneTile_p1 + pipe_overhead_tile  #of cycles to compute one tile including the pipeline setup operhead for part-1

    # Formulation for Part-2 over a single tile
    outer_ops_cycles_p2 = op1_p2_mul_cycle + op2_p2_divmul_cycle
    inner_ops_cycles_p2 = op3_p2_mul_cycle + op4_p2_mul_cycle + op5_p2_sub_cycle + op6_p2_sub_cycle + op7_p2_mul_cycle

    cycle_oneTile_p2_outer = outer_ops_cycles_p2 * (Stile_oc/SysArray_col) * (DTile_oc/Stile_oc)
    cycle_oneTile_p2_inner = inner_ops_cycles_p2 * ((Stile_oh * Stile_ow * Stile_oc * Stile_batch) / SysArray_col)  * \
                                    (DTile_oh/Stile_oh) * (DTile_ow/Stile_ow) * (DTile_oc/Stile_oc) * (DTile_batch/Stile_batch)
    
    ComputeTile_cycles_p2_outer = cycle_oneTile_p2_outer # setup overhead is not needed for the outer part. It will be counted since it is given in the outer part
    ComputeTile_cycles_p2_inner = cycle_oneTile_p2_inner + pipe_overhead_tile 

    # Total Compute Cycles
    Ntile_inner = (OH/DTile_oh) * (OW/DTile_ow) * (Batch/DTile_batch)  # Number of inner and outer tiles
    Ntile_outer = OC/DTile_oc

    compute_cycles = math.ceil(((ComputeTile_cycles_p1 * Ntile_inner) + (ComputeTile_cycles_p2_inner * Ntile_inner) + ComputeTile_cycles_p2_outer) * Ntile_outer)
     
    ######## Model for the DRAM stall cycles 
    if (Hardware_param.Buffering_scheme_VMEM == "single"):
        #of cycles required to load/store each tile of each kind of data, both VMEM shares the same AXI 
        tile_4D = DTile_oh * DTile_ow * DTile_oc * DTile_batch
        tile_1D = DTile_oc

        # Tiled cycles for both Part-1 & Part-2
        dataTile_load_cycles = math.ceil((tile_4D * bw_ifmap) / RBw_DRAM_to_VMEM)
        prev_gradTile_load_cycles = math.ceil((tile_4D * bw_ifmap) / RBw_DRAM_to_VMEM)
        istdTile_load_cycles = math.ceil((tile_1D * bw_ifmap) / RBw_DRAM_to_VMEM)
        meanTile_load_cycles = math.ceil((tile_1D * bw_ifmap) / RBw_DRAM_to_VMEM)
        scaleTile_load_cycles = math.ceil((tile_1D * bw_ifmap) / RBw_DRAM_to_VMEM)
        data_normTile_load_cycles = math.ceil((tile_4D * bw_ifmap) / RBw_DRAM_to_VMEM)

        data_normTile_store_cycles = math.ceil((tile_4D * bw_ifmap) / WBw_VMEM_to_DRAM)
        scale_gradTile_store_cycles = math.ceil((tile_1D * bw_ofmap) / WBw_VMEM_to_DRAM)
        offset_gradTile_store_cycles = math.ceil((tile_1D * bw_ofmap) / WBw_VMEM_to_DRAM)
        data_gradTile_store_cycles = math.ceil((tile_4D * bw_ofmap) / WBw_VMEM_to_DRAM)

        # VMEM is single buffered, so there is no overlap with computation
        tensor4D_stall_cycles = (dataTile_load_cycles + \
                                 prev_gradTile_load_cycles * 2 + \
                                 data_normTile_load_cycles + data_normTile_store_cycles + \
                                 data_gradTile_store_cycles) * (Ntile_inner * Ntile_outer)
        
        tensor1D_stal_cycles = (istdTile_load_cycles + meanTile_load_cycles + scaleTile_load_cycles + \
                                scale_gradTile_store_cycles + offset_gradTile_store_cycles) * Ntile_outer

        DRAM_stall_cycles = tensor4D_stall_cycles + tensor1D_stal_cycles

    else:
        print("Model do not exist for VMEM non single buffer scheme now")

    print("compute_cycles:", compute_cycles)
    print("DRAM_stall_cycles:", DRAM_stall_cycles)
    SIMDResult_inflayer.cycles['compute'] = compute_cycles
    SIMDResult_inflayer.cycles['DRAM_stall'] = DRAM_stall_cycles
    SIMDResult_inflayer.cycles['total'] = compute_cycles + DRAM_stall_cycles

    ###### Number of arithmetic Op counts
    if MUL_flag == True:
        mul5_p2 = divmul_p2_outer
        div_p2 = 0
    else:
        mul5_p2 = 0
        div_p2 = divmul_p2_outer

    SIMDResult_inflayer.arithmetic['sub'] = sub_p1 + sub1_p2_inner + sub2_p2_inner
    SIMDResult_inflayer.arithmetic['add'] = add1_p1 + add2_p1 
    SIMDResult_inflayer.arithmetic['mul'] = mul1_p1 + mul2_p1 + mul1_p2_outer + mul2_p2_inner + mul3_p2_inner + mul4_p2_inner + mul5_p2
    SIMDResult_inflayer.arithmetic['div'] = div_p2

    ####### Other Access that happen at every compute cycle: InsMem & PipeReg
    # Access for Instruction Memory
    Nos_of_IF = (sub_p1 + sub1_p2_inner + sub2_p2_inner + add1_p1 + add2_p1 + \
                                            mul1_p1 + mul2_p1 + mul1_p2_outer + mul2_p2_inner + mul3_p2_inner + mul4_p2_inner + mul5_p2 + div_p2)/SysArray_col

    InsMem_access = Nos_of_IF * SIMD_Ins_size
    SIMDResult_inflayer.SRAM_access['InsMem'] = InsMem_access


def pooling_backward_model(Hardware_param, LayerObj, SIMDResult_inflayer):
    #This function generates the cycle count and data access cost for the gradient computation of Max_pool, avg_pool, and global_avg_pool layer
    # Keyword note: OH, OW are used for the grad which is basically input to the layer; IH, IW are used for the data_out which is the output of the layer
    # bw_ifmap--> bitwidth of grad which is input to the layer; bw_ofmap--> bitwidth of data_out which is the output of the layer
    # IC = OC, so used OC as the parameter

    #unpacking the parameters
    SysArray_col = Hardware_param.SysArray_col
    RBw_DRAM_to_VMEM = Hardware_param.RBw_DRAM_to_VMEM
    WBw_VMEM_to_DRAM = Hardware_param.WBw_VMEM_to_DRAM
    SIMD_Ins_size = Hardware_param.SIMD_Ins_size

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
    #DTile_kw = LayerObj.DTile_kw
    #DTile_kh = LayerObj.DTile_kh
    DTile_iw = LayerObj.DTile_iw
    DTile_ih = LayerObj.DTile_ih 
    DTile_batch = LayerObj.DTile_batch

    Stile_ow = LayerObj.Stile_ow
    Stile_oh = LayerObj.Stile_oh
    Stile_oc = LayerObj.Stile_oc
    Stile_kw = LayerObj.Stile_kw
    Stile_kh = LayerObj.Stile_kh
    #Stile_iw = LayerObj.Stile_iw 
    #Stile_ih = LayerObj.Stile_ih 
    Stile_batch = LayerObj.Stile_batch

    Loop_order = LayerObj.Loop_order
    fusion_status = LayerObj.fusion_status

    assert fusion_status == "NoFusion"  #the model works for the no fusion case
    #Invalid tiling: in pooling KH and KW diemnsion should not be tiled for DRAM tiles. completely unnecessary and inefficient 
    #assert (KW/DTile_kw) == 1
    #assert (KH/DTile_kh) == 1

    #### DRAM Access Model
    grad_access_DRAM = (DTile_oh * DTile_ow * DTile_oc * DTile_batch) * (OH/DTile_oh) * (OW/DTile_ow) * (OC/DTile_oc) * (Batch/DTile_batch) * bw_ifmap
    index_access_DRAM = grad_access_DRAM
    data_out_access_DRAM = (DTile_ih * DTile_iw * DTile_oc * DTile_batch) * (OH/DTile_oh) * (OW/DTile_ow) * (OC/DTile_oc) * (Batch/DTile_batch) * bw_ofmap

    if Layer_name == "MaxPool_Grad":
        ifmap_access_DRAM = grad_access_DRAM + index_access_DRAM
    elif Layer_name == "AvgPool_Grad":
        ifmap_access_DRAM = grad_access_DRAM   # there is no index input for the average pool layer

    #print("ifmap_access_DRAM:", ifmap_access_DRAM)
    #print("ofmap_access_DRAM:", data_out_access_DRAM)
    SIMDResult_inflayer.DRAM_access['ifmap'] = ifmap_access_DRAM
    SIMDResult_inflayer.DRAM_access['ofmap'] = data_out_access_DRAM

    #### OP Counts 
    inner_multiplier = (Stile_oh * Stile_ow * Stile_oc * Stile_batch) * (DTile_oh/Stile_oh) * (DTile_ow/Stile_ow) * (DTile_oc/Stile_oc) * (DTile_batch/Stile_batch)
    outer_multiplier = (OH/DTile_oh) * (OW/DTile_ow) * (OC/DTile_oc) * (Batch/DTile_batch)
    kernel_multiplier = KH * KW

    Nos_of_CondMv_op = inner_multiplier * kernel_multiplier * outer_multiplier  # this is for max_pool only
    Nos_of_DIV_op = inner_multiplier * outer_multiplier                         # This is for average_pool only
    Nos_of_ADD_op = inner_multiplier * kernel_multiplier * outer_multiplier  # this is for both max_pool and average_pool

    if Layer_name == "MaxPool_Grad":
        total_ops = Nos_of_CondMv_op + Nos_of_ADD_op
    elif Layer_name == "AvgPool_Grad":
        total_ops = Nos_of_DIV_op + Nos_of_ADD_op

    #### On-Chip SRAM Access
    # Not using bw_ofmap at all here. For the pool_grad it will be complicated for the hardware to seperate the #of last add ops that produces the final output. 
    # Hence, omitting using bw_ofmap for the (IH * IW * OC * Bathc) portion of output from the ADD ops. Instead, uniformly using bw_ifmap for all VMEM access.
    # Also ideally for DIV op, one operand probably will come from IMM instead of VMEM. avoiding seperating that as well since there is only two pooling_grad layer
    # during ResNet-50 training and the impact of pool_grad will be very small
    data_access_VMEM = (2 + 1) * total_ops * bw_ifmap 

    #print("data_access_VMEM:", data_access_VMEM)
    SIMDResult_inflayer.SRAM_access['VMEM'] = data_access_VMEM

    #### Compute Cycles
    op1_maxpool_CondMv_cycle = Hardware_param.cond_move_cycles
    op2_maxpool_ADD_cycle = 1 + 2  #2 extra NOP cycles for read after write dependency
    op1_avgpool_DIV_cycle = Hardware_param.div_cycles
    op2_avgpool_ADD_cycle1 = 1  # there will be some ADD ops for which there is no NOP
    op2_avgpool_ADD_cycle2 = 1 + 2  #there will be some ADD ops for which there is 2 NOP

    #of cycles to fill the pipeline for a tile, there are 6 pipeline stages
    pipe_overhead_tile = (6 - 1) + (SysArray_col - 1) 


    if Layer_name == "MaxPool_Grad":
        all_ops_cycles = op1_maxpool_CondMv_cycle + op2_maxpool_ADD_cycle
        cycle_oneTile = all_ops_cycles * (inner_multiplier * kernel_multiplier / SysArray_col)

    elif Layer_name == "AvgPool_Grad":
        cycle_oneTile_DIV = op1_avgpool_DIV_cycle * (inner_multiplier /SysArray_col)
        ADD_Stall = inner_multiplier  #of ADD ops for which there is 2 NOPs since there is an associated DIV ops
        ADD_NoStall = ADD_Stall * (kernel_multiplier - 1)  #of ADD ops for which there is no NOP
        cycle_oneTile_ADD = (ADD_NoStall * op2_avgpool_ADD_cycle1 + ADD_Stall * op2_avgpool_ADD_cycle2) / SysArray_col
        cycle_oneTile = cycle_oneTile_DIV + cycle_oneTile_ADD
        
    #of cycles to compute one tile including the pipeline setup operhead
    ComputeTile_cycles = cycle_oneTile + pipe_overhead_tile

    #Total compute cycles
    Number_of_Tile = outer_multiplier
    compute_cycles = math.ceil(ComputeTile_cycles * Number_of_Tile)   # giving the outer ceil to avoid fraction cycle numbers

    ######## Model for the DRAM stall cycles
    if (Hardware_param.Buffering_scheme_VMEM == "single"):
        #of cycles required to load/store each tile of each kind of data, both VMEM shares the same AXI 
        gradTile_load_cycles =  math.ceil((DTile_oh * DTile_ow * DTile_oc * DTile_batch * bw_ifmap) / RBw_DRAM_to_VMEM)
        indexTile_load_cycles = gradTile_load_cycles
        data_outTile_store_cycles = math.ceil((DTile_ih * DTile_iw * DTile_oc * DTile_batch * bw_ofmap) / WBw_VMEM_to_DRAM)

        # VMEM is single buffered, so there is no overlap with computation.
        if Layer_name == "MaxPool_Grad":
            ifmap_stall_cycles = (gradTile_load_cycles + indexTile_load_cycles) * Number_of_Tile
        elif Layer_name == "AvgPool_Grad":
            ifmap_stall_cycles = gradTile_load_cycles * Number_of_Tile
        
        ofmap_stall_cycles = data_outTile_store_cycles * Number_of_Tile
        DRAM_stall_cycles = ifmap_stall_cycles + ofmap_stall_cycles

    else:
        print("Model do not exist for VMEM non single buffer scheme now")

    #print("compute_cycles:", compute_cycles)
    #print("DRAM_stall_cycles:", DRAM_stall_cycles)
    SIMDResult_inflayer.cycles['compute'] = compute_cycles
    SIMDResult_inflayer.cycles['DRAM_stall'] = DRAM_stall_cycles
    SIMDResult_inflayer.cycles['total'] = compute_cycles + DRAM_stall_cycles

    ###### Number of arithmetic Op counts
    if Layer_name == "MaxPool_Grad":
        SIMDResult_inflayer.arithmetic['CondMove'] = Nos_of_CondMv_op
    elif Layer_name == "AvgPool_Grad":
        SIMDResult_inflayer.arithmetic['div'] = Nos_of_DIV_op
    SIMDResult_inflayer.arithmetic['add'] = Nos_of_ADD_op

    ####### Other Access that happen at every compute cycle: InsMem & PipeReg
    # Access for Instruction Memory
    Nos_of_IF = total_ops/SysArray_col
    InsMem_access = Nos_of_IF * SIMD_Ins_size
    SIMDResult_inflayer.SRAM_access['InsMem'] = InsMem_access


                             
