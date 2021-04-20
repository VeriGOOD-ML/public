#This file contains the additional functions to support training

import logging
import math
import numpy as np
from Data_Objects import HardwareObject, SAResult_Inflayer, SIMDResult_Inflayer
from Layer_Object import LayerObject


def common_SIMD_backward_model(Hardware_param, LayerObj, SIMDResult_inflayer):
    # This is a common function for several SIMD operation during the backward pass

    #unpacking the parameters
    CompilerOut_layer = LayerObj.CompilerOut_layer  # the full compiler output for the layer
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


    ################## Computing VMEM access: omitting fusion status condition now, default: no fusion (will add condition for fusion later)
    # dictionary to hold the number of computation step within each tile of a layer to produce an output, add more layer as you move forward
    comp_step_rsum = "None"
    if layer_op == "reduce_sum":
        comp_step_rsum = CompilerOut_layer['inputs'][0]['shape_symbols']['N']
        #print("comp_step_rsum:", comp_step_rsum)

    comp_step_dict = {'sgd4d' : 2, 'sgd2d': 2, 'sgd1d': 2, 'relu_grad': 1, 'elem_add_grad': 1, 'reduce_sum': comp_step_rsum}
    #print("comp_step_dict:", comp_step_dict)

    ifmap_access_VMEM = 2 * DRAM_access_num_out * bw_ifmap * comp_step_dict[layer_op]  # for each computation 2 operand read and one operand write
    psum_access_VMEM = DRAM_access_num_out * bw_psum * (comp_step_dict[layer_op] - 1)
    # for now assuming quantization is free and happen inside the ALU before writing ofmap back to VMEM
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
    pipe_overhead_tile = (6 - 1) + (SysArray_col - 1)  #for now using PE col, after Sean knows how to handle the corner cases will veriy this
    
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



def batch_norm_forward_estimate(Hardware_param, LayerObj, SIMDResult_inflayer):
    # This function analytically estimates the cost for Banch_norm forward layer during training, will update the function with more accurate tiling formulation later

    #unpacking the parameters
    CompilerOut_layer = LayerObj.CompilerOut_layer  # the full compiler output for the layer
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
    DRAM_access_1 = (2 + 1) * IC * (2 * effective_mini_batch + 4)
    DRAM_access_2 = (IC + IC * effective_mini_batch * 2) * 4
    total_DRAM_access = (DRAM_access_1 + DRAM_access_2) * bw_ifmap
    #print("total_DRAM_access:", total_DRAM_access)

    #SRAM access count
    VMEM_access_1 = (2 + 1) * IC * (2 * effective_mini_batch + 4)
    VMEM_access_2 = ((2 + 1) * IC * effective_mini_batch) * 4
    total_VMEM_access = (VMEM_access_1 + VMEM_access_2) * bw_ifmap
    #print("total_VMEM_access:", total_VMEM_access)

    comp_cycle_1 = (IC * (2 * effective_mini_batch + 3) + (IC * inv_sqrt_cycles)) / SysArray_col   #omitting pipeline overhead
    comp_cycle_2 = (IC * effective_mini_batch * 4) / SysArray_col
    compute_cycles = comp_cycle_1 + comp_cycle_2
    #print("compute_cycles:", compute_cycles)

    DRAM_stall_cycles = total_DRAM_access/RBw_DRAM_to_VMEM
    #print("DRAM_stall_cycles:", DRAM_stall_cycles)

    # Access to instruction memory
    Nos_of_IF = compute_cycles  #number of instruction fetched
    InsMem_access = Nos_of_IF * SIMD_Ins_size    # in bit

    SIMDResult_inflayer.SRAM_access['InsMem'] = InsMem_access

    SIMDResult_inflayer.DRAM_access['ifmap'] = total_DRAM_access   # putting all DARM access in ifmap DRAM access for simplification
    SIMDResult_inflayer.SRAM_access['VMEM'] = total_VMEM_access

    SIMDResult_inflayer.cycles['compute'] = compute_cycles
    SIMDResult_inflayer.cycles['DRAM_stall'] = DRAM_stall_cycles
    SIMDResult_inflayer.cycles['total'] = compute_cycles + DRAM_stall_cycles


def batch_norm_backward_estimate(Hardware_param, LayerObj, SIMDResult_inflayer):
    # This function analytically estimates the cost for Banch_norm_backward pass layer, will update the function with more accurate tiling formulation later

    #unpacking the parameters
    CompilerOut_layer = LayerObj.CompilerOut_layer  # the full compiler output for the layer
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

    # Access to instruction memory
    Nos_of_IF = compute_cycles  #number of instruction fetched
    InsMem_access = Nos_of_IF * SIMD_Ins_size    # in bit

    SIMDResult_inflayer.SRAM_access['InsMem'] = InsMem_access

    SIMDResult_inflayer.DRAM_access['ifmap'] = total_DRAM_access   # putting all DARM access in ifmap DRAM access for simplification
    SIMDResult_inflayer.SRAM_access['VMEM'] = total_VMEM_access

    SIMDResult_inflayer.cycles['compute'] = compute_cycles
    SIMDResult_inflayer.cycles['DRAM_stall'] = DRAM_stall_cycles
    SIMDResult_inflayer.cycles['total'] = compute_cycles + DRAM_stall_cycles







    

        








   








