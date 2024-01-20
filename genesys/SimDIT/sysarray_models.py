# This file contains the functions for the data access and cycle count models for the Layers executed on the Systolic Array

import logging
import math
import numpy as np
from data_objects import HardwareObject, SAResult_Inflayer, SIMDResult_Inflayer
from layer_object import LayerObject


def conv_access_model(Hardware_param, LayerObj, SysResult_inflayer):
    # data access model for convolution layer

    #unpacking the parameters. Doing this unpacking at the beginning of each function
    bw_filter = LayerObj.bw_filter; bw_ifmap = LayerObj.bw_ifmap; bw_ofmap = LayerObj.bw_ofmap
    bw_psum = LayerObj.bw_psum; bw_bias = LayerObj.bw_bias

    OW = LayerObj.OW
    OH = LayerObj.OH
    OC = LayerObj.OC
    KW = LayerObj.KW
    KH = LayerObj.KH
    IC = LayerObj.IC 
    IW = LayerObj.IW 
    IH = LayerObj.IH
    Batch = LayerObj.Batch

    DTile_ow = LayerObj.DTile_ow
    DTile_oh = LayerObj.DTile_oh
    DTile_oc = LayerObj.DTile_oc
    DTile_kw = LayerObj.DTile_kw
    DTile_kh = LayerObj.DTile_kh
    DTile_ic = LayerObj.DTile_ic
    DTile_iw = LayerObj.DTile_iw
    DTile_ih = LayerObj.DTile_ih 
    DTile_batch = LayerObj.DTile_batch

    Loop_order = LayerObj.Loop_order
    fusion_status = LayerObj.fusion_status
    fusion_flag = LayerObj.fusion_flag      # fusion_flag overwrites fusion_status

    #print(Size_IBUF)
    #print(Loop_order)
    #print(Pad)

    # Determining which dataflow out of the three dataflow class form the input loop order
    WS_key = ['ow', 'oh', 'n']
    OS_key = ['kw', 'kh', 'ic']
    IS_key = ['oc']

    for key in WS_key:
        if Loop_order[0] == key:
            dataflow = "weight_stationary"
            break

    for key in OS_key:
        if Loop_order[0] == key:
            dataflow = "output_stationary"
            break


    for key in IS_key:
        if Loop_order[0] == key:
            dataflow = "input_stationary"
            break       

    #print(dataflow)

    ######### Model for DRAM accesses
    if (fusion_status == "NoFusion"):
        if dataflow == "weight_stationary":

            #imap access
            ifmap_access_DRAM = (DTile_iw * DTile_ih * DTile_ic * DTile_batch) * (OW/DTile_ow) * (OH/DTile_oh) * (Batch/DTile_batch) \
                                * (IC/DTile_ic) * (KW/DTile_kw) * (KH/DTile_kh) * math.ceil((OC/DTile_oc)) * bw_ifmap     # in bit
            
            #######filter access
            #common multiplier regardless of the variant of WS dataflow
            filter_access_common = (DTile_kw * DTile_kh * DTile_ic * DTile_oc) * (IC/DTile_ic) * (KW/DTile_kw) * (KH/DTile_kh) * (OC/DTile_oc) 

            ow_multiplier = OW/DTile_ow
            oh_multiplier = OH/DTile_oh
            n_multiplier = Batch/DTile_batch

            WS_Dict = {'ow': ow_multiplier, 'oh': oh_multiplier, 'n': n_multiplier}
            #print("WS_Dict:", WS_Dict)

            # First determining how many keys from the innermost loop matched in the given loop order
            loopids = {'first': "match", 'second': "nomatch", 'third': "nomatch"}  
            #the first loop id will always be one of the keys since this is under WS category, hence first one is matched by default. 
            #beginning with no match for the second and third ids and will change them to match depending on the cases
            #print("BEFORE:", loopids)

            for key in WS_key:
                if Loop_order[1] == key:
                    loopids['second'] = "match"

            if loopids['second'] == "nomatch":
                WScase = "oneKey" #case determined, only one key match, case 1, no further calculation needed
            else:
                for key in WS_key:
                    if Loop_order[2] == key:
                        loopids['third'] = "match"
                if loopids['third'] == "nomatch":
                    WScase = "twoKey"   #case determined, two keys matched, case 2, no further calculation needed
                else:
                    WScase = "threeKey"  #case determined, all three keys matched, case 3

            #print("AFTER:", loopids)
            #print("WS Case:", WScase)

            #Depending on the WScase, now determining filter multiplier based on how many innermost loops matches the WS_keys
            if WScase == "threeKey":
                filter_multiplier = 1  # all three key matched, so optimal WS, filter multiplier is 1
            elif WScase == "twoKey":
                for key in WS_key:
                    if key != Loop_order[0] and key != Loop_order[1]:  # tow key matched and one key does not match
                        mulkey = key
                #print("mulkey:", mulkey)
                filter_multiplier = WS_Dict[mulkey]
                
            elif WScase == "oneKey":
                mulkey1 = "empty"
                mulkey2 = "empty"
                for key in WS_key:
                    if key != Loop_order[0]:        # only one key matched, hence two unmatched key to identify
                        if mulkey1 == "empty" and mulkey2 == "empty":
                            mulkey1 = key           # one unmatched key is placed in mulkey 1
                        else:
                            mulkey2 = key           # another unmatched key is placed in mulkey 2
                print("mulkey1:", mulkey1)
                print("mulkey2:", mulkey2)
                filter_multiplier = WS_Dict[mulkey1] * WS_Dict[mulkey2]
            
            #print("filter_multiplier:", filter_multiplier)

            filter_access_DRAM = filter_access_common * filter_multiplier * bw_filter   # in bit

            #psum access
            ofpsm_access_DRAM = (DTile_ow * DTile_oh * DTile_oc * DTile_batch) * (OW/DTile_ow) * (OH/DTile_oh) * (Batch/DTile_batch) * (OC/DTile_oc) \
                                * (2 * math.ceil(IC/DTile_ic) * math.ceil(KW/DTile_kw) * math.ceil(KH/DTile_kh) - 1)

            psum_access_DRAM = (ofpsm_access_DRAM - (OW * OH * OC * Batch)) * bw_psum  #in bit

            #ofmap access
            ofmap_access_DRAM = OW * OH * OC * Batch * bw_ofmap  # in bit
            if fusion_flag == True:
                ofmap_access_DRAM = 0

            #bias access (oc at the outermost loop)
            bias_access_DRAM = DTile_oc * (OC/DTile_oc) * bw_bias
        
            #print("ifmap_access_DRAM:", ifmap_access_DRAM)
            #print("filter_access_DRAM:", filter_access_DRAM)
            #print("ofpsm_access_DRAM:", ofpsm_access_DRAM)
            #print("psum_access_DRAM:", psum_access_DRAM)
            print("ofmap_access_DRAM:", ofmap_access_DRAM)         
            #print("bias_access_DRAM:", bias_access_DRAM)

        elif dataflow == "output_stationary":
            print("not supported yet")
        elif dataflow == "input_stationary":
            print("not supported yet")
        else:
            print("Invalid dataflow")

    else:
        print("model for fusion do not exist yet")


    ##### Model for SRAM accesses 
    SRAM_stationary_flag = "NoStationary"  # genesys systolic PE hardware does not support any stationary logic for SRAM accesses

    if SRAM_stationary_flag == "NoStationary":
        conv_SRAM_access_NoStationary(Hardware_param, LayerObj, SysResult_inflayer)

    else:
        print("may write generic code for SRAM stationary logic based on dataflow")

    SysResult_inflayer.DRAM_access['filter'] = filter_access_DRAM
    SysResult_inflayer.DRAM_access['ifmap'] = ifmap_access_DRAM
    SysResult_inflayer.DRAM_access['ofmap'] = ofmap_access_DRAM
    SysResult_inflayer.DRAM_access['psum'] = psum_access_DRAM
    SysResult_inflayer.DRAM_access['bias'] = bias_access_DRAM

def conv_SRAM_access_NoStationary(Hardware_param, LayerObj, SysResult_inflayer):
    # Genesys PE hardware does not support any stationary logic for SRAM accesses
    # Hence SRAM access pattern does not depend on loop order or dataflow and this function gives the SRAM access pattern for this scenario

    # unpacking the parameters
    bw_filter = LayerObj.bw_filter; bw_ifmap = LayerObj.bw_ifmap; bw_ofmap = LayerObj.bw_ofmap
    bw_psum = LayerObj.bw_psum; bw_bias = LayerObj.bw_bias

    OW = LayerObj.OW
    OH = LayerObj.OH
    OC = LayerObj.OC
    KW = LayerObj.KW
    KH = LayerObj.KH
    IC = LayerObj.IC 
    IW = LayerObj.IW 
    IH = LayerObj.IH
    Batch = LayerObj.Batch

    DTile_ow = LayerObj.DTile_ow
    DTile_oh = LayerObj.DTile_oh
    DTile_oc = LayerObj.DTile_oc
    DTile_kw = LayerObj.DTile_kw
    DTile_kh = LayerObj.DTile_kh
    DTile_ic = LayerObj.DTile_ic
    DTile_iw = LayerObj.DTile_iw
    DTile_ih = LayerObj.DTile_ih 
    DTile_batch = LayerObj.DTile_batch

    Stile_ow = LayerObj.Stile_ow
    Stile_oh = LayerObj.Stile_oh
    Stile_oc = LayerObj.Stile_oc
    Stile_kw = LayerObj.Stile_kw
    Stile_kh = LayerObj.Stile_kh
    Stile_ic = LayerObj.Stile_ic 
    Stile_iw = LayerObj.Stile_iw 
    Stile_ih = LayerObj.Stile_ih 
    Stile_batch = LayerObj.Stile_batch

    #ifmap access
    ifmap_DRAM_loop_mul = (OW/DTile_ow) * (OH/DTile_oh) * (Batch/DTile_batch) * (IC/DTile_ic) * (KW/DTile_kw) * (KH/DTile_kh) * math.ceil((OC/DTile_oc))

    ifmap_access_SRAM = (Stile_iw * Stile_ih * Stile_ic * Stile_batch) * (DTile_ow/Stile_ow) * (DTile_oh/Stile_oh) * (DTile_batch/Stile_batch) \
                            * (DTile_ic/Stile_ic) * (DTile_kw/Stile_kw) * (DTile_kh/Stile_kh) * math.ceil((DTile_oc/Stile_oc)) * ifmap_DRAM_loop_mul * bw_ifmap # in bit


    # filter access
    filter_DRAM_loop_mul = math.ceil((OW/DTile_ow)) * math.ceil((OH/DTile_oh)) * math.ceil((Batch/DTile_batch)) \
                           * (IC/DTile_ic) * (KW/DTile_kw) * (KH/DTile_kh) * (OC/DTile_oc)

    filter_access_SRAM = (Stile_kw * Stile_kh * Stile_ic * Stile_oc) * (DTile_ic/Stile_ic) * (DTile_kw/Stile_kw) * (DTile_kh/Stile_kh) * (DTile_oc/Stile_oc) \
                        * math.ceil((DTile_ow/Stile_ow)) * math.ceil((DTile_oh/Stile_oh)) * math.ceil((DTile_batch/Stile_batch)) \
                        * filter_DRAM_loop_mul * bw_filter # in bit


    # psum access
    pDRAM_loop_mula = math.ceil(IC/DTile_ic) * math.ceil(KW/DTile_kw) * math.ceil(KH/DTile_kh)
    pDRAM_loop_mulb = (OW/DTile_ow) * (OH/DTile_oh) * (Batch/DTile_batch) * (OC/DTile_oc)

    psum_access_SRAM = (Stile_ow * Stile_oh * Stile_oc * Stile_batch) * (DTile_ow/Stile_ow) * (DTile_oh/Stile_oh) * (DTile_batch/Stile_batch) * (DTile_oc/Stile_oc) \
                      * (2 * math.ceil(DTile_ic/Stile_ic) * math.ceil(DTile_kw/Stile_kw) * math.ceil(DTile_kh/Stile_kh) * pDRAM_loop_mula - 1) \
                      * pDRAM_loop_mulb * bw_psum  # in bit
    

    # bias access, for each ofmap location, bias term need to be added once,
    #bias_access_SRAM = OH * OW * OC * Batch * bw_bias     
    bias_DRAM_loop_mul = (OW/DTile_ow) * (OH/DTile_oh) * (Batch/DTile_batch) * (OC/DTile_oc)
    bias_access_SRAM = (Stile_oc) * DTile_ow * DTile_oh * DTile_batch * (DTile_oc/Stile_oc) * bias_DRAM_loop_mul * bw_bias  # in bit

    #print("ifmap_access_SRAM:", ifmap_access_SRAM)
    #print("filter_access_SRAM:", filter_access_SRAM)
    #print("psum_access_SRAM:", psum_access_SRAM)
    #print("bias_access_SRAM:", bias_access_SRAM)

    SysResult_inflayer.SRAM_access['filter'] = filter_access_SRAM
    SysResult_inflayer.SRAM_access['ifmap'] = ifmap_access_SRAM
    SysResult_inflayer.SRAM_access['psum'] = psum_access_SRAM
    SysResult_inflayer.SRAM_access['bias'] = bias_access_SRAM


def conv_cycle_model(Hardware_param, LayerObj, SysResult_inflayer):
    #compute cycle and DRAM stall cycle count model for the convolution layer

    # unpacking the parameters
    SysArray_row = Hardware_param.SysArray_row; SysArray_col = Hardware_param.SysArray_col

    OW = LayerObj.OW
    OH = LayerObj.OH
    OC = LayerObj.OC
    KW = LayerObj.KW
    KH = LayerObj.KH
    IC = LayerObj.IC 
    IW = LayerObj.IW 
    IH = LayerObj.IH
    Batch = LayerObj.Batch

    DTile_ow = LayerObj.DTile_ow
    DTile_oh = LayerObj.DTile_oh
    DTile_oc = LayerObj.DTile_oc
    DTile_kw = LayerObj.DTile_kw
    DTile_kh = LayerObj.DTile_kh
    DTile_ic = LayerObj.DTile_ic
    DTile_iw = LayerObj.DTile_iw
    DTile_ih = LayerObj.DTile_ih 
    DTile_batch = LayerObj.DTile_batch

    Stile_ow = LayerObj.Stile_ow
    Stile_oh = LayerObj.Stile_oh
    Stile_oc = LayerObj.Stile_oc
    Stile_kw = LayerObj.Stile_kw
    Stile_kh = LayerObj.Stile_kh
    Stile_ic = LayerObj.Stile_ic 
    Stile_iw = LayerObj.Stile_iw 
    Stile_ih = LayerObj.Stile_ih 
    Stile_batch = LayerObj.Stile_batch

    fusion_status = LayerObj.fusion_status

    ### determining the on-chip compute cycles, compute cycles do not depend on loop order
    cycle_oneTile = (DTile_ow/Stile_ow) * (DTile_oh/Stile_oh) * (DTile_kw/Stile_kw) * (DTile_kh/Stile_kh) * (DTile_batch/Stile_batch) \
                    * math.ceil(DTile_ic/Stile_ic) * math.ceil(DTile_oc/Stile_oc)
    #print(cycle_oneTile)

    #pipeline overhead for each DRAM tile
    pipe_overhead_tile = (SysArray_row - 1) + (SysArray_col - 1)  

    #omitting the use of any ceil since DRAM tile size will be integer multiple of loops, 
    Number_of_Tile =  (OW/DTile_ow) * (OH/DTile_oh) * (Batch/DTile_batch) * (IC/DTile_ic) * (KW/DTile_kw) * (KH/DTile_kh) * (OC/DTile_oc)

    compute_cycles = math.ceil((cycle_oneTile + pipe_overhead_tile) * Number_of_Tile)   # giving the outer ceil to avoid fraction cycle numbers

    #print("compute_cycles:", compute_cycles)

    SysResult_inflayer.cycles['compute'] = compute_cycles
    
    #of cycles to compute one tile including the pipeline setup operhead, need this variable to compute DRAM stall cycles
    ComputeTile_cycles = cycle_oneTile + pipe_overhead_tile  

    ######## model for the DRAM stall cycles, depends on loop order, fusion etc
    if (fusion_status == "NoFusion"): #Model for the version where there is no fusion
        DRAM_stall_cycles = conv_stall_model_nofu(Hardware_param, LayerObj, ComputeTile_cycles, SysResult_inflayer)
    else:
        print("model for fusion do not exist yet")

    SysResult_inflayer.cycles['total'] = compute_cycles + DRAM_stall_cycles

    ####### Counting number of MAC operations: writing in a generic way for future extension (ceiling affects cycle count and #of MAC differently)
    PE_tile_mac = (Stile_ow * Stile_oh * Stile_oc * Stile_batch) * (Stile_ic * Stile_kw * Stile_kh)
    SRAM_tile_mac = PE_tile_mac * (DTile_ow/Stile_ow) * (DTile_oh/Stile_oh) * (DTile_kw/Stile_kw) * (DTile_kh/Stile_kh) * (DTile_batch/Stile_batch) \
                                                                                    * (DTile_ic/Stile_ic) * (DTile_oc/Stile_oc)
    Nos_of_mac = SRAM_tile_mac * (OW/DTile_ow) * (OH/DTile_oh) * (Batch/DTile_batch) * (IC/DTile_ic) * (KW/DTile_kw) * (KH/DTile_kh) * (OC/DTile_oc)

    print("Nos of MAC:", Nos_of_mac)
    SysResult_inflayer.arithmetic['mac'] = Nos_of_mac


def conv_stall_model_nofu(Hardware_param, LayerObj, ComputeTile_cycles, SysResult_inflayer):
    #DRAM stall cycle count model for the convolution layer when there is no fusion

    bw_filter = LayerObj.bw_filter; bw_ifmap = LayerObj.bw_ifmap; bw_ofmap = LayerObj.bw_ofmap
    bw_psum = LayerObj.bw_psum; bw_bias = LayerObj.bw_bias
    SysArray_row = Hardware_param.SysArray_row; SysArray_col = Hardware_param.SysArray_col
    RBW_DRAM_to_WBUF = Hardware_param.RBW_DRAM_to_WBUF # in bit/cycle, bias is also loaded through the same AXI interface
    RBW_DRAM_to_IBUF = Hardware_param.RBW_DRAM_to_IBUF
    RBW_DRAM_to_OBUF = Hardware_param.RBW_DRAM_to_OBUF
    WBW_OBUF_to_DRAM = Hardware_param.WBW_OBUF_to_DRAM

    OW = LayerObj.OW
    OH = LayerObj.OH
    OC = LayerObj.OC
    KW = LayerObj.KW
    KH = LayerObj.KH
    IC = LayerObj.IC 
    IW = LayerObj.IW 
    IH = LayerObj.IH
    Batch = LayerObj.Batch

    DTile_ow = LayerObj.DTile_ow
    DTile_oh = LayerObj.DTile_oh
    DTile_oc = LayerObj.DTile_oc
    DTile_kw = LayerObj.DTile_kw
    DTile_kh = LayerObj.DTile_kh
    DTile_ic = LayerObj.DTile_ic
    DTile_iw = LayerObj.DTile_iw
    DTile_ih = LayerObj.DTile_ih 
    DTile_batch = LayerObj.DTile_batch

    Loop_order = LayerObj.Loop_order
    fusion_flag = LayerObj.fusion_flag      

    # Determining which dataflow out of the three dataflow class form the input loop order
    WS_key = ['ow', 'oh', 'n']
    OS_key = ['kw', 'kh', 'ic']
    IS_key = ['oc']

    for key in WS_key:
        if Loop_order[0] == key:
            dataflow = "weight_stationary"
            break

    for key in OS_key:
        if Loop_order[0] == key:
            dataflow = "output_stationary"
            break


    for key in IS_key:
        if Loop_order[0] == key:
            dataflow = "input_stationary"
            break       

    #print("Dataflow:", dataflow)


    if dataflow == "weight_stationary":
        # The current DRAM stall model is valid for any WS loop order with oc at the outermost loop (DUE TO SOME CORNER SITUATIONs, EXTENSION IS POSSIBLE)
        
        Loop_order1 = ['ow', 'oh', 'kw', 'kh', 'ic', 'n', 'oc'] # current GeneSys Loop order
        Loop_order2 = ['ow', 'oh', 'n', 'kw', 'kh', 'ic', 'oc'] # an optimal WS loop order, there are equivalent varients of these loop order

        if (Loop_order == Loop_order1 and (OW/DTile_ow * OH/DTile_oh) > 2) or (Loop_order == Loop_order2 and (OW/DTile_ow * OH/DTile_oh * Batch/DTile_batch) > 2):
            # The tiling condition ensures that the numbers of WS tiles is at least 3 to be able to normally execute the 3 stage double-buffered DRAM pipeline
            No_Warning = "True"
        else:
            print("WARNING: Number of WS tile is less than 3, additional stalls for the 3-stage double-buffered DRAM pipeline is not modeled yet")
            print("Nos of WS tile:", (OW/DTile_ow * OH/DTile_oh * Batch/DTile_batch))
            print("Nos of DRAM WS + OS tiles:", (OW/DTile_ow) * (OH/DTile_oh) * (Batch/DTile_batch) * (IC/DTile_ic) * (KW/DTile_kw) * (KH/DTile_kh))
            print("Nos of total DRAM tiles:", (OW/DTile_ow) * (OH/DTile_oh) * (Batch/DTile_batch) * (IC/DTile_ic) * (KW/DTile_kw) * (KH/DTile_kh) * (OC/DTile_oc))

        #print("OH:", OH, "OW:", OW, "DTile_oh:", DTile_oh, "DTile_ow:", DTile_ow)
        
        if (Loop_order == Loop_order1) or (Loop_order == Loop_order2):
            if Loop_order == Loop_order1:
                filter_multiplier = Batch/DTile_batch
            elif Loop_order == Loop_order2:
                filter_multiplier = 1
            #print(filter_multiplier)

            #of tiles where weights are being loaded (regardless of bias)
            NT_weight = (KW/DTile_kw) * (KH/DTile_kh) * (IC/DTile_ic) * (OC/DTile_oc) * filter_multiplier
            #of tiles where (weight + bias) are being loaded. (bias is loaded with the oc loop)
            NT_wgt_bias = OC/DTile_oc
            #of tiles where only weights are being loaded
            NT_wgt_only = NT_weight - NT_wgt_bias

            #of tiles where psum is written to the DRAM
            NT_ps_wrt = (OW/DTile_ow) * (OH/DTile_oh) * (Batch/DTile_batch) * (IC/DTile_ic) * (KW/DTile_kw) * (KH/DTile_kh) * (OC/DTile_oc)
            #of tiles where psum write only happens
            NT_ps_wrtonly = (OW/DTile_ow) * (OH/DTile_oh) * (Batch/DTile_batch) * (OC/DTile_oc)
            #of tiles where both psum read and write occur
            NT_ps_rdwrt = NT_ps_wrt - NT_ps_wrtonly

            #print("NT_weight:", NT_weight, ";", "NT_wgt_bias:", NT_wgt_bias, ";", "NT_wgt_only:", NT_wgt_only)
            #print("NT_ps_wrt:", NT_ps_wrt, ";", "NT_ps_wrtonly:", NT_ps_wrtonly, ";", "NT_ps_rdwrt:", NT_ps_rdwrt)

            ## Performing CASE counts
            #CASE-5: #of tiles where weight+bias is being loaded (exclude the first tile)
            NT_case5 = NT_wgt_bias - 1
            #CASE-4: #of tiles where only weight is being loaded
            NT_case4 = NT_wgt_only
            #CASE-1: #of tiles where ifmap read and psum write happens (exclude the last 2 tiles)
            NT_case1 = (NT_ps_wrtonly - 2) - NT_case5
            #CASE-2: #of tiles where ifmap read and psum read+write happens
            NT_case2 = NT_ps_rdwrt - NT_case4

            #print("NT_case1:", NT_case1, "NT_case2:", NT_case2, "NT_case4:", NT_case4, "NT_case5:", NT_case5)

            ## condition to address the situation when tiles from ic, kw, kh, oc, n loops are equal to their original dimensions
            if (NT_case2 + NT_case4 == 0):
                NT_case2 = 0
                NT_case4 = 0

            print("NT_case1:", NT_case1, "NT_case2:", NT_case2, "NT_case4:", NT_case4, "NT_case5:", NT_case5)

            #The following two tiles are placing as seperate cases for future exception code when WS tiles can be < 3. There it is possible for these cases to be zero 
            NT_case7 = 1   # The second tile
            NT_case8 = 1   # The second last tile

            #of cycles required to load/store each tile of each kind of data
            WgtTile_load_cycles = math.ceil((DTile_kw * DTile_kh * DTile_ic * DTile_oc * bw_filter) / RBW_DRAM_to_WBUF)
            BiasTile_load_cycles = math.ceil((DTile_oc * bw_bias) / RBW_DRAM_to_WBUF)
            ifmapTile_load_cycles = math.ceil((DTile_iw * DTile_ih * DTile_ic * DTile_batch * bw_ifmap) / RBW_DRAM_to_IBUF)
            psumTile_load_cycles = math.ceil((DTile_ow * DTile_oh * DTile_oc * DTile_batch * bw_psum) / RBW_DRAM_to_OBUF)
            psumTile_store_cycles = math.ceil((DTile_ow * DTile_oh * DTile_oc * DTile_batch * bw_psum) / WBW_OBUF_to_DRAM)
            #do not need to use 8-bit ofmap. Since SIMD operations are 32 bit and there is always at least a ReLU layer after each
            #Conv layer, the output of conv will go to SIMD and the quantization of 32 to 8 bit happens at SIMD. Hence the ofmap from a conv will be 32 bit

            #print("computeTile_cycles:", ComputeTile_cycles)
            #print("WgtTile_load_cycles:", WgtTile_load_cycles)
            #print("BiasTile_load_cycles:", BiasTile_load_cycles)
            #print("ifmapTile_load_cycles:", ifmapTile_load_cycles)
            #print("psumTile_load_cycles:", psumTile_load_cycles)
            #print("psumTile_store_cycles:", psumTile_store_cycles)

            # Determining the #of stall cycles for each case
            #Case1
            L11 = ifmapTile_load_cycles - ComputeTile_cycles
            L12 = psumTile_store_cycles - ComputeTile_cycles
            stall_case1 = max(0, L11, L12) * NT_case1

            #Case2
            L21 = ifmapTile_load_cycles - ComputeTile_cycles
            L22 = (psumTile_load_cycles + psumTile_store_cycles - ComputeTile_cycles) #######Assuming one AXI for both read and write of psum. 
            stall_case2 = max(0, L21, L22) * NT_case2

            #Case4
            L41 = ifmapTile_load_cycles - ComputeTile_cycles
            L42 = WgtTile_load_cycles - ComputeTile_cycles
            L43 = (psumTile_load_cycles + psumTile_store_cycles - ComputeTile_cycles) ######Assuming one AXI for both read and write of psum. 
            stall_case4 = max(0, L41, L42, L43) * NT_case4

            #Case5
            L51 = ifmapTile_load_cycles - ComputeTile_cycles
            L52 = psumTile_store_cycles - ComputeTile_cycles
            L53 = (WgtTile_load_cycles + BiasTile_load_cycles) - ComputeTile_cycles
            stall_case5 = max(0, L51, L52, L53) * NT_case5

            print("stall_case1:", stall_case1, "; stall_case2:", stall_case2, "; stall_case4:", stall_case4, "; stall_case5:", stall_case5)

            #First tile
            Lf1 = ifmapTile_load_cycles
            Lf2 = WgtTile_load_cycles + BiasTile_load_cycles
            stall_first = max(Lf1, Lf2)

            #Last tile
            stall_last = psumTile_store_cycles

            #Case7
            L71 = ifmapTile_load_cycles - ComputeTile_cycles
            stall_case7 = max(0, L71) * NT_case7

            #Case8
            L81 = psumTile_store_cycles - ComputeTile_cycles
            stall_case8 = max(0, L81)

            print("stall_first:", stall_first, "; stall_last:", stall_last, "; stall_case7:", stall_case7, "; stall_case8:", stall_case8)

            #of total DRAM stall cycles
            DRAM_stall_cycles = stall_case1 + stall_case2 + stall_case4 + stall_case5 + stall_case7 + stall_case8 + stall_first + stall_last
            print("DRAM_stall_cycles nofusion:", DRAM_stall_cycles)


            ####### incorporating fusion:
            if fusion_flag == True:
                # taking the weighted average stall per tile. Some cases occur more frequencty than the others. Hence weighted average is a better estimate
                if (NT_case1 + NT_case2 + NT_case4 + NT_case5) <= 0:
                    avg_pertile_stall = 0
                else:
                    avg_pertile_stall = math.ceil((stall_case1 + stall_case2 + stall_case4 + stall_case5) / (NT_case1 + NT_case2 + NT_case4 + NT_case5))
                print("avg_pertile_stall:", avg_pertile_stall)

                nos_of_ofmap_tile = (OW/DTile_ow) * (OH/DTile_oh) * (Batch/DTile_batch) * (OC/DTile_oc)

                stall_four_cases = stall_case1 + stall_case2 + stall_case4 + stall_case5
                reduction_four_cases = math.ceil(avg_pertile_stall * (nos_of_ofmap_tile - 2))   # excluding last two tiles

                if reduction_four_cases > stall_four_cases:  #handling corner situation due to averaging: reduction from four cases need to be <= stall from all four cases
                    net_reduction = stall_four_cases
                else:
                    net_reduction = reduction_four_cases

                fusion_reduced_stall = net_reduction + stall_case8 + stall_last        # last two tiles are always ofmap write

                DRAM_stall_cycles = DRAM_stall_cycles - fusion_reduced_stall

                print("nos_of_ofmap_tile:", nos_of_ofmap_tile)
                print("fusion_reduced_stall:", fusion_reduced_stall)
                print("DRAM_stall_cycles with fusion:", DRAM_stall_cycles)
                assert (DRAM_stall_cycles >= 0)


            SysResult_inflayer.cycles['DRAM_Stall'] = DRAM_stall_cycles

        else:
            print("WS DRAM stall model do not exist yet, need some modification")

    elif dataflow == "output_stationary":
        print("DARM stall model do not exist yet")
    elif dataflow == "input_stationary":
        print("DRAM stall model do not exist yet")
    else:
        print("Invalid dataflow")

    return DRAM_stall_cycles


def gemm_access_model(Hardware_param, LayerObj, SysResult_inflayer):
    # data access model for fully connected layer (i.e., gemm)

    #unpacking the parameters. Doing this unpacking at the beginning of each function
    bw_filter = LayerObj.bw_filter; bw_ifmap = LayerObj.bw_ifmap; bw_ofmap = LayerObj.bw_ofmap
    bw_psum = LayerObj.bw_psum; bw_bias = LayerObj.bw_bias

    OC = LayerObj.OC
    IC = LayerObj.IC 
    Batch = LayerObj.Batch

    DTile_oc = LayerObj.DTile_oc
    DTile_ic = LayerObj.DTile_ic 
    DTile_batch = LayerObj.DTile_batch

    Loop_order = LayerObj.Loop_order
    fusion_status = LayerObj.fusion_status
    fusion_flag = LayerObj.fusion_flag      # fusion_flag overwrites fusion_status

    # Current implementation is for one loop order only, this is sort of the most optimal loop order for gemm analytically.
    # So no need to implement the support for any loop order for gemm
    if Batch > 1 and Loop_order == ['n', 'ic', 'oc']:      # weight stationary category
        LayerObj.Loop_order = ['ow', 'oh', 'n', 'kw', 'kh', 'ic', 'oc']   # converting FC loop order to convolution loop order
        conv_access_model(Hardware_param, LayerObj, SysResult_inflayer)
        LayerObj.Loop_order = Loop_order   # doing this to retain the original Loop order in LayerObj so that it can be used in later function calls

    elif Batch == 1 and Loop_order == ['n', 'ic', 'oc']:   #output stationary category
        ###### Model for DRAM access cost
        if (fusion_status == "NoFusion"):
            # ifmap access
            if math.ceil(IC/DTile_ic) == 1: 
                ifmap_oc_multiplier = 1         # the loop becomes input stationary wrt DRAM access
            else:
                ifmap_oc_multiplier = OC/DTile_oc
            #print(ifmap_oc_multiplier)  
            ifmap_access_DRAM = (DTile_ic) * (IC/DTile_ic) * (OC/DTile_oc) * bw_ifmap

            # filter access
            filter_access_DRAM = (DTile_ic * DTile_oc) * (IC/DTile_ic) * (OC/DTile_oc) * bw_filter  # in bit
            
            # ofmap access, no pusm DRAM access since output stationary
            ofmap_access_DRAM = (DTile_oc) * (OC/DTile_oc) * bw_ofmap
            if fusion_flag == True:
                ofmap_access_DRAM = 0

            # bias access
            bias_access_DRAM = (DTile_oc) * (OC/DTile_oc) * bw_bias           
        else:
            print("model for fusion do not exist yet")

        ##### Model for SRAM accesses (Original SRAM access do not depend on fusion)
        SRAM_stationary_flag = "NoStationary"  # genesys systolic PE hardware does not support any stationary logic for SRAM accesses
        if SRAM_stationary_flag == "NoStationary":
            conv_SRAM_access_NoStationary(Hardware_param, LayerObj, SysResult_inflayer)
        else:
            print("will write generic code for SRAM stationary logic based on dataflow")

        SysResult_inflayer.DRAM_access['filter'] = filter_access_DRAM
        SysResult_inflayer.DRAM_access['ifmap'] = ifmap_access_DRAM
        SysResult_inflayer.DRAM_access['ofmap'] = ofmap_access_DRAM
        #SysResult_inflayer.DRAM_access['psum'] = psum_access_DRAM
        SysResult_inflayer.DRAM_access['bias'] = bias_access_DRAM

    else:
        print("The input loop order is not optimal and not supported")


def gemm_cycle_model(Hardware_param, LayerObj, SysResult_inflayer):
    #compute cycle and DRAM stall cycle count model for the fully connecetd layer

    # unpacking the parameters
    SysArray_row = Hardware_param.SysArray_row; SysArray_col = Hardware_param.SysArray_col

    OW = LayerObj.OW
    OH = LayerObj.OH
    OC = LayerObj.OC
    KW = LayerObj.KW
    KH = LayerObj.KH
    IC = LayerObj.IC 
    IW = LayerObj.IW 
    IH = LayerObj.IH
    Batch = LayerObj.Batch

    DTile_ow = LayerObj.DTile_ow
    DTile_oh = LayerObj.DTile_oh
    DTile_oc = LayerObj.DTile_oc
    DTile_kw = LayerObj.DTile_kw
    DTile_kh = LayerObj.DTile_kh
    DTile_ic = LayerObj.DTile_ic
    DTile_iw = LayerObj.DTile_iw
    DTile_ih = LayerObj.DTile_ih 
    DTile_batch = LayerObj.DTile_batch

    Stile_ow = LayerObj.Stile_ow
    Stile_oh = LayerObj.Stile_oh
    Stile_oc = LayerObj.Stile_oc
    Stile_kw = LayerObj.Stile_kw
    Stile_kh = LayerObj.Stile_kh
    Stile_ic = LayerObj.Stile_ic 
    Stile_iw = LayerObj.Stile_iw 
    Stile_ih = LayerObj.Stile_ih 
    Stile_batch = LayerObj.Stile_batch

    Loop_order = LayerObj.Loop_order
    fusion_status = LayerObj.fusion_status

    # Current implementation is for one loop order only, this is sort of the most optimal loop order for gemm analytically.
    # So no need to implement the support for any loop order for gemm
    if Batch > 1 and Loop_order == ['n', 'ic', 'oc']:      # weight stationary category
        LayerObj.Loop_order = ['ow', 'oh', 'n', 'kw', 'kh', 'ic', 'oc']   # converting FC loop order to convolution loop order
        conv_cycle_model(Hardware_param, LayerObj, SysResult_inflayer)
        LayerObj.Loop_order = Loop_order   # doing this to retain the original Loop order in LayerObj so that it can be used in later function calls if needed

    elif Batch == 1 and Loop_order == ['n', 'ic', 'oc']:   
        ### determining computing cycles, using the convolution equations cause that works
        #determining the on-chip compute cycles, compute cycles do not depend on loop order, or fusion
        cycle_oneTile = (DTile_ow/Stile_ow) * (DTile_oh/Stile_oh) * (DTile_kw/Stile_kw) * (DTile_kh/Stile_kh) * (DTile_batch/Stile_batch) \
                    * math.ceil(DTile_ic/Stile_ic) * math.ceil(DTile_oc/Stile_oc)
        #print("cycle_oneTile:", cycle_oneTile)
        #pipeline overhead for each DRAM tile
        pipe_overhead_tile = (SysArray_row - 1) + (SysArray_col - 1)  
        #omitting the use of any ceil since DRAM tile size will be integer multiple of loops, 
        Number_of_Tile =  (OW/DTile_ow) * (OH/DTile_oh) * (Batch/DTile_batch) * (IC/DTile_ic) * (KW/DTile_kw) * (KH/DTile_kh) * (OC/DTile_oc)

        compute_cycles = math.ceil((cycle_oneTile + pipe_overhead_tile) * Number_of_Tile)   # giving the outer ceil to avoid fraction cycle numbers
        #print("compute_cycles:", compute_cycles)

        SysResult_inflayer.cycles['compute'] = compute_cycles
    
        #of cycles to compute one tile including the pipeline setup operhead, need this variable to compute DRAM stall cycles
        ComputeTile_cycles = cycle_oneTile + pipe_overhead_tile

        ######## model for the DRAM stall cycles, depends on loop order, fusion etc
        if (fusion_status == "NoFusion"): #Model for the version where there is no fusion
            DRAM_stall_cycles = gemmb1_stall_model_nofu(Hardware_param, LayerObj, ComputeTile_cycles, SysResult_inflayer)  # stall model for batch = 1, output stationary
        else:
            print("model for fusion do not exist yet")

        SysResult_inflayer.cycles['total'] = compute_cycles + DRAM_stall_cycles

        ####### Counting number of MAC operations: using the convolution equations cause that works (ceiling affects cycle count and #of MAC differently)
        PE_tile_mac = (Stile_ow * Stile_oh * Stile_oc * Stile_batch) * (Stile_ic * Stile_kw * Stile_kh)
        SRAM_tile_mac = PE_tile_mac * (DTile_ow/Stile_ow) * (DTile_oh/Stile_oh) * (DTile_kw/Stile_kw) * (DTile_kh/Stile_kh) * (DTile_batch/Stile_batch) \
                                                                                        * (DTile_ic/Stile_ic) * (DTile_oc/Stile_oc)
        Nos_of_mac = SRAM_tile_mac * (OW/DTile_ow) * (OH/DTile_oh) * (Batch/DTile_batch) * (IC/DTile_ic) * (KW/DTile_kw) * (KH/DTile_kh) * (OC/DTile_oc)

        print("Nos of MAC:", Nos_of_mac)
        SysResult_inflayer.arithmetic['mac'] = Nos_of_mac

    else:
        print("The input loop order is not optimal and not supported")


def gemmb1_stall_model_nofu(Hardware_param, LayerObj, ComputeTile_cycles, SysResult_inflayer):
    #DRAM stall cycle count model for the gemm layer when there is no fusion, batch size = 1, output stationary

    bw_filter = LayerObj.bw_filter; bw_ifmap = LayerObj.bw_ifmap; bw_ofmap = LayerObj.bw_ofmap
    bw_psum = LayerObj.bw_psum; bw_bias = LayerObj.bw_bias
    RBW_DRAM_to_WBUF = Hardware_param.RBW_DRAM_to_WBUF # in bit/cycle, bias is also loaded through the same AXI interface
    RBW_DRAM_to_IBUF = Hardware_param.RBW_DRAM_to_IBUF
    RBW_DRAM_to_OBUF = Hardware_param.RBW_DRAM_to_OBUF
    WBW_OBUF_to_DRAM = Hardware_param.WBW_OBUF_to_DRAM

    OC = LayerObj.OC
    IC = LayerObj.IC 
    Batch = LayerObj.Batch

    DTile_oc = LayerObj.DTile_oc
    DTile_ic = LayerObj.DTile_ic
    DTile_batch = LayerObj.DTile_batch

    fusion_flag = LayerObj.fusion_flag      # fusion_flag overwrites fusion_status

    #of cycles required to load/store each tile of each kind of data
    WgtTile_load_cycles = math.ceil((DTile_ic * DTile_oc * bw_filter) / RBW_DRAM_to_WBUF)
    BiasTile_load_cycles = math.ceil((DTile_oc * bw_bias) / RBW_DRAM_to_WBUF)
    ifmapTile_load_cycles = math.ceil((DTile_ic * bw_ifmap) / RBW_DRAM_to_IBUF)
    ofmapTile_store_cycles = math.ceil((DTile_oc * bw_ofmap) / WBW_OBUF_to_DRAM)
    #do not need to use 8-bit ofmap. Since SIMD operations are 32 bit and there is always at least a ReLU layer after each
    #Conv layer, the output of conv will go to SIMD and the quantization of 32 to 8 bit happens at SIMD. Hence the ofmap from a conv will be 32 bit

    if fusion_flag == True:
        ofmapTile_store_cycles = 0   #for gemm no pusm, all DRAM write is ofmap. Hence setting ofmapTile_store_cycles = 0 will autometically give stall cycles for fusion
        print("ofmapTile_store_cycles:", ofmapTile_store_cycles)

    #print("ComputeTile_cycles:", ComputeTile_cycles)
    #print("WgtTile_load_cycles:", WgtTile_load_cycles)

    if math.ceil(IC/DTile_ic) == 1:
        dataflow = "input_stationary"  
    else:
        dataflow = "output_stationary"

    if dataflow == "input_stationary":  
        ## Performing CASE counts, there is only one case
        #Case 1: #of tiles where weight+bias is loaded, and ofmap write occurs, except the first two and last two tiles
        NT_case1 = (OC/DTile_oc) - 2

        # Using this condition to seperately address the situation when OC/DTile_oc is also 1 and NT_case1 becomes negative
        NT_case1_flag = "None"
        if NT_case1 < 0:
            NT_case1 = 0
            NT_case1_flag = "Negative"   
        #print("NT_case1:", NT_case1)

        # Determining the #of stall cycles for each case
        #Case1
        L11 = (WgtTile_load_cycles + BiasTile_load_cycles) - ComputeTile_cycles
        L12 = ofmapTile_store_cycles - ComputeTile_cycles
        stall_case1 = max(0, L11, L12) * NT_case1

        #First tile
        Lf1 = ifmapTile_load_cycles
        Lf2 = WgtTile_load_cycles + BiasTile_load_cycles
        stall_first = max(Lf1, Lf2)

        #Second tile
        L2nd = (WgtTile_load_cycles + BiasTile_load_cycles) - ComputeTile_cycles
        stall_second = max(0, L2nd)

        #Second last tile
        L2ndlst = ofmapTile_store_cycles - ComputeTile_cycles
        stall_secondlast = max(0, L2ndlst)

        #Last tile
        stall_last = ofmapTile_store_cycles

        #print("stall_case1:", stall_case1, "; stall_first:", stall_first, "; stall_second:", stall_second, "; stall_secondlast:", stall_secondlast,\
        #                                                                                                         "; stall_last:", stall_last)
        #of total DRAM stall cycles
        if NT_case1_flag == "Negative":
            DRAM_stall_cycles = stall_first + stall_last
        else:
            DRAM_stall_cycles = stall_case1 + stall_first + stall_second + stall_secondlast + stall_last
    
    elif dataflow == "output_stationary":  
        #of tiles where weights are being loaded (regardless of bias)
        NT_weight = (IC/DTile_ic) * (OC/DTile_oc) 
        #of tiles where (weight + bias) are being loaded. (bias is loaded with the oc loop)
        NT_wgt_bias = OC/DTile_oc
        #of tiles where only weights are being loaded
        NT_wgt_only = NT_weight - NT_wgt_bias
        #of tiles where ofmap is written to the DRAM
        NT_ofmap_wrt = (OC/DTile_oc)

        #print("NT_weight:", NT_weight, ";", "NT_wgt_bias:", NT_wgt_bias, ";", "NT_wgt_only:", NT_wgt_only, "; NT_ofmap_wrt:", NT_ofmap_wrt)

        ## Performing CASE counts
        #CASE-1: #of tiles where weight+bias is being loaded (exclude the first tile)
        NT_case1 = NT_wgt_bias - 1
        #CASE-4: #of tiles where ofmap write occurs (excluding the last tile, ofmap write does not happen at the second last tile)
        NT_case4 = NT_ofmap_wrt - 1
        #CASE-3: #of tiles where weightonly read and ifmap read happens (excluding the second tile)
        NT_case3 = (NT_wgt_only - 1) - NT_case4

        #print("NT_case1:", NT_case1, "NT_case3:", NT_case3, "NT_case4:", NT_case4)

        # Determining the #of stall cycles for each case
        #Case1
        L11 = (WgtTile_load_cycles + BiasTile_load_cycles) - ComputeTile_cycles
        L12 = ifmapTile_load_cycles - ComputeTile_cycles
        stall_case1 = max(0, L11, L12) * NT_case1

        #Case3
        L31 = WgtTile_load_cycles  - ComputeTile_cycles
        L32 = ifmapTile_load_cycles - ComputeTile_cycles
        stall_case3 = max(0, L31, L32) * NT_case3

        #Case4
        L41 = WgtTile_load_cycles  - ComputeTile_cycles
        L42 = ifmapTile_load_cycles - ComputeTile_cycles
        L43 = ofmapTile_store_cycles - ComputeTile_cycles
        stall_case4 = max(0, L41, L42, L43) * NT_case4

        #print("stall_case1:", stall_case1, "; stall_case3:", stall_case3, "; stall_case4:", stall_case4)

        #First tile
        Lf1 = ifmapTile_load_cycles
        Lf2 = WgtTile_load_cycles + BiasTile_load_cycles
        stall_first = max(Lf1, Lf2)

        #Second tile
        L2nd1 = WgtTile_load_cycles - ComputeTile_cycles
        L2nd2 = ifmapTile_load_cycles - ComputeTile_cycles
        stall_second = max(0, L2nd1, L2nd2)

        #Second last tile, there is no data read/write for the second last tile, only compute
        stall_secondlast = 0

        #Last tile
        stall_last = ofmapTile_store_cycles

        #print("stall_first:", stall_first, "; stall_second:", stall_second, "; stall_secondlast:", stall_secondlast, "; stall_last:", stall_last)

        #of total DRAM stall cycles
        DRAM_stall_cycles = stall_case1 + stall_case3 + stall_case4 + stall_first + stall_second + stall_secondlast + stall_last

    #print("DRAM_stall_cycles:", DRAM_stall_cycles)
    SysResult_inflayer.cycles['DRAM_Stall'] = DRAM_stall_cycles

    return DRAM_stall_cycles












