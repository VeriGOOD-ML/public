# This file implements the internal tiling generator

import logging
import math
import numpy as np
import copy
from pprint import pprint
import json

class LayerSpecTemplate(object):
    # This is the template to specify the spec of any layer
    def __init__(self):
        self.OW = "None"
        self.OH = "None"
        self.OC = "None"
        self.KW = "None"
        self.KH = "None"
        self.IC = "None"
        self.Batch = "None"
        self.Stride = "None"        
        self.IW = "None"
        self.IH = "None"

        ######Tiling parameters from DRAM to SRAM level 
        self.DTile_ow = "None"
        self.DTile_oh = "None"
        self.DTile_oc = "None"
        self.DTile_kw = "None"
        self.DTile_kh = "None"
        self.DTile_ic = "None"
        self.DTile_batch = "None"
        self.DTile_iw = "None"
        self.DTile_ih = "None"

        #Tiling parameters from SRAM to PE level
        self.Stile_ow = "None"
        self.Stile_oh = "None"
        self.Stile_oc = "None"
        self.Stile_kw = "None"
        self.Stile_kh = "None"
        self.Stile_ic = "None"
        self.Stile_batch = "None"
        self.Stile_iw = "None"
        self.Stile_ih = "None"


def generate_tile_relu(LSpec, Hardware_param):
    # Unpacking the parameters
    # SIMD layers are single buffered
    # do the channle padding inside and then pass it as a parameter to override the channel in LSpec object
    bw_input = Hardware_param.SAbw_psum
    #bw_output = Hardware_param.SAbw_psum
    SysArray_col = Hardware_param.SysArray_col
    size_VMEM1_kb = Hardware_param.Size_VMEM1
    #size_VMEM2_kb = Hardware_param.Size_VMEM2

    min_VMEM1_bit = SysArray_col * bw_input
    assert (size_VMEM1_kb * 8 * 1024) >= min_VMEM1_bit
    
    OW = LSpec.OW
    OH = LSpec.OH
    OC = LSpec.OC
    Batch = LSpec.Batch

    # Zero padding the channel dimension if array column > OC. Besides, Make OC an integer multiple of SysArray_col
    if SysArray_col >= OC:
        OC = SysArray_col
    else:
        if (OC % SysArray_col) != 0:
            rem = OC % SysArray_col
            OC = OC - rem + SysArray_col
    #print("OC:", OC)

    # Operand1 goes to VMEM1, Output goes to VMEM2. Hence, the tiling of the operand1 need to fit in VMEM1: VMEM1 = VMEM2
    # Set OH_tile = OW_tile, these two are first priority dimension, OC is second prority dimension given OC_tile >= SysArray_col, and Batch is the 3rd prority dimension
    OC_tile = SysArray_col
    Batch_tile = 1
    VMEM1_bit = size_VMEM1_kb * 8 * 1024
    OW_tile = OW
    OH_tile = OH
    valid_flag = True
    i = 2
    while ((OC_tile * OW_tile * OH_tile * Batch_tile * bw_input) > VMEM1_bit) or valid_flag == False:
        OW_tile = OW/i
        OH_tile = OH/i
        i = i + 1
        valid_flag = True
        if (OW_tile % 1) != 0 or (OH_tile % 1) != 0:
            valid_flag = False
    #print ("OW_tile:", OW_tile, "OH_tile:", OH_tile)

    #Starting from a valid OC_tile, now making it better
    j = 2
    while (OC_tile * OW_tile * OH_tile * Batch_tile * bw_input) < VMEM1_bit:
        OC_tile_new = SysArray_col * j
        if (OC_tile_new * OW_tile * OH_tile * Batch_tile * bw_input) > VMEM1_bit:
            break
        if OC_tile_new > OC:  
            break
        OC_tile = OC_tile_new
        j = j + 1
    #print("OC_tile:", OC_tile)


    #Finding batch_tile same way as OW_tile and OH_tile
    Batch_tile = Batch
    valid_flag = True
    k = 2
    while ((OC_tile * OW_tile * OH_tile * Batch_tile * bw_input) > VMEM1_bit) or valid_flag == False:
        Batch_tile = Batch/k
        k = k + 1
        valid_flag = True
        if (Batch_tile % 1) != 0:
            valid_flag = False
    #print ("Batch_tile:", Batch_tile)

    # All tiling calculated: placing asserts to ensure the tiles fit in the respective buffer
    assert (OW_tile * OH_tile * OC_tile * Batch_tile * bw_input) <= VMEM1_bit

    # Storing all results in LSpec object
    #Replacing the original OC with Padded OC
    LSpec.OC = OC
    # SRAM to DRAM tiling results are storing in the LSpec object
    LSpec.DTile_ow = OW_tile
    LSpec.DTile_oh = OH_tile
    LSpec.DTile_oc = OC_tile
    LSpec.DTile_kw = 1
    LSpec.DTile_kh = 1
    LSpec.DTile_ic = "None"
    LSpec.DTile_batch = Batch_tile
    LSpec.DTile_iw = (LSpec.DTile_ow - 1) * LSpec.Stride + LSpec.DTile_kw
    LSpec.DTile_ih = (LSpec.DTile_oh - 1) * LSpec.Stride + LSpec.DTile_kh

    # DRAM to SRAM Tiling depend on the array size only
    LSpec.Stile_ow = 1
    LSpec.Stile_oh = 1
    LSpec.Stile_oc = SysArray_col
    LSpec.Stile_kw = 1
    LSpec.Stile_kh = 1
    LSpec.Stile_ic = "None"
    LSpec.Stile_batch = 1
    LSpec.Stile_iw = 1
    LSpec.Stile_ih = 1

def generate_tile_elemadd(LSpec, Hardware_param):
    # Op1 need to fit in VMEM1, op2 need to fit in VMEM2, output need to fit in either VMEM1 or VMEM2. The tile sizes for op1, op2, and output are same.
    # So basically the tiling generation process is exactly same as relu
    generate_tile_relu(LSpec, Hardware_param)

def generate_tile_pool(LSpec, Hardware_param):
    # Unpacking the parameters
    # SIMD layers are single buffered
    # do the channle padding inside and then pass it as a parameter to override the channel in LSpec object
    bw_input = Hardware_param.SAbw_psum
    SysArray_col = Hardware_param.SysArray_col
    size_VMEM1_kb = Hardware_param.Size_VMEM1

    #VMEM1 stores Op1 and VMEM2 stores output
    min_VMEM1_bit = SysArray_col * bw_input * 7 * 7    # This constraint is true for ResNet50 since it has max 7*7 window sized pool layer
    assert (size_VMEM1_kb * 8 * 1024) >= min_VMEM1_bit

    print("min_VMEM1_kb requirement:", min_VMEM1_bit / (8 * 1024))

    OW = LSpec.OW
    OH = LSpec.OH
    OC = LSpec.OC
    KW = LSpec.KW
    KH = LSpec.KH
    Stride = LSpec.Stride
    Batch = LSpec.Batch

    # Zero padding the channel dimension if array column > OC. Besides, Make OC an integer multiple of SysArray_col
    if SysArray_col >= OC:
        OC = SysArray_col
    else:
        if (OC % SysArray_col) != 0:
            rem = OC % SysArray_col
            OC = OC - rem + SysArray_col
    print("OC:", OC)

    # Operand1 goes to VMEM1, Output goes to VMEM2. The tiling of the operand1 need to fit in VMEM1 and output in VMEM2
    # Set OH_tile = OW_tile, these two are first priority dimension, OC is second prority dimension given OC_tile >= SysArray_col, and Batch is the 3rd prority dimension
    # if input fits in VMEM1 then output will also fit in VMEM2 since OW,OH < IW,IH and VMEM1 = VMEM2. Hence, using only input tile size as the condition is enough
    OC_tile = SysArray_col
    Batch_tile = 1
    VMEM1_bit = size_VMEM1_kb * 8 * 1024  # stores input
    OW_tile = OW
    OH_tile = OH
    IW_tile = (OW_tile - 1) * Stride + KW
    IH_tile = (OH_tile - 1) * Stride + KH
    valid_flag = True
    i = 2    
    while ((OC_tile * IW_tile * IH_tile * Batch_tile * bw_input) > VMEM1_bit) or valid_flag == False:
        OW_tile = OW/i
        OH_tile = OH/i
        IW_tile = (OW_tile - 1) * Stride + KW
        IH_tile = (OH_tile - 1) * Stride + KH
        i = i + 1
        valid_flag = True
        if (OW_tile % 1) != 0 or (OH_tile % 1) != 0:
            valid_flag = False
    print ("OW_tile:", OW_tile, "OH_tile:", OH_tile, "IW_tile:", IW_tile, "IH_tile:", IH_tile)

    #Starting from a valid OC_tile, now making it better
    j = 2
    while (OC_tile * IW_tile * IH_tile * Batch_tile * bw_input) < VMEM1_bit:
        OC_tile_new = SysArray_col * j
        if (OC_tile_new * IW_tile * IH_tile * Batch_tile * bw_input) > VMEM1_bit:
            break
        if OC_tile_new > OC:  
            break
        OC_tile = OC_tile_new
        j = j + 1
    print("OC_tile:", OC_tile)

    #Finding batch_tile same way as OW_tile and OH_tile
    Batch_tile = Batch
    valid_flag = True
    k = 2
    while ((OC_tile * IW_tile *IH_tile * Batch_tile * bw_input) > VMEM1_bit) or valid_flag == False:
        Batch_tile = Batch/k
        k = k + 1
        valid_flag = True
        if (Batch_tile % 1) != 0:
            valid_flag = False
    print ("Batch_tile:", Batch_tile)

    # All tiling calculated: placing asserts to ensure the tiles fit in the respective buffer
    assert (IW_tile * IH_tile * OC_tile * Batch_tile * bw_input) <= VMEM1_bit

    # Storing all results in LSpec object
    #Replacing the original OC with Padded OC
    LSpec.OC = OC
    # SRAM to DRAM tiling results are storing in the LSpec object
    LSpec.DTile_ow = OW_tile
    LSpec.DTile_oh = OH_tile
    LSpec.DTile_oc = OC_tile
    LSpec.DTile_kw = KW
    LSpec.DTile_kh = KH
    LSpec.DTile_ic = "None"
    LSpec.DTile_batch = Batch_tile
    LSpec.DTile_iw = IW_tile
    LSpec.DTile_ih = IH_tile

    # DRAM to SRAM Tiling depend on the array size only
    LSpec.Stile_ow = 1
    LSpec.Stile_oh = 1
    LSpec.Stile_oc = SysArray_col
    LSpec.Stile_kw = 1
    LSpec.Stile_kh = 1
    LSpec.Stile_ic = "None"
    LSpec.Stile_batch = 1
    LSpec.Stile_iw = 1
    LSpec.Stile_ih = 1


def generate_tile_conv(LSpec, Hardware_param):
    # Unpacking the parameters
    # all the buffers in the Systolic Array are Double Buffered
    # do the channle padding inside and then pass it as a parameter to override the channel in LSpec object
    bw_ifmap = Hardware_param.SAbw_ifmap
    bw_filter = Hardware_param.SAbw_filter
    bw_psum = Hardware_param.SAbw_psum
    bw_ofmap = Hardware_param.SAbw_ofmap  
    bw_bias = Hardware_param.SAbw_bias 
    
    SysArray_row = Hardware_param.SysArray_row
    SysArray_col = Hardware_param.SysArray_col
    size_WBUF_kb = Hardware_param.Size_WBUF
    size_IBUF_kb = Hardware_param.Size_IBUF
    size_OBUF_kb = Hardware_param.Size_OBUF
    size_BBUF_kb = Hardware_param.Size_BBUF 

    print("size_WBUF_kb:", size_WBUF_kb, "size_IBUF_kb:", size_IBUF_kb, "size_OBUF_kb:", size_OBUF_kb, "size_BBUF_kb:", size_BBUF_kb)

    # Placing the Min size constraints on each buffer
    max_kernel = 3   # using 3 so that maximum 3*3 sized kernels do not need to be split during tiling
    max_stride = 2   # maximum stride in ResNet50 is 2
    min_ow_oh = 7    # using 7 so that OH=OW=7 do not need to get split during tiling
    min_iw_ih = (min_ow_oh - 1) * max_stride + max_kernel

    min_WBUF_bit = 2 * SysArray_col * SysArray_row * max_kernel * max_kernel * bw_filter  #2 for double buffering,     
    min_IBUF_bit = 2 * SysArray_row * min_iw_ih * min_iw_ih * bw_ifmap
    min_OBUF_bit = 2 * SysArray_col * min_ow_oh * min_ow_oh * bw_psum 
    min_BBUF_bit = 2 * SysArray_col * bw_bias

    print("min_WBUF_kb requirement = ", min_WBUF_bit / (8 * 1024))
    print("min_IBUF_kb requirement = ", min_IBUF_bit / (8 * 1024))
    print("min_OBUF_kb requirement = ", min_OBUF_bit / (8 * 1024))
    print("min_BBUF_kb requirement = ", min_BBUF_bit / (8 * 1024))

    assert (size_WBUF_kb * 8 * 1024) >= min_WBUF_bit
    assert (size_IBUF_kb * 8 * 1024) >= min_IBUF_bit
    assert (size_OBUF_kb * 8 * 1024) >= min_OBUF_bit
    assert (size_BBUF_kb * 8 * 1024) >= min_BBUF_bit

    OW = LSpec.OW
    OH = LSpec.OH
    OC = LSpec.OC
    KW = LSpec.KW 
    KH = LSpec.KH 
    IC = LSpec.IC 
    Batch = LSpec.Batch
    Stride = LSpec.Stride

    # Zero padding the channel dimension if array row, column > IC, OC. Besides, Make IC, OC an integer multiple of SysArray_row, SysArray_col
    if SysArray_col >= OC:
        OC = SysArray_col
    else:
        if (OC % SysArray_col) != 0:
            rem = OC % SysArray_col
            OC = OC - rem + SysArray_col
 
    if SysArray_row >= IC:
        IC = SysArray_row
    else:
        if (IC % SysArray_row) != 0:
            rem = IC % SysArray_row
            IC = IC - rem + SysArray_row
    print("OC:", OC, "IC:", IC)

    # KH_tile = KW_tile is the first priority dimension and IC is the 2nd priority dimension to benefit psum accumulation in a weight stationary dataflow
    # Set OH_tile = OW_tile, these two are 3rd priority dimension, and Batch is the 4th priority dimension 
    # OC is the last priority dimension

    WBUFeff_bit = (size_WBUF_kb * 8 * 1024) / 2   # 2 for double buffering
    IBUFeff_bit = (size_IBUF_kb * 8 * 1024) / 2
    OBUFeff_bit = (size_OBUF_kb * 8 * 1024) / 2
    BBUFeff_bit = (size_BBUF_kb * 8 * 1024) / 2

    # Finding KH_tile, KW_tile 
    OC_tile = SysArray_col
    IC_tile = SysArray_row
    Batch_tile = 1    
    KW_tile = KW 
    KH_tile = KH
    i = 2
    # KH_tile and KW_tile sized ifmap need to fit in IBUF as well so that minimum OH_tile = OW_tile = 1
    while ((KW_tile * KH_tile * IC_tile * OC_tile * bw_filter) > WBUFeff_bit) or ((KW_tile * KH_tile * IC_tile * Batch_tile * bw_ifmap) > IBUFeff_bit): 
        KW_tile = math.ceil(KW/i)     #not trying to make kernel size as a integer multiple of kernel tile since the size can be odd
        KH_tile = math.ceil(KH/i)     #due to min WBUF and min IBUF size constraints, this while loop will not be activated when KW,KH <= max_kernel
        i = i + 1
    #print("KW_tile:", KW_tile, "KH_tile:", KH_tile)

    #Starting from a valid IC_tile, now making it better
    j = 2 
    while ((KW_tile * KH_tile * IC_tile * OC_tile * bw_filter) < WBUFeff_bit) and ((KW_tile * KH_tile * IC_tile * Batch_tile * bw_ifmap) < IBUFeff_bit): 
        IC_tile_new = SysArray_row * j
        if (KW_tile * KH_tile * IC_tile_new * OC_tile * bw_filter) > WBUFeff_bit:
            break
        if (KW_tile * KH_tile * IC_tile_new * Batch_tile * bw_ifmap) > IBUFeff_bit:
            break
        if IC_tile_new > IC:
            break
        IC_tile = IC_tile_new 
        j = j + 1
    #print("IC_tile:", IC_tile)

    # Finding OW_tile, OH_tile
    OW_tile = OW
    OH_tile = OH
    IW_tile = (OW_tile - 1) * Stride + KW_tile
    IH_tile = (OH_tile - 1) * Stride + KH_tile
    valid_flag = True
    i = 2 
    while ((OW_tile * OH_tile * OC_tile * Batch_tile * bw_psum) > OBUFeff_bit) or ((IW_tile * IH_tile * IC_tile * Batch_tile * bw_ifmap) > IBUFeff_bit) or valid_flag == False:
        OW_tile = OW/i
        OH_tile = OH/i
        IW_tile = (OW_tile - 1) * Stride + KW_tile
        IH_tile = (OH_tile - 1) * Stride + KH_tile
        i = i + 1
        valid_flag = True
        # making OW, OH an integer multiple of OW_tile, OH_tile if KW, KH <= max_kernel. This will ensure this condition for inference convolutions (except 1st conv)
        if KH <= max_kernel and KW <= max_kernel:
            if (OW_tile % 1) != 0 or (OH_tile % 1) != 0:
                valid_flag = False
        else:
            OW_tile = math.ceil(OW_tile)
            OH_tile = math.ceil(OH_tile)
            IW_tile = (OW_tile - 1) * Stride + KW_tile
            IH_tile = (OH_tile - 1) * Stride + KH_tile
    #print ("OW_tile:", OW_tile, "OH_tile:", OH_tile, "IW_tile:", IW_tile, "IH_tile:", IH_tile)

    #Finding Batch_tile same way as OW_tile and OH_tile
    Batch_tile = Batch
    valid_flag = True
    k = 2
    while ((OW_tile * OH_tile * OC_tile * Batch_tile * bw_psum) > OBUFeff_bit) or ((IW_tile * IH_tile * IC_tile * Batch_tile * bw_ifmap) > IBUFeff_bit) or valid_flag == False:
        Batch_tile = Batch/k
        k = k + 1
        valid_flag = True
        if (Batch_tile % 1) != 0:
            valid_flag = False
    #print ("Batch_tile:", Batch_tile)


    #Starting from a valid OC_tile, now making it better 
    j = 2
    while ((KW_tile * KH_tile * IC_tile * OC_tile * bw_filter) < WBUFeff_bit) and \
            ((OW_tile * OH_tile * OC_tile * Batch_tile * bw_psum) < OBUFeff_bit) and \
            ((OC_tile * bw_bias) < BBUFeff_bit):
        
        OC_tile_new = SysArray_col * j
        if (KW_tile * KH_tile * IC_tile * OC_tile_new * bw_filter) > WBUFeff_bit:
            break
        if (OW_tile * OH_tile * OC_tile_new * Batch_tile * bw_psum) > OBUFeff_bit:
            break
        if (OC_tile_new * bw_bias) > BBUFeff_bit:
            break
        if OC_tile_new > OC:  
            break
        OC_tile = OC_tile_new
        j = j + 1
    #print("OC_tile:", OC_tile)


    # All tiling calculated: placing asserts to ensure all tiles fit in their respective buffer
    assert (KW_tile * KH_tile * IC_tile * OC_tile * bw_filter) <= WBUFeff_bit
    assert (IW_tile * IH_tile * IC_tile * Batch_tile * bw_ifmap) <= IBUFeff_bit
    assert (OW_tile * OH_tile * OC_tile * Batch_tile * bw_psum) <= OBUFeff_bit
    assert (OC_tile * bw_bias) <= BBUFeff_bit

    # Storing all results in LSpec object
    #Replacing the original OC, IC with Padded OC, IC
    LSpec.OC = OC 
    LSpec.IC = IC
    # SRAM to DRAM tiling results are storing in the LSpec object
    LSpec.DTile_ow = OW_tile
    LSpec.DTile_oh = OH_tile
    LSpec.DTile_oc = OC_tile
    LSpec.DTile_kw = KW_tile
    LSpec.DTile_kh = KH_tile
    LSpec.DTile_ic = IC_tile
    LSpec.DTile_batch = Batch_tile
    LSpec.DTile_iw = IW_tile
    LSpec.DTile_ih = IH_tile

    # DRAM to SRAM Tiling depend on the array size only
    LSpec.Stile_ow = 1
    LSpec.Stile_oh = 1
    LSpec.Stile_oc = SysArray_col
    LSpec.Stile_kw = 1
    LSpec.Stile_kh = 1
    LSpec.Stile_ic = SysArray_row
    LSpec.Stile_batch = 1
    LSpec.Stile_iw = 1
    LSpec.Stile_ih = 1


def generate_tile_gemm(LSpec, Hardware_param):
    # The tiling generation process for gemm is a subset of conv
    generate_tile_conv(LSpec, Hardware_param)
    


############################### Tiling generation for the additional training operations #############################
def generate_tile_mean_istd(LSpec, Hardware_param):
    # Unpacking the parameters
    # SIMD layers are single buffered
    # do the channle padding inside and then pass it as a parameter to override the channel in LSpec object
    bw_input = Hardware_param.SAbw_psum
    bw_output = Hardware_param.SAbw_psum
    SysArray_col = Hardware_param.SysArray_col
    size_VMEM1_kb = Hardware_param.Size_VMEM1
    size_VMEM2_kb = Hardware_param.Size_VMEM2
    print("size_VMEM1_kb:", size_VMEM1_kb)

    # data goes to VMEM1, temp goes to VMEM1, mean and istd goes to VMEM2
    min_VMEM1_bit = SysArray_col * bw_input +  SysArray_col * bw_input # two terms since data and temp both go to VMEM1
    min_VMEM2_bit = 2 * SysArray_col * bw_output   # 2 since mean and istd both go to VMEM2
    assert (size_VMEM1_kb * 8 * 1024) >= min_VMEM1_bit
    assert (size_VMEM2_kb * 8 * 1024) >= min_VMEM2_bit

    print("min_VMEM1_kb requirement:", min_VMEM1_bit / (8 * 1024))
    print("min_VMEM2_kb requirement:", min_VMEM2_bit / (8 * 1024))
        
    IW = LSpec.IW
    IH = LSpec.IH
    OC = LSpec.OC
    Batch = LSpec.Batch

    # Zero padding the channel dimension if array column > OC. Besides, Make OC an integer multiple of SysArray_col
    if SysArray_col >= OC:
        OC = SysArray_col
    else:
        if (OC % SysArray_col) != 0:
            rem = OC % SysArray_col
            OC = OC - rem + SysArray_col
    print("OC:", OC)

    # IW_tile = IH_tile is the first priority dimension; Batch_tile is the second priority dimension; OC is the third priority dimension
    # for Temp, SysArray_col * bw_input amount of space is allocated in VMEM1, this size for temp is fixed regardless of tile sizes

    VMEM1eff_bit = (size_VMEM1_kb * 8 * 1024) - SysArray_col * bw_input  # avilable VMEM1 size to store data
    VMEM2eff_bit = (size_VMEM2_kb * 8 * 1024) / 2    # mean and istd both are stored in VMEM2. Hence, for each of them half of VMEM2 is available

    # data goes to VMEM1. Hence, the tiling of data need to fit in VMEM1 effective. dimension of data is: IW * IH * Batch * OC
    OC_tile = SysArray_col
    Batch_tile = 1
    IW_tile = IW
    IH_tile = IH
    valid_flag = True
    i = 2
    while ((OC_tile * IW_tile * IH_tile * Batch_tile * bw_input) > VMEM1eff_bit) or valid_flag == False:
        IW_tile = IW/i
        IH_tile = IH/i
        i = i + 1
        valid_flag = True
        if (IW_tile % 1) != 0 or (IH_tile % 1) != 0:
            valid_flag = False
    print ("IW_tile:", IW_tile, "IH_tile:", IH_tile)

    #Finding batch_tile same way as IW_tile and IH_tile
    Batch_tile = Batch
    valid_flag = True
    k = 2
    while ((OC_tile * IW_tile * IH_tile * Batch_tile * bw_input) > VMEM1eff_bit) or valid_flag == False:
        Batch_tile = Batch/k
        k = k + 1
        valid_flag = True
        if (Batch_tile % 1) != 0:
            valid_flag = False
    print ("Batch_tile:", Batch_tile)

    #Starting from a valid OC_tile, now making it better 
    j = 2
    while ((OC_tile * IW_tile * IH_tile * Batch_tile * bw_input) < VMEM1eff_bit) and ((OC_tile * bw_output) < VMEM2eff_bit):        
        OC_tile_new = SysArray_col * j
        if (OC_tile_new * IW_tile * IH_tile * Batch_tile * bw_input) > VMEM1eff_bit:
            break
        if (OC_tile_new * bw_output) > VMEM2eff_bit:
            break
        if OC_tile_new > OC:  
            break
        OC_tile = OC_tile_new
        j = j + 1
    print("OC_tile:", OC_tile)

    # All tiling calculated: placing asserts to ensure the tiles fit in the respective buffer
    assert (IW_tile * IH_tile * OC_tile * Batch_tile * bw_input) <= VMEM1eff_bit
    assert (OC_tile * bw_output) <= VMEM2eff_bit

    # Storing all results in LSpec object
    #Replacing the original OC with Padded OC
    LSpec.OC = OC
    # SRAM to DRAM tiling results are storing in the LSpec object
    LSpec.DTile_iw = IW_tile
    LSpec.DTile_ih = IH_tile
    LSpec.DTile_oc = OC_tile
    LSpec.DTile_batch = Batch_tile

    # DRAM to SRAM Tiling depend on the array size only
    LSpec.Stile_iw = 1
    LSpec.Stile_ih = 1
    LSpec.Stile_oc = SysArray_col
    LSpec.Stile_batch = 1


def generate_tile_batchnorm_forward(LSpec, Hardware_param):
    # Unpacking the parameters
    # SIMD layers are single buffered
    # do the channle padding inside and then pass it as a parameter to override the channel in LSpec object
    bw_input = Hardware_param.SAbw_psum
    #bw_output = Hardware_param.SAbw_psum
    SysArray_col = Hardware_param.SysArray_col
    size_VMEM1_kb = Hardware_param.Size_VMEM1
    size_VMEM2_kb = Hardware_param.Size_VMEM2
    print("size_VMEM1_kb:", size_VMEM1_kb)

    # data/output goes to VMEM1, mean, istd, scale, and offset go to VMEM2
    min_VMEM1_bit = SysArray_col * bw_input 
    min_VMEM2_bit = 4 * SysArray_col * bw_input   # 4 since four items are stored in VMEM2
    assert (size_VMEM1_kb * 8 * 1024) >= min_VMEM1_bit
    assert (size_VMEM2_kb * 8 * 1024) >= min_VMEM2_bit

    print("min_VMEM1_kb requirement:", min_VMEM1_bit / (8 * 1024))
    print("min_VMEM2_kb requirement:", min_VMEM2_bit / (8 * 1024))

    IW = LSpec.IW
    IH = LSpec.IH
    OC = LSpec.OC
    Batch = LSpec.Batch

    # Zero padding the channel dimension if array column > OC. Besides, Make OC an integer multiple of SysArray_col
    if SysArray_col >= OC:
        OC = SysArray_col
    else:
        if (OC % SysArray_col) != 0:
            rem = OC % SysArray_col
            OC = OC - rem + SysArray_col
    print("OC:", OC)

    # IW_tile = IH_tile is the first priority dimension; Batch_tile is the second priority dimension; OC is the third priority dimension
    # No Temp location needed

    VMEM1eff_bit = size_VMEM1_kb * 8 * 1024          # VMEM1 stores data/output
    VMEM2eff_bit = (size_VMEM2_kb * 8 * 1024) / 4    # mean, istd, scale, offset all are stored in VMEM2. Hence, for each of them 1/4 of VMEM2 is available

    # data goes to VMEM1. Hence, the tiling of data need to fit in VMEM1 effective. dimension of data is: IW * IH * Batch * OC
    OC_tile = SysArray_col
    Batch_tile = 1
    IW_tile = IW
    IH_tile = IH
    valid_flag = True
    i = 2
    while ((OC_tile * IW_tile * IH_tile * Batch_tile * bw_input) > VMEM1eff_bit) or valid_flag == False:
        IW_tile = IW/i
        IH_tile = IH/i
        i = i + 1
        valid_flag = True
        if (IW_tile % 1) != 0 or (IH_tile % 1) != 0:
            valid_flag = False
    print ("IW_tile:", IW_tile, "IH_tile:", IH_tile)

    #Finding batch_tile same way as IW_tile and IH_tile
    Batch_tile = Batch
    valid_flag = True
    k = 2
    while ((OC_tile * IW_tile * IH_tile * Batch_tile * bw_input) > VMEM1eff_bit) or valid_flag == False:
        Batch_tile = Batch/k
        k = k + 1
        valid_flag = True
        if (Batch_tile % 1) != 0:
            valid_flag = False
    print ("Batch_tile:", Batch_tile)

    #Starting from a valid OC_tile, now making it better 
    j = 2
    while ((OC_tile * IW_tile * IH_tile * Batch_tile * bw_input) < VMEM1eff_bit) and ((OC_tile * bw_input) < VMEM2eff_bit):        
        OC_tile_new = SysArray_col * j
        if (OC_tile_new * IW_tile * IH_tile * Batch_tile * bw_input) > VMEM1eff_bit:
            break
        if (OC_tile_new * bw_input) > VMEM2eff_bit:
            break
        if OC_tile_new > OC:  
            break
        OC_tile = OC_tile_new
        j = j + 1
    print("OC_tile:", OC_tile)

    # All tiling calculated: placing asserts to ensure the tiles fit in the respective buffer
    assert (IW_tile * IH_tile * OC_tile * Batch_tile * bw_input) <= VMEM1eff_bit
    assert (OC_tile * bw_input) <= VMEM2eff_bit

    # Storing all results in LSpec object
    #Replacing the original OC with Padded OC
    LSpec.OC = OC
    # SRAM to DRAM tiling results are storing in the LSpec object
    LSpec.DTile_iw = IW_tile
    LSpec.DTile_ih = IH_tile
    LSpec.DTile_oc = OC_tile
    LSpec.DTile_batch = Batch_tile

    # DRAM to SRAM Tiling depend on the array size only
    LSpec.Stile_iw = 1
    LSpec.Stile_ih = 1
    LSpec.Stile_oc = SysArray_col
    LSpec.Stile_batch = 1


def generate_tile_batchnorm_backward(LSpec, Hardware_param):
    # Unpacking the parameters
    # SIMD layers are single buffered
    # do the channle padding inside and then pass it as a parameter to override the channel in LSpec object
    bw_input = Hardware_param.SAbw_psum
    #bw_output = Hardware_param.SAbw_psum
    SysArray_col = Hardware_param.SysArray_col
    size_VMEM1_kb = Hardware_param.Size_VMEM1
    size_VMEM2_kb = Hardware_param.Size_VMEM2
    print("size_VMEM1_kb:", size_VMEM1_kb)

    # data/data_hat, temp, offset_grad, and scale are stored in VMEM1
    # input grad/output data_grad, mean, istd, and scale_grad are stored in VMEM2
    min_VMEM1_bit = SysArray_col * bw_input +  3 * SysArray_col * bw_input  #since one 4D data and three more 1D items are stored in each VMEM
    min_VMEM2_bit = SysArray_col * bw_input +  3 * SysArray_col * bw_input    
    assert (size_VMEM1_kb * 8 * 1024) >= min_VMEM1_bit
    assert (size_VMEM2_kb * 8 * 1024) >= min_VMEM2_bit

    print("min_VMEM1_kb requirement:", min_VMEM1_bit / (8 * 1024))
    print("min_VMEM2_kb requirement:", min_VMEM2_bit / (8 * 1024))

    IW = LSpec.IW
    IH = LSpec.IH
    OC = LSpec.OC
    Batch = LSpec.Batch

    # Zero padding the channel dimension if array column > OC. Besides, Make OC an integer multiple of SysArray_col
    if SysArray_col >= OC:
        OC = SysArray_col
    else:
        if (OC % SysArray_col) != 0:
            rem = OC % SysArray_col
            OC = OC - rem + SysArray_col
    print("OC:", OC)

    # IW_tile = IH_tile is the first priority dimension; Batch_tile is the second priority dimension; OC is the third priority dimension
    # for Temp, SysArray_col * bw_input amount of space is allocated in VMEM1, this size for temp is fixed regardless of tile sizes

    # Since min OC_tile = SysArray_col, any valid OC_tile will work for Temp.
    # VMEM2 stores one 4D grad_tile and three 1D_tiles for mean, istd, and scale_grad.
    # VMEM1 stores one 4D data_tile and three 1D_tiles for temp, scale, and offset_grad.
    # since temp <= an 1D tile for mean, istd etc, I can use VMEM2 only to find the tiling parameters and that will work for VMEM1 too.

    VMEM2eff_bit = (size_VMEM2_kb * 8 * 1024) # avilable VMEM2 size to store grad + mean + istd + scale_grad

    # Finding tiling for height and width dimensions
    OC_tile = SysArray_col
    Batch_tile = 1
    IW_tile = IW
    IH_tile = IH
    valid_flag = True
    i = 2
    while (((OC_tile * IW_tile * IH_tile * Batch_tile * bw_input) + (3 * OC_tile * bw_input)) > VMEM2eff_bit) or valid_flag == False:
        IW_tile = IW/i
        IH_tile = IH/i
        i = i + 1
        valid_flag = True
        if (IW_tile % 1) != 0 or (IH_tile % 1) != 0:
            valid_flag = False
    print ("IW_tile:", IW_tile, "IH_tile:", IH_tile)

    #Finding batch_tile same way as IW_tile and IH_tile
    Batch_tile = Batch
    valid_flag = True
    k = 2
    while (((OC_tile * IW_tile * IH_tile * Batch_tile * bw_input) + (3 * OC_tile * bw_input)) > VMEM2eff_bit) or valid_flag == False:
        Batch_tile = Batch/k
        k = k + 1
        valid_flag = True
        if (Batch_tile % 1) != 0:
            valid_flag = False
    print ("Batch_tile:", Batch_tile)

    #Starting from a valid OC_tile, now making it better 
    j = 2
    while ((OC_tile * IW_tile * IH_tile * Batch_tile * bw_input) + (3 * OC_tile * bw_input)) < VMEM2eff_bit:        
        OC_tile_new = SysArray_col * j
        if ((OC_tile_new * IW_tile * IH_tile * Batch_tile * bw_input) + (3 * OC_tile_new * bw_input)) > VMEM2eff_bit:
            break
        if OC_tile_new > OC:  
            break
        OC_tile = OC_tile_new
        j = j + 1
    print("OC_tile:", OC_tile)

    # All tiling calculated: placing asserts to ensure the tiles fit in the respective buffer
    assert (IW_tile * IH_tile * OC_tile * Batch_tile * bw_input + 3 * OC_tile * bw_input) <= VMEM2eff_bit

    # Storing all results in LSpec object
    #Replacing the original OC with Padded OC
    LSpec.OC = OC
    # SRAM to DRAM tiling results are storing in the LSpec object
    LSpec.DTile_iw = IW_tile
    LSpec.DTile_ih = IH_tile
    LSpec.DTile_oc = OC_tile
    LSpec.DTile_batch = Batch_tile

    # DRAM to SRAM Tiling depend on the array size only
    LSpec.Stile_iw = 1
    LSpec.Stile_ih = 1
    LSpec.Stile_oc = SysArray_col
    LSpec.Stile_batch = 1


def generate_tile_comnSIMDback(Hardware_param, Batch_override, CompOut_layer):
    # This is a common function to generate tiling for the common SIMD operations
    LSpec = LayerSpecTemplate()

    layer_op = CompOut_layer['operation']
    #print("operation_name:", layer_op)

    # this is to avoid layer name spelling mistake inside the code of this function
    assert (layer_op == "sgd4d" or layer_op == "sgd2d" or layer_op == "sgd1d" or layer_op == "relu_grad" or layer_op == "reduce_sum")

    ############ extracting the layer spec.
    if layer_op == "relu_grad" or layer_op == "sgd4d":
        LSpec.OW = CompOut_layer['iterable_dimensions']['W']
        LSpec.OH = CompOut_layer['iterable_dimensions']['H']
        LSpec.OC = CompOut_layer['iterable_dimensions']['C']
        LSpec.Batch = CompOut_layer['iterable_dimensions']['N']
        LSpec.Stride = 1
    elif layer_op == "sgd2d" or layer_op == "reduce_sum":
        LSpec.OW = 1
        LSpec.OH = 1
        LSpec.OC = CompOut_layer['iterable_dimensions']['C']
        LSpec.Batch = CompOut_layer['iterable_dimensions']['N']
        LSpec.Stride = 1
    elif layer_op == "sgd1d":
        LSpec.OW = 1
        LSpec.OH = 1
        LSpec.OC = CompOut_layer['iterable_dimensions']['C']
        LSpec.Batch = 1
        LSpec.Stride = 1

    ############## doing batch override: N is not batch dimension in sgd4d and sgd2d. Hence not doing batch override for them
    if layer_op == "relu_grad" or layer_op == "reduce_sum":
        LSpec.Batch = Batch_override


    ############## Tiling generation
    #print("LSpec before tiling generation process:", vars(LSpec))
    if layer_op == "relu_grad" or layer_op == "sgd4d":
        # 4D data of dimension (H*W*C*N). Op1 need to fit in VMEM1, op2 need to fit in VMEM2, output need to fit in either VMEM1 or VMEM2. 
        # The tile sizes for op1, op2, and output are same. So basically the tiling generation process is exactly same as relu/elem_add
        generate_tile_relu(LSpec, Hardware_param)
    elif layer_op == "sgd2d" or layer_op == "reduce_sum":
        # 2D data of dimension (N*C). Hence, inside LSpec have set the OH, OW dimensions to be 1
        # for sgd2d: Op1 need to fit in VMEM1, op2 need to fit in VMEM2, output need to fit in either VMEM1 or VMEM2. 
        # for reduce_sum: Op1 need to fit in VMEM1, output need to fit in VMEM2. Output is 1D (volume less than op1). So, using Op1 to determine the tiling is enough
        # So with OH=OW=1, the tiling generation process is exactly same as relu
        generate_tile_relu(LSpec, Hardware_param)
    elif layer_op == "sgd1d":
        # 1D data of dimension C. Hence, inside LSpec have set the OH, OW, and Batch dimensions to be 1
        # Op1 need to fit in VMEM1, output need to fit in VMEM2. The tile sizes for op1 and output are same.
        # So with Batch = 1, OH=OW=1, the tiling generation process is exactly same as relu
        generate_tile_relu(LSpec, Hardware_param)
    print("LSpec after tiling generation process:", vars(LSpec))

    
    ################## Creating the DNN spec format
    if layer_op == "relu_grad" or layer_op == "sgd4d":
        CompilerOut_layer = copy.deepcopy(CompOut_layer)

        CompilerOut_layer['iterable_dimensions'] = {'N': LSpec.Batch, 'C': LSpec.OC, 'H': LSpec.OH, 'W': LSpec.OW}
        #print("iterable_dimensions:", CompilerOut_layer['iterable_dimensions'])

        #print("Input operand 1")
        CompilerOut_layer['inputs'][0]['shape_symbols'] = {'N': LSpec.Batch, 'H': LSpec.OH, 'W': LSpec.OW, 'C': LSpec.OC}
        #print("shape_symbols:" ,CompilerOut_layer['inputs'][0]['shape_symbols'])

        
        CompilerOut_layer['inputs'][0]['tiling']= {'DRAM': {'N': LSpec.Batch, 'H': LSpec.OH, 'W': LSpec.OW, 'C': LSpec.OC}, \
                                                   'VMEM1': {'N': LSpec.DTile_batch, 'H': LSpec.DTile_oh, 'W': LSpec.DTile_ow, 'C': LSpec.DTile_oc}, \
                                                   'SIMD': {'N': LSpec.Stile_batch, 'H': LSpec.Stile_oh, 'W': LSpec.Stile_ow, 'C': LSpec.Stile_oc}}
        #print("tiling:", CompilerOut_layer['inputs'][0]['tiling'])

        #print("Input operand 2")
        CompilerOut_layer['inputs'][1]['shape_symbols'] = {'N': LSpec.Batch, 'H': LSpec.OH, 'W': LSpec.OW, 'C': LSpec.OC}
        #print("shape_symbols:" ,CompilerOut_layer['inputs'][1]['shape_symbols'])

        
        CompilerOut_layer['inputs'][1]['tiling']= {'DRAM': {'N': LSpec.Batch, 'H': LSpec.OH, 'W': LSpec.OW, 'C': LSpec.OC}, \
                                                   'VMEM2': {'N': LSpec.DTile_batch, 'H': LSpec.DTile_oh, 'W': LSpec.DTile_ow, 'C': LSpec.DTile_oc}, \
                                                   'SIMD': {'N': LSpec.Stile_batch, 'H': LSpec.Stile_oh, 'W': LSpec.Stile_ow, 'C': LSpec.Stile_oc}}
        #print("tiling:", CompilerOut_layer['inputs'][1]['tiling'])


        print("Output")
        CompilerOut_layer['outputs'][0]['shape_symbols'] = {'N': LSpec.Batch, 'H': LSpec.OH, 'W': LSpec.OW, 'C': LSpec.OC}
        #print("shape_symbols:" ,CompilerOut_layer['outputs'][0]['shape_symbols'])

        
        CompilerOut_layer['outputs'][0]['tiling']= {'DRAM': {'N': LSpec.Batch, 'H': LSpec.OH, 'W': LSpec.OW, 'C': LSpec.OC}, \
                                                   'VMEM1': {'N': LSpec.DTile_batch, 'H': LSpec.DTile_oh, 'W': LSpec.DTile_ow, 'C': LSpec.DTile_oc}, \
                                                   'SIMD': {'N': LSpec.Stile_batch, 'H': LSpec.Stile_oh, 'W': LSpec.Stile_ow, 'C': LSpec.Stile_oc}}
        print("tiling:", CompilerOut_layer['outputs'][0]['tiling'])

    
    elif layer_op == "sgd2d" or layer_op == "reduce_sum":
        CompilerOut_layer = copy.deepcopy(CompOut_layer)

        CompilerOut_layer['iterable_dimensions'] = {'N': LSpec.Batch, 'C': LSpec.OC}
        #print("iterable_dimensions:", CompilerOut_layer['iterable_dimensions'])

        #print("Input operand 1")
        CompilerOut_layer['inputs'][0]['shape_symbols'] = {'N': LSpec.Batch, 'C': LSpec.OC}
        #print("shape_symbols:" ,CompilerOut_layer['inputs'][0]['shape_symbols'])

       
        CompilerOut_layer['inputs'][0]['tiling']= {'DRAM': {'N': LSpec.Batch, 'C': LSpec.OC}, \
                                                   'VMEM1': {'N': LSpec.DTile_batch, 'C': LSpec.DTile_oc}, \
                                                   'SIMD': {'N': LSpec.Stile_batch, 'C': LSpec.Stile_oc}}
        #print("tiling:", CompilerOut_layer['inputs'][0]['tiling'])

        if layer_op == "sgd2d":
            #print("Input operand 2")
            CompilerOut_layer['inputs'][1]['shape_symbols'] = {'N': LSpec.Batch, 'C': LSpec.OC}
            #print("shape_symbols:" ,CompilerOut_layer['inputs'][1]['shape_symbols'])

           
            CompilerOut_layer['inputs'][1]['tiling']= {'DRAM': {'N': LSpec.Batch, 'C': LSpec.OC}, \
                                                       'VMEM2': {'N': LSpec.DTile_batch, 'C': LSpec.DTile_oc}, \
                                                       'SIMD': {'N': LSpec.Stile_batch, 'C': LSpec.Stile_oc}}
            #print("tiling:", CompilerOut_layer['inputs'][1]['tiling'])

            print("Output")
            CompilerOut_layer['outputs'][0]['shape_symbols'] = {'N': LSpec.Batch, 'C': LSpec.OC}
            #print("shape_symbols:" ,CompilerOut_layer['outputs'][0]['shape_symbols'])

           
            CompilerOut_layer['outputs'][0]['tiling']= {'DRAM': {'N': LSpec.Batch, 'C': LSpec.OC}, \
                                                       'VMEM1': {'N': LSpec.DTile_batch, 'C': LSpec.DTile_oc}, \
                                                       'SIMD': {'N': LSpec.Stile_batch, 'C': LSpec.Stile_oc}}
            print("tiling:", CompilerOut_layer['outputs'][0]['tiling'])

        elif layer_op == "reduce_sum":  # has only one input and output has only one dimension
            print("Output")
            CompilerOut_layer['outputs'][0]['shape_symbols'] = {'C': LSpec.OC}
            #print("shape_symbols:" ,CompilerOut_layer['outputs'][0]['shape_symbols'])

           
            CompilerOut_layer['outputs'][0]['tiling']= {'DRAM': {'C': LSpec.OC}, \
                                                       'VMEM2': {'C': LSpec.DTile_oc}, \
                                                       'SIMD': {'C': LSpec.Stile_oc}}
            print("tiling:", CompilerOut_layer['outputs'][0]['tiling'])

    elif layer_op == "sgd1d":
        CompilerOut_layer = copy.deepcopy(CompOut_layer)

        CompilerOut_layer['iterable_dimensions'] = {'C': LSpec.OC}
        #print("iterable_dimensions:", CompilerOut_layer['iterable_dimensions'])

        #print("Input operand 1")
        CompilerOut_layer['inputs'][0]['shape_symbols'] = {'C': LSpec.OC}
        #print("shape_symbols:" ,CompilerOut_layer['inputs'][0]['shape_symbols'])
        
        CompilerOut_layer['inputs'][0]['tiling']= {'DRAM': {'C': LSpec.OC}, 'VMEM1': {'C': LSpec.DTile_oc}, 'SIMD': {'C': LSpec.Stile_oc}}
        #print("tiling:", CompilerOut_layer['inputs'][0]['tiling'])

        #print("Input operand 2")
        CompilerOut_layer['inputs'][1]['shape_symbols'] = {'C': LSpec.OC}
        #print("shape_symbols:" ,CompilerOut_layer['inputs'][1]['shape_symbols'])
        
        CompilerOut_layer['inputs'][1]['tiling']= {'DRAM': {'C': LSpec.OC}, 'VMEM2': {'C': LSpec.DTile_oc}, 'SIMD': {'C': LSpec.Stile_oc}}
        #print("tiling:", CompilerOut_layer['inputs'][1]['tiling'])

        print("Output")
        CompilerOut_layer['outputs'][0]['shape_symbols'] = {'C': LSpec.OC}
        #print("shape_symbols:" ,CompilerOut_layer['outputs'][0]['shape_symbols'])
        
        CompilerOut_layer['outputs'][0]['tiling']= {'DRAM': {'C': LSpec.OC}, 'VMEM1': {'C': LSpec.DTile_oc}, 'SIMD': {'C': LSpec.Stile_oc}}
        print("tiling:", CompilerOut_layer['outputs'][0]['tiling'])

 
    #with open("CompilerOut_layer.json", "w") as f:
    #    json.dump(CompilerOut_layer, f, indent=4)

    return CompilerOut_layer
    
    

def generate_tile_maxpool_grad(LSpec, Hardware_param):
    # Unpacking the parameters
    # SIMD layers are single buffered
    # do the channle padding inside and then pass it as a parameter to override the channel in LSpec object
    bw_input = Hardware_param.SAbw_psum
    SysArray_col = Hardware_param.SysArray_col
    size_VMEM1_kb = Hardware_param.Size_VMEM1
    print("size_VMEM1_kb:", size_VMEM1_kb)

    # VMEM1 stores grad and output; VMEM2 stores temp and index
    # The tiling of grad is same as index. The requirement for temp is (SysArray_col * bw_input) regardless of tiling
    # minimum tile size of grad is (SysArray_col * bw_input)
    # minimum tile size of output is (SysArray_col * bw_input * Kh * Kw). Sinec this is >= temp size, Can use only VMEM1 to find the tiling.
    # This will gurantee that (index_tile + temp) also fits in VMEM2 since VMEM1=VMEM2

    min_VMEM1_bit = (SysArray_col * bw_input * 7 * 7) + SysArray_col * bw_input     # first term for output, 2nd term for grad
                                                                                    # This constraint is true for ResNet50 since it has max 7*7 window sized pool layer    
    assert (size_VMEM1_kb * 8 * 1024) >= min_VMEM1_bit
    print("min_VMEM1_kb requirement:", min_VMEM1_bit / (8 * 1024))

    OW = LSpec.OW
    OH = LSpec.OH
    OC = LSpec.OC
    KW = LSpec.KW
    KH = LSpec.KH
    Stride = LSpec.Stride
    Batch = LSpec.Batch

    # Zero padding the channel dimension if array column > OC. Besides, Make OC an integer multiple of SysArray_col
    if SysArray_col >= OC:
        OC = SysArray_col
    else:
        if (OC % SysArray_col) != 0:
            rem = OC % SysArray_col
            OC = OC - rem + SysArray_col
    print("OC:", OC)

    # (grad_tile + output_tile) need to fit in VMEM1. This will also ensure that (index_tile + temp) fit in VMEM2
    # Set OH_tile = OW_tile, these two are first priority dimension, OC is second prority dimension given OC_tile >= SysArray_col, and Batch is the 3rd prority dimension    
    OC_tile = SysArray_col
    Batch_tile = 1
    VMEM1_bit = size_VMEM1_kb * 8 * 1024  # stores grad+output
    OW_tile = OW
    OH_tile = OH
    IW_tile = (OW_tile - 1) * Stride + KW
    IH_tile = (OH_tile - 1) * Stride + KH
    valid_flag = True
    i = 2   
    while (((OC_tile * IW_tile * IH_tile * Batch_tile * bw_input) + (OC_tile * OW_tile * OH_tile * Batch_tile * bw_input)) > VMEM1_bit) or valid_flag == False:
        OW_tile = OW/i
        OH_tile = OH/i
        IW_tile = (OW_tile - 1) * Stride + KW
        IH_tile = (OH_tile - 1) * Stride + KH
        i = i + 1
        valid_flag = True
        if (OW_tile % 1) != 0 or (OH_tile % 1) != 0:
            valid_flag = False
    print ("OW_tile:", OW_tile, "OH_tile:", OH_tile, "IW_tile:", IW_tile, "IH_tile:", IH_tile)

    #Starting from a valid OC_tile, now making it better
    j = 2
    while ((OC_tile * IW_tile * IH_tile * Batch_tile * bw_input) + (OC_tile * OW_tile * OH_tile * Batch_tile * bw_input)) < VMEM1_bit:
        OC_tile_new = SysArray_col * j        
        if ((OC_tile_new * IW_tile * IH_tile * Batch_tile * bw_input) + (OC_tile_new * OW_tile * OH_tile * Batch_tile * bw_input)) > VMEM1_bit:
            break
        if OC_tile_new > OC:  
            break
        OC_tile = OC_tile_new
        j = j + 1
    print("OC_tile:", OC_tile)

    #Finding batch_tile same way as OW_tile and OH_tile
    Batch_tile = Batch
    valid_flag = True
    k = 2
    while (((OC_tile * IW_tile * IH_tile * Batch_tile * bw_input) + (OC_tile * OW_tile * OH_tile * Batch_tile * bw_input)) > VMEM1_bit) or valid_flag == False:
        Batch_tile = Batch/k
        k = k + 1
        valid_flag = True
        if (Batch_tile % 1) != 0:
            valid_flag = False
    print ("Batch_tile:", Batch_tile)

    # All tiling calculated: placing asserts to ensure the tiles fit in the respective buffer
    assert ((IW_tile * IH_tile * OC_tile * Batch_tile * bw_input) + (OW_tile * OH_tile * OC_tile * Batch_tile * bw_input)) <= VMEM1_bit

    # Storing all results in LSpec object
    #Replacing the original OC with Padded OC
    LSpec.OC = OC
    # SRAM to DRAM tiling results are storing in the LSpec object
    LSpec.DTile_ow = OW_tile
    LSpec.DTile_oh = OH_tile
    LSpec.DTile_oc = OC_tile
    LSpec.DTile_kw = KW
    LSpec.DTile_kh = KH
    LSpec.DTile_ic = "None"
    LSpec.DTile_batch = Batch_tile
    LSpec.DTile_iw = IW_tile
    LSpec.DTile_ih = IH_tile

    # DRAM to SRAM Tiling depend on the array size only
    LSpec.Stile_ow = 1
    LSpec.Stile_oh = 1
    LSpec.Stile_oc = SysArray_col
    LSpec.Stile_kw = 1
    LSpec.Stile_kh = 1
    LSpec.Stile_ic = "None"
    LSpec.Stile_batch = 1
    LSpec.Stile_iw = 1
    LSpec.Stile_ih = 1


def generate_tile_avgpool_grad(LSpec, Hardware_param):
    # VMEM1 stores the output: dimension (IW * IH * OC * Batch)
    # VMEM2 stores the grad and temp. dimension of grad is (OW * OH * OC * Batch)
    # The requirement for temp is (SysArray_col * bw_input) regardless of tiling
    # size of (grad_tile + temp) is always <= size of output tile. Hence to find tiling, output need to fit in VMEM1. 
    # This will also ensure that (grad_tile + temp) fits in VMEM2
    # Therefore, the tiling generation process is exactly same as pool_forward
    generate_tile_pool(LSpec, Hardware_param)

    # Placing some asserts to double sure all tiles fit in their respective buffers
    bw_input = Hardware_param.SAbw_psum
    SysArray_col = Hardware_param.SysArray_col
    size_VMEM1_kb = Hardware_param.Size_VMEM1
    size_VMEM2_kb = Hardware_param.Size_VMEM2

    assert (LSpec.DTile_iw * LSpec.DTile_ih * LSpec.DTile_oc * LSpec.DTile_batch * bw_input) <= (size_VMEM1_kb * 8 * 1024)
    assert ((LSpec.DTile_ow * LSpec.DTile_oh * LSpec.DTile_oc * LSpec.DTile_batch * bw_input) + (SysArray_col * bw_input)) <= (size_VMEM2_kb * 8 * 1024)































'''
def discarded code():
    ########## for relu layer
    j = 2
    while (OC_tile * OW_tile * OH_tile * bw_input) < VMEM1_bit:
        if (OC_tile * j * OW_tile * OH_tile * bw_input) >= VMEM1_bit:
            break
        if OC_tile >= OC:  # This condition works since OC is a integer multiple of SysArray_col
            break
        OC_tile = OC_tile * j 
        j = j + 1
    print("OC_tile:", OC_tile)
    
    ######### for pooling layer   
    valid_flag = True
    if (OC_tile * IW_tile * IH_tile * Batch_tile * bw_input) > VMEM1_bit:  #this condition checks if full OH & OW fit in VMEM2, whether full IH & IW also fit in VMEM1 or not
        valid_flag = False

    i = 2
    while ((OC_tile * OW_tile * OH_tile * Batch_tile * bw_output) > VMEM2_bit) or valid_flag == False:
        OW_tile = OW/i
        OH_tile = OH/i
        IW_tile = (OW_tile - 1) * Stride + KW
        IH_tile = (OH_tile - 1) * Stride + KH
        i = i + 1
        valid_flag = True
        if (OW_tile % 1) != 0 or (OH_tile % 1) != 0:
            valid_flag = False
        if (OC_tile * IW_tile * IH_tile * Batch_tile * bw_input) > VMEM1_bit:
            valid_flag = False
'''   






    















    






