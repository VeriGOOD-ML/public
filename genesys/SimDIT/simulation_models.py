import logging
import math
import numpy as np
from data_objects import HardwareObject, SAResult_Inflayer, SIMDResult_Inflayer
from layer_object import LayerObject
from sysarray_models import conv_access_model, conv_cycle_model, gemm_access_model, gemm_cycle_model
from simd_models import relu_access_model, relu_cycle_model, elemadd_access_model, elemadd_cycle_model, pool_access_model, pool_cycle_model
from simd_models import softmax_access_model, softmax_cycle_model, roialignpool_access_model, roialignpool_cycle_model
from train_models import common_SIMD_backward_model, batch_norm_forward_estimate, batch_norm_backward_estimate
from train_models import mean_istd_model, batch_norm_forward_model, batch_norm_backward_model, pooling_backward_model


def models_inference_high(Hardware_param, LayerObj):
    #This function is mainly creating the hierarchy for the simulator

    SysResult_inflayer = SAResult_Inflayer()      # this result object will store all the results for the systolic array hardware
    SIMDResult_inflayer = SIMDResult_Inflayer()   # this result object will store all the results for the SIMD array hardware

    #unpacking the necessary variables
    Layer_name = LayerObj.Layer_name
    Exe_Hardware = LayerObj.Exe_Hardware

    Buffering_scheme_SA = Hardware_param.Buffering_scheme_SA
    Buffering_scheme_VMEM = Hardware_param.Buffering_scheme_VMEM

    #print(Layer_name)
    #print(Buffering_scheme_SA)

    if (Exe_Hardware == "Systolic"):
        print("execution hardware systolic")
        if (Buffering_scheme_SA == "double"):
            print("Systolic double buffered scheme")
            if (Layer_name == "Convolution"):
                conv_access_model(Hardware_param, LayerObj, SysResult_inflayer)
                #ifmap_access_SRAM, filter_access_SRAM, psum_access_SRAM, bias_access_SRAM = SRAM_access
                #ifmap_access_DRAM, filter_access_DRAM, psum_access_DRAM, ofmap_access_DRAM, bias_access_DRAM = DRAM_access
                conv_cycle_model(Hardware_param, LayerObj, SysResult_inflayer)

            elif (Layer_name == "GEMM"):
                gemm_access_model(Hardware_param, LayerObj, SysResult_inflayer)
                gemm_cycle_model(Hardware_param, LayerObj, SysResult_inflayer)
            else:
                print("model do not exist now for layers other than convolution and gemm")

        else:
            print("Model do not exist for SA non double buffer scheme now")


    elif Exe_Hardware == "SIMD":
        print("execution hardware SIMD")
        if (Buffering_scheme_VMEM == "single"):
            print("SIMD single buffered scheme")
            if(Layer_name == "ReLU"):
                relu_access_model(Hardware_param, LayerObj, SIMDResult_inflayer)
                relu_cycle_model(Hardware_param, LayerObj, SIMDResult_inflayer)
            elif(Layer_name == "ElemAdd"):
                elemadd_access_model(Hardware_param, LayerObj, SIMDResult_inflayer)
                elemadd_cycle_model(Hardware_param, LayerObj, SIMDResult_inflayer)
            elif Layer_name == "MaxPool" or Layer_name == "AvgPool":
                pool_access_model(Hardware_param, LayerObj, SIMDResult_inflayer)
                pool_cycle_model(Hardware_param, LayerObj, SIMDResult_inflayer)
            elif Layer_name == "Softmax":
                softmax_access_model(Hardware_param, LayerObj, SIMDResult_inflayer)
                softmax_cycle_model(Hardware_param, LayerObj, SIMDResult_inflayer)
            elif Layer_name == "ROIAlignPool":
                roialignpool_access_model(Hardware_param, LayerObj, SIMDResult_inflayer)
                roialignpool_cycle_model(Hardware_param, LayerObj, SIMDResult_inflayer)
            else:
                print("Model do not exist yet for this layer, Layer_name:", Layer_name)

        else:
            print("Model do not exist for SIMD non single buffer scheme now")

    return SysResult_inflayer, SIMDResult_inflayer



def models_training_high(Hardware_param, LayerObj):
    #This function is mainly creating the hierarchy for the simulator

    SysResult_inflayer = SAResult_Inflayer()      # this result object will store all the results for the systolic array hardware
    SIMDResult_inflayer = SIMDResult_Inflayer()   # this result object will store all the results for the SIMD array hardware

    #unpacking the necessary variables
    Layer_name = LayerObj.Layer_name
    Exe_Hardware = LayerObj.Exe_Hardware

    Buffering_scheme_SA = Hardware_param.Buffering_scheme_SA
    Buffering_scheme_VMEM = Hardware_param.Buffering_scheme_VMEM

    #print(Layer_name)
    #print(Buffering_scheme_SA)

    if (Exe_Hardware == "Systolic"):
        print("execution hardware systolic")
        if (Buffering_scheme_SA == "double"):
            print("Systolic double buffered scheme")
            if (Layer_name == "Convolution"):
                conv_access_model(Hardware_param, LayerObj, SysResult_inflayer)
                #ifmap_access_SRAM, filter_access_SRAM, psum_access_SRAM, bias_access_SRAM = SRAM_access
                #ifmap_access_DRAM, filter_access_DRAM, psum_access_DRAM, ofmap_access_DRAM, bias_access_DRAM = DRAM_access
                conv_cycle_model(Hardware_param, LayerObj, SysResult_inflayer)

            elif (Layer_name == "GEMM"):
                gemm_access_model(Hardware_param, LayerObj, SysResult_inflayer)
                gemm_cycle_model(Hardware_param, LayerObj, SysResult_inflayer)
            else:
                print("model do not exist now for layers other than convolution and gemm")

        else:
            print("Model do not exist for SA non double buffer scheme now")


    elif Exe_Hardware == "SIMD":
        print("execution hardware SIMD")
        if (Buffering_scheme_VMEM == "single"):
            print("SIMD single buffered scheme")
            if(Layer_name == "ReLU"):
                relu_access_model(Hardware_param, LayerObj, SIMDResult_inflayer)
                relu_cycle_model(Hardware_param, LayerObj, SIMDResult_inflayer)
            elif(Layer_name == "ElemAdd"):
                elemadd_access_model(Hardware_param, LayerObj, SIMDResult_inflayer)
                elemadd_cycle_model(Hardware_param, LayerObj, SIMDResult_inflayer)
            elif Layer_name == "MaxPool" or Layer_name == "AvgPool":
                pool_access_model(Hardware_param, LayerObj, SIMDResult_inflayer)
                pool_cycle_model(Hardware_param, LayerObj, SIMDResult_inflayer)
            elif Layer_name == "Softmax":
                softmax_access_model(Hardware_param, LayerObj, SIMDResult_inflayer)
                softmax_cycle_model(Hardware_param, LayerObj, SIMDResult_inflayer)
            elif Layer_name == "ROIAlignPool":
                roialignpool_access_model(Hardware_param, LayerObj, SIMDResult_inflayer)
                roialignpool_cycle_model(Hardware_param, LayerObj, SIMDResult_inflayer)
            elif Layer_name == "Common_SIMD_Backward":
                common_SIMD_backward_model(Hardware_param, LayerObj, SIMDResult_inflayer) 
            elif Layer_name == 'MeanIstd':
                mean_istd_model(Hardware_param, LayerObj, SIMDResult_inflayer)
            elif Layer_name == 'BatchNorm':
                #batch_norm_forward_estimate(Hardware_param, LayerObj, SIMDResult_inflayer)
                batch_norm_forward_model(Hardware_param, LayerObj, SIMDResult_inflayer)
            elif Layer_name == "BatchNorm_Backward":
                #batch_norm_backward_estimate(Hardware_param, LayerObj, SIMDResult_inflayer)
                batch_norm_backward_model(Hardware_param, LayerObj, SIMDResult_inflayer)
            elif Layer_name == "MaxPool_Grad" or Layer_name == "AvgPool_Grad":
                pooling_backward_model(Hardware_param, LayerObj, SIMDResult_inflayer)
            else:
                print("Model do not exist yet for this layer, Layer_name:", Layer_name)

        else:
            print("Model do not exist for SIMD non single buffer scheme now")

    return SysResult_inflayer, SIMDResult_inflayer





    

    





