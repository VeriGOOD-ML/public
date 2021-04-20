# Simulator file, this is the interface file between the compiler and simulator

import logging
import math
import numpy as np
from SimulationModels import models_inference_high, models_training_high
#from GenerateTiling import Generate_tiling
import csv
import json
from pprint import pprint
from Data_Objects import HardwareObject, SAResult_Inflayer, SIMDResult_Inflayer
from Layer_Object import LayerObject

def simulate(CompilerOutput, Hardware_config):
    #Inputs:
    # Hardware_config: a .json file containing the hardware parameterization
    # CompilerOutput: .json file from the compiler at the hardware point specified by Hardware_config

    Result_header = ['Layer name', 'WBUF access', 'IBUF access', 'OBUF access', 'BUBF access', 'VMEM access', 'IMM access', 'InsMem access', \
                                   'DARM access filter', 'DRAM access ifmap', 'DRAM access psum', 'DRAM access ofmap', 'DRAM access bias', 'total DRAM access', \
                                   'SA compute cycles', 'SA stall cycles', 'SIMD compute cycles', 'SIMD stall cycles', 'total cycles']

    Layer_class_high = ["Convolution", "ConvIm2Col", "TransConv", "GEMM", "ReLU", "ElemAdd", "MaxPool", "AvgPool", "Sigmoid", "BatchNorm", "Softmax", "ROIAlignPool"]  # high class layers
    # May move "Softmax", "ROIAlignPool" from high to low later

    Mode = CompilerOutput['mode']   #using this to seperate inference and training mode
    Hardware_param = HardwareObject(Hardware_config, Mode)
    
    ##################################### Parsing Genesys Compiler outputs
    #print("Mode:", Mode) 
    if Mode == "inference":
        #Opening a .csv file to write the layer wise result
        with open("Layer_wise_result_Inference.csv", "w") as csvFile:  
            writer = csv.writer(csvFile)
            writer.writerow(Result_header)

            ######### Parsing the compiler output layer by layer inside the following for loop
            layer_nos = len(CompilerOutput['program'])  #number of layers in the neural network model
            print("layer_nos:", layer_nos)

            #layer_id = 1
            for layer_id in range(layer_nos):   # 1 would be replaced by layer_nos
            #if layer_id == 1:   # For Testing Purpose
                print("layer_id", layer_id)

                CompOut_layer = CompilerOutput['program'][layer_id]   # compiler output for each individual layer

                if layer_id == (layer_nos - 1):
                    next_layer = "None"         # no next layer for the last layer
                else:
                    next_layer = CompilerOutput['program'][layer_id + 1]['operation']  # this is needed to determine the ofmap bitwidth of SIMD
                    #print(CompOut_layer)
                    print("next_layer:", next_layer)

                LayerObj = LayerObject(Hardware_param, CompOut_layer, next_layer)   # this object contains all layer related parameters
                Layer_name = LayerObj.Layer_name
                print("Layer_name:", Layer_name)

                Layer_class = "low"   #initially setting the layer_class to be low
                for layername in Layer_class_high:
                    if layername == Layer_name:
                        Layer_class = "high"
                        break
                #print(Layer_class)

                # The top classification of the simulator is high/low layer class, Hence keeping that here
                if (Layer_class == "high"):
                    SysResult_inflayer, SIMDResult_inflayer = models_inference_high(Hardware_param, LayerObj)
                    
                    #print("Systolic SRAM_access:", SysResult_inflayer.SRAM_access)
                    #print("Systolic DRAM_access:", SysResult_inflayer.DRAM_access)
                    #print("Systolic cycle counts:", SysResult_inflayer.cycles)

                    #print("SIMDResult_inflayer.SRAM_access:", SIMDResult_inflayer.SRAM_access)
                    #print("SIMDResult_inflayer.DRAM_access:", SIMDResult_inflayer.DRAM_access)
                    #print("SIMDResult_inflayer.arithmetic:", SIMDResult_inflayer.arithmetic)
                    #print("SIMDResult_inflayer.cycles:", SIMDResult_inflayer.cycles)
                    #print("SIMDResult_inflayer.pipe_reg_access:", SIMDResult_inflayer.pipe_reg_access)
                    #print("SIMDResult_inflayer.indextbl_access:", SIMDResult_inflayer.indextbl_access)
                    #print("SIMDResult_inflayer.addrgen_add:", SIMDResult_inflayer.addrgen_add)
                    
                elif (Layer_class == "low"):
                    print("model do not exist for low class yet")

                else:
                    print("invalid layer class")

                ########### Writing down the results of inference of all layers in a .csv file here (inside the layer for loop)
                if LayerObj.Exe_Hardware == "Systolic":
                    WBUF_access = SysResult_inflayer.SRAM_access['filter']/(8*1024)
                    IBUF_access = SysResult_inflayer.SRAM_access['ifmap']/(8*1024)
                    OBUF_access = SysResult_inflayer.SRAM_access['psum']/(8*1024)
                    BUBF_access = SysResult_inflayer.SRAM_access['bias']/(8*1024)
                    VMEM_access = 0
                    IMM_access = 0
                    InsMem_access = 0

                    DARM_access_filter = SysResult_inflayer.DRAM_access['filter']/(8*1024)
                    DRAM_access_ifmap = SysResult_inflayer.DRAM_access['ifmap']/(8*1024)
                    DRAM_access_psum = SysResult_inflayer.DRAM_access['psum']/(8*1024)
                    DRAM_access_ofmap = SysResult_inflayer.DRAM_access['ofmap']/(8*1024)
                    DRAM_access_bias = SysResult_inflayer.DRAM_access['bias']/(8*1024)
                    total_DRAM_access = DARM_access_filter + DRAM_access_ifmap + DRAM_access_psum + DRAM_access_ofmap + DRAM_access_bias

                    SA_compute_cycles = SysResult_inflayer.cycles['compute']
                    SA_stall_cycles =  SysResult_inflayer.cycles['DRAM_Stall']
                    SIMD_compute_cycles = 0
                    SIMD_stall_cycles = 0
                    total_cycles = SA_compute_cycles + SA_stall_cycles + SIMD_compute_cycles + SIMD_stall_cycles
                
                
                if LayerObj.Exe_Hardware == "SIMD":
                    WBUF_access = 0
                    IBUF_access = SIMDResult_inflayer.SRAM_access['IBUF']/(8*1024)
                    OBUF_access = SIMDResult_inflayer.SRAM_access['OBUF']/(8*1024)
                    BUBF_access = 0
                    VMEM_access = SIMDResult_inflayer.SRAM_access['VMEM']/(8*1024)
                    IMM_access = SIMDResult_inflayer.SRAM_access['IMM']/(8*1024)
                    InsMem_access = SIMDResult_inflayer.SRAM_access['InsMem']/(8*1024)

                    DARM_access_filter = SIMDResult_inflayer.DRAM_access['filter']/(8*1024)
                    DRAM_access_ifmap = SIMDResult_inflayer.DRAM_access['ifmap']/(8*1024)
                    DRAM_access_psum = SIMDResult_inflayer.DRAM_access['intermediate']/(8*1024)
                    DRAM_access_ofmap = SIMDResult_inflayer.DRAM_access['ofmap']/(8*1024)
                    DRAM_access_bias = 0
                    total_DRAM_access = DARM_access_filter + DRAM_access_ifmap + DRAM_access_psum + DRAM_access_ofmap + DRAM_access_bias

                    SA_compute_cycles = 0
                    SA_stall_cycles =  0
                    SIMD_compute_cycles = SIMDResult_inflayer.cycles['compute']
                    SIMD_stall_cycles = SIMDResult_inflayer.cycles['DRAM_stall']
                    total_cycles = SA_compute_cycles + SA_stall_cycles + SIMD_compute_cycles + SIMD_stall_cycles

                res_row_write = [Layer_name, WBUF_access, IBUF_access, OBUF_access, BUBF_access, VMEM_access, IMM_access, InsMem_access,\
                                    DARM_access_filter, DRAM_access_ifmap, DRAM_access_psum, DRAM_access_ofmap, DRAM_access_bias, total_DRAM_access, \
                                    SA_compute_cycles, SA_stall_cycles, SIMD_compute_cycles, SIMD_stall_cycles, total_cycles]

                #print(res_row_write)
                writer.writerow(res_row_write)

                ####### This is to create a numpy array to add the results of all layers for a network
                row_for_array = [WBUF_access, IBUF_access, OBUF_access, BUBF_access, VMEM_access, IMM_access, InsMem_access,\
                                    DARM_access_filter, DRAM_access_ifmap, DRAM_access_psum, DRAM_access_ofmap, DRAM_access_bias, total_DRAM_access, \
                                    SA_compute_cycles, SA_stall_cycles, SIMD_compute_cycles, SIMD_stall_cycles, total_cycles]

                #print("row_for_array:", row_for_array)
                if layer_id == 0:
                    full_res_array = np.array(row_for_array)
                else:
                    full_res_array = np.vstack([ full_res_array , row_for_array])
        
        ###The array containing inference result of all layer
        #print("full_res_array", full_res_array)
        Final_inf_result = np.sum(full_res_array, axis = 0)  # summing the result across all layers, column-wise summation
        #print("Final_inf_result:", Final_inf_result)

        return Final_inf_result

    else:
        print("running training graph for one iteration")
        #Opening a .csv file to write the layer wise result
        with open("Layer_wise_result_Training.csv", "w") as csvFile:  
            writer = csv.writer(csvFile)
            writer.writerow(Result_header)

            ######### Parsing the compiler output layer by layer inside the following for loop
            layer_nos = len(CompilerOutput['program'])  #number of layers in the neural network model
            print("layer_nos:", layer_nos)

            #layer_id = 0
            for layer_id in range(layer_nos):   # 1 would be replaced by layer_nos
            #if layer_id == 0:   # For Testing Purpose
                print("layer_id", layer_id)

                CompOut_layer = CompilerOutput['program'][layer_id]   # compiler output for each individual layer
                #print(CompOut_layer)
                
                if layer_id == (layer_nos - 1):
                    next_layer = "None"         # no next layer for the last layer
                else:
                    next_layer = CompilerOutput['program'][layer_id + 1]['operation']  # this is needed to determine the ofmap bitwidth of SIMD
                    print("next_layer:", next_layer)

                LayerObj = LayerObject(Hardware_param, CompOut_layer, next_layer)   # this object contains all layer related parameters
                Layer_name = LayerObj.Layer_name
                print("Layer_name:", Layer_name)

                SysResult_inflayer, SIMDResult_inflayer = models_training_high(Hardware_param, LayerObj)
                
                #print("Systolic SRAM_access:", SysResult_inflayer.SRAM_access)
                #print("Systolic DRAM_access:", SysResult_inflayer.DRAM_access)
                #print("Systolic cycle counts:", SysResult_inflayer.cycles)

                #print("SIMDResult_inflayer.SRAM_access:", SIMDResult_inflayer.SRAM_access)
                #print("SIMDResult_inflayer.DRAM_access:", SIMDResult_inflayer.DRAM_access)
                #print("SIMDResult_inflayer.arithmetic:", SIMDResult_inflayer.arithmetic)
                #print("SIMDResult_inflayer.cycles:", SIMDResult_inflayer.cycles)
                #print("SIMDResult_inflayer.pipe_reg_access:", SIMDResult_inflayer.pipe_reg_access)
                #print("SIMDResult_inflayer.indextbl_access:", SIMDResult_inflayer.indextbl_access)
                #print("SIMDResult_inflayer.addrgen_add:", SIMDResult_inflayer.addrgen_add)

                ########### Writing down the results of inference of all layers in a .csv file here (inside the layer for loop)
                if LayerObj.Exe_Hardware == "Systolic":
                    WBUF_access = SysResult_inflayer.SRAM_access['filter']/(8*1024)
                    IBUF_access = SysResult_inflayer.SRAM_access['ifmap']/(8*1024)
                    OBUF_access = SysResult_inflayer.SRAM_access['psum']/(8*1024)
                    BUBF_access = SysResult_inflayer.SRAM_access['bias']/(8*1024)
                    VMEM_access = 0
                    IMM_access = 0
                    InsMem_access = 0

                    DARM_access_filter = SysResult_inflayer.DRAM_access['filter']/(8*1024)
                    DRAM_access_ifmap = SysResult_inflayer.DRAM_access['ifmap']/(8*1024)
                    DRAM_access_psum = SysResult_inflayer.DRAM_access['psum']/(8*1024)
                    DRAM_access_ofmap = SysResult_inflayer.DRAM_access['ofmap']/(8*1024)
                    DRAM_access_bias = SysResult_inflayer.DRAM_access['bias']/(8*1024)
                    total_DRAM_access = DARM_access_filter + DRAM_access_ifmap + DRAM_access_psum + DRAM_access_ofmap + DRAM_access_bias

                    SA_compute_cycles = SysResult_inflayer.cycles['compute']
                    SA_stall_cycles =  SysResult_inflayer.cycles['DRAM_Stall']
                    SIMD_compute_cycles = 0
                    SIMD_stall_cycles = 0
                    total_cycles = SA_compute_cycles + SA_stall_cycles + SIMD_compute_cycles + SIMD_stall_cycles
                
                
                if LayerObj.Exe_Hardware == "SIMD":
                    WBUF_access = 0
                    IBUF_access = SIMDResult_inflayer.SRAM_access['IBUF']/(8*1024)
                    OBUF_access = SIMDResult_inflayer.SRAM_access['OBUF']/(8*1024)
                    BUBF_access = 0
                    VMEM_access = SIMDResult_inflayer.SRAM_access['VMEM']/(8*1024)
                    IMM_access = SIMDResult_inflayer.SRAM_access['IMM']/(8*1024)
                    InsMem_access = SIMDResult_inflayer.SRAM_access['InsMem']/(8*1024)

                    DARM_access_filter = SIMDResult_inflayer.DRAM_access['filter']/(8*1024)
                    DRAM_access_ifmap = SIMDResult_inflayer.DRAM_access['ifmap']/(8*1024)
                    DRAM_access_psum = SIMDResult_inflayer.DRAM_access['intermediate']/(8*1024)
                    DRAM_access_ofmap = SIMDResult_inflayer.DRAM_access['ofmap']/(8*1024)
                    DRAM_access_bias = 0
                    total_DRAM_access = DARM_access_filter + DRAM_access_ifmap + DRAM_access_psum + DRAM_access_ofmap + DRAM_access_bias

                    SA_compute_cycles = 0
                    SA_stall_cycles =  0
                    SIMD_compute_cycles = SIMDResult_inflayer.cycles['compute']
                    SIMD_stall_cycles = SIMDResult_inflayer.cycles['DRAM_stall']
                    total_cycles = SA_compute_cycles + SA_stall_cycles + SIMD_compute_cycles + SIMD_stall_cycles

                res_row_write = [Layer_name, WBUF_access, IBUF_access, OBUF_access, BUBF_access, VMEM_access, IMM_access, InsMem_access,\
                                    DARM_access_filter, DRAM_access_ifmap, DRAM_access_psum, DRAM_access_ofmap, DRAM_access_bias, total_DRAM_access, \
                                    SA_compute_cycles, SA_stall_cycles, SIMD_compute_cycles, SIMD_stall_cycles, total_cycles]

                #print(res_row_write)
                writer.writerow(res_row_write)

                ####### This is to create a numpy array to add the results of all layers for a network
                row_for_array = [WBUF_access, IBUF_access, OBUF_access, BUBF_access, VMEM_access, IMM_access, InsMem_access,\
                                    DARM_access_filter, DRAM_access_ifmap, DRAM_access_psum, DRAM_access_ofmap, DRAM_access_bias, total_DRAM_access, \
                                    SA_compute_cycles, SA_stall_cycles, SIMD_compute_cycles, SIMD_stall_cycles, total_cycles]

                #print("row_for_array:", row_for_array)
                if layer_id == 0:
                    full_res_array = np.array(row_for_array)
                else:
                    full_res_array = np.vstack([ full_res_array , row_for_array])
                
        ###The array containing training result of all layer: this result is for one iteration, i.e., one forward + one backward pass
        #print("full_res_array", full_res_array)
        Final_train_result_single_iteration = np.sum(full_res_array, axis = 0)  # summing the result across all layers, column-wise summation
        #print("Final_train_result_single_iteration:", Final_train_result_single_iteration)

        return Final_train_result_single_iteration

























