import logging
import math
import numpy as np
from simulation_models import models_inference_high, models_training_high
import csv
import json
from pprint import pprint
from data_objects import HardwareObject, SAResult_Inflayer, SIMDResult_Inflayer
from layer_object import LayerObject, TilingFlags

def fusion_status(layer_id, DNNSpecNet):
    valid_layer_list = ['conv_bias', 'conv', 'gemm', 'gemm_no_bias', 'relu', 'elem_add', 'max_pool', 'avg_pool', 'global_avg_pool', \
                            'sgd4d', 'sgd2d', 'sgd1d', 'relu_grad', 'elem_add_grad', 'reduce_sum', 'mean_var', 'batch_norm', 'batchnorm_grad_x_mu', \
                            'max_pool_grad', 'average_pool_grad', 'global_average_pool_grad']

    layer_nos = len(DNNSpecNet['program'])  #number of layers in the neural network model
    
    if layer_id == 0:  # first layer
        prevlayer = "None"
        
        i = 1
        next_flag = False
        while (next_flag == False):
            nextlayer = DNNSpecNet['program'][layer_id + i]['operation']  
            if nextlayer in valid_layer_list:
                next_flag = True
            i = i + 1

    elif layer_id == (layer_nos - 1):  #last layer
        j = 1
        prev_flag = False
        while (prev_flag == False):
            prevlayer = DNNSpecNet['program'][layer_id - j]['operation']  
            if prevlayer in valid_layer_list:
                prev_flag = True
            j = j + 1

        nextlayer = "None"

    else:
        j = 1
        prev_flag = False
        while (prev_flag == False):
            prevlayer = DNNSpecNet['program'][layer_id - j]['operation']  
            if prevlayer in valid_layer_list:
                prev_flag = True
            j = j + 1

        i = 1
        next_flag = False
        while (next_flag == False):
            nextlayer = DNNSpecNet['program'][layer_id + i]['operation']  
            if nextlayer in valid_layer_list:
                next_flag = True
            i = i + 1

    print("prevlayer:", prevlayer)
    
    sa_layer_list = ['conv_bias', 'conv', 'gemm', 'gemm_no_bias']
    simd_layer_list = ['relu', 'elem_add', 'max_pool', 'avg_pool', 'global_avg_pool', \
                            'sgd4d', 'sgd2d', 'sgd1d', 'relu_grad', 'elem_add_grad', 'reduce_sum', 'mean_var', 'batch_norm', 'batchnorm_grad_x_mu', \
                            'max_pool_grad', 'average_pool_grad', 'global_average_pool_grad']

    current_layer = DNNSpecNet['program'][layer_id]['operation'] 

    if (current_layer in sa_layer_list) and (nextlayer in simd_layer_list):
        fusion_flag = True
    elif (current_layer in simd_layer_list) and (prevlayer in sa_layer_list):
        fusion_flag = True
    else:
        fusion_flag = False

    # Some additional check to make sure fusion pairs are within the supported list
    if fusion_flag == True:
        check = False
        if (current_layer == 'conv_bias' or current_layer == 'conv') and (nextlayer == 'relu'):
            check = True
        elif (current_layer == 'conv_bias' or current_layer == 'conv') and (nextlayer == 'elem_add'):
            check = True
        elif (current_layer == 'conv_bias' or current_layer == 'conv') and (nextlayer == 'mean_var'):
            check = True
        elif (current_layer == 'conv_bias' or current_layer == 'conv') and (nextlayer == 'sgd4d'):
            check = True
        elif (current_layer == 'gemm' or current_layer == 'gemm_no_bias') and (nextlayer == 'sgd2d'):
            check = True

        elif (current_layer == 'relu') and (prevlayer == 'conv_bias' or prevlayer == 'conv'):
            check = True
        elif (current_layer == 'elem_add') and (prevlayer == 'conv_bias' or prevlayer == 'conv'):
            check = True
        elif (current_layer == 'mean_var') and (prevlayer == 'conv_bias' or prevlayer == 'conv'):
            check = True
        elif (current_layer == 'sgd4d') and (prevlayer == 'conv_bias' or prevlayer == 'conv'):
            check = True
        elif (current_layer == 'sgd2d') and (prevlayer == 'gemm' or prevlayer == 'gemm_no_bias'):
            check = True

        assert check

    return fusion_flag


def valid_layer_nos(DNNSpecNet):
    valid_layer_list = ['conv_bias', 'conv', 'gemm', 'gemm_no_bias', 'relu', 'elem_add', 'max_pool', 'avg_pool', 'global_avg_pool', \
                            'sgd4d', 'sgd2d', 'sgd1d', 'relu_grad', 'elem_add_grad', 'reduce_sum', 'mean_var', 'batch_norm', 'batchnorm_grad_x_mu', \
                            'max_pool_grad', 'average_pool_grad', 'global_average_pool_grad']
    
    additional_layer = ['batchnorm_grad_inv_std', 'batchnorm_grad_xhat', 'batchnorm_grad_dbeta', 'batchnorm_grad_dgamma', 'batchnorm_grad_dgamma_mul_xhat', \
                            'batchnorm_grad_gamma_inv_std', 'batchnorm_grad_scaled_gy', 'batchnorm_grad_dx_rhs', 'batchnorm_grad_dx']

    valid_layer_list_extend = valid_layer_list + additional_layer
    
    layer_nos = len(DNNSpecNet['program'])  #number of layers in the neural network model

    layer_count = 0
    layer_count_extend = 0

    for layer_id in range(layer_nos):
        layer_op = DNNSpecNet['program'][layer_id]['operation']
        
        if layer_op in valid_layer_list:
            layer_count = layer_count + 1

        if layer_op in valid_layer_list_extend:
            layer_count_extend = layer_count_extend + 1

    return layer_count, layer_count_extend



def simulate(DNNSpecNet, Hardware_config, Optimal_WS_loop, TGflag, FusionChoice):

    Result_header = ['Layer name', 'WBUF access', 'IBUF access', 'OBUF access', 'BBUF access', 'VMEM access', 'IMM access', 'InsMem access', \
                                   'DRAM access filter', 'DRAM access ifmap', 'DRAM access psum', 'DRAM access ofmap', 'DRAM access bias', 'total DRAM access', \
                                   'SA compute cycles', 'SA stall cycles', 'SIMD compute cycles', 'SIMD stall cycles', 'total cycles', 'Op count']

    Mode = DNNSpecNet['mode']   #using this to seperate inference and training mode
    Hardware_param = HardwareObject(Hardware_config, Mode)
    
    ##################################### Parsing DNN specification file
    #print("Mode:", Mode) 
    if Mode == "inference":
        print("running inference graph")
        layer_wise_res_file_name = "Layer_wise_result_Inference.csv"
    elif Mode == "training":
        print("running training graph for one iteration")
        layer_wise_res_file_name = "Layer_wise_result_Training.csv"


    count_fused_layer = 0
    #Opening a .csv file to write the layer wise result
    with open(layer_wise_res_file_name, "w") as csvFile:  
        writer = csv.writer(csvFile)
        writer.writerow(Result_header)

        ######### Parsing the DNN specifications layer by layer inside the following for loop
        layer_nos = len(DNNSpecNet['program'])  #number of layers in the neural network model
        #print("layer_nos:", layer_nos)

        #layer_id = 254 #for testing a single layer
        for layer_id in range(layer_nos):   
        #if layer_id == 254:   #for testing a single layer
            print("layer_id", layer_id)

            CompOut_layer = DNNSpecNet['program'][layer_id]   # DNN spec for each individual layer

            if layer_id == (layer_nos - 1):
                next_layer = "None"         # no next layer for the last layer                
            else:
                next_layer = DNNSpecNet['program'][layer_id + 1]['operation']  # this is needed to determine the ofmap bitwidth of SIMD                
                #print(CompOut_layer)
            print("next_layer:", next_layer)

            if FusionChoice == True:
                fusion_flag = fusion_status(layer_id, DNNSpecNet)
            else:
                fusion_flag = False
            print("fusion_flag:", fusion_flag)
            if fusion_flag == True:
                count_fused_layer = count_fused_layer + 1

            LayerObj = LayerObject(Hardware_param, CompOut_layer, next_layer, Optimal_WS_loop, TGflag, fusion_flag, Mode)   # this object contains all layer related parameters
            #print("LayerObj:", vars(LayerObj))

            Layer_name = LayerObj.Layer_name
            print("Layer_name:", Layer_name)

            if Mode == "inference":
                SysResult_inflayer, SIMDResult_inflayer = models_inference_high(Hardware_param, LayerObj)
            elif Mode == "training":
                SysResult_inflayer, SIMDResult_inflayer = models_training_high(Hardware_param, LayerObj)
            
            print("Systolic SRAM_access:", SysResult_inflayer.SRAM_access)
            print("Systolic DRAM_access:", SysResult_inflayer.DRAM_access)
            print("Systolic cycle counts:", SysResult_inflayer.cycles)
            #print ("Systolic MAC count:",SysResult_inflayer.arithmetic)

            print("SIMDResult_inflayer.SRAM_access:", SIMDResult_inflayer.SRAM_access)
            print("SIMDResult_inflayer.DRAM_access:", SIMDResult_inflayer.DRAM_access)
            #print("SIMDResult_inflayer.arithmetic:", SIMDResult_inflayer.arithmetic)
            print("SIMDResult_inflayer.cycles:", SIMDResult_inflayer.cycles)
                
            ########### Writing down the results of inference of all layers in a .csv file here (inside the layer for loop)
            if LayerObj.Exe_Hardware == "Systolic":
                WBUF_access = SysResult_inflayer.SRAM_access['filter']/(8*1024)
                IBUF_access = SysResult_inflayer.SRAM_access['ifmap']/(8*1024)
                OBUF_access = SysResult_inflayer.SRAM_access['psum']/(8*1024)
                BBUF_access = SysResult_inflayer.SRAM_access['bias']/(8*1024)
                VMEM_access = 0
                IMM_access = 0
                InsMem_access = 0

                DRAM_access_filter = SysResult_inflayer.DRAM_access['filter']/(8*1024)
                DRAM_access_ifmap = SysResult_inflayer.DRAM_access['ifmap']/(8*1024)
                DRAM_access_psum = SysResult_inflayer.DRAM_access['psum']/(8*1024)
                DRAM_access_ofmap = SysResult_inflayer.DRAM_access['ofmap']/(8*1024)
                DRAM_access_bias = SysResult_inflayer.DRAM_access['bias']/(8*1024)
                total_DRAM_access = DRAM_access_filter + DRAM_access_ifmap + DRAM_access_psum + DRAM_access_ofmap + DRAM_access_bias

                SA_compute_cycles = SysResult_inflayer.cycles['compute']
                SA_stall_cycles =  SysResult_inflayer.cycles['DRAM_Stall']
                SIMD_compute_cycles = 0
                SIMD_stall_cycles = 0
                total_cycles = SA_compute_cycles + SA_stall_cycles + SIMD_compute_cycles + SIMD_stall_cycles

                op_count = SysResult_inflayer.arithmetic['mac'] * 2    # multiply and accumulate, hence 2 op

                # additional statistics for the layers executed on systolic array (last four is zero data to be able to do the column-wise sum)
                total_SRAM_access_SA = WBUF_access + IBUF_access + OBUF_access + BBUF_access
                total_DRAM_access_SA = total_DRAM_access
                total_cycles_SA = SA_compute_cycles + SA_stall_cycles
                op_count_SA = op_count
                total_SRAM_access_SIMD = 0
                total_DRAM_access_SIMD = 0
                total_cycles_SIMD = 0
                op_count_SIMD = 0
            
            
            if LayerObj.Exe_Hardware == "SIMD":
                WBUF_access = 0
                IBUF_access = SIMDResult_inflayer.SRAM_access['IBUF']/(8*1024)  
                OBUF_access = SIMDResult_inflayer.SRAM_access['OBUF']/(8*1024)
                BBUF_access = 0
                VMEM_access = SIMDResult_inflayer.SRAM_access['VMEM']/(8*1024)
                IMM_access = SIMDResult_inflayer.SRAM_access['IMM']/(8*1024)
                InsMem_access = SIMDResult_inflayer.SRAM_access['InsMem']/(8*1024)
                #for noFusion case, IBUF & OBUF acess are zero for the SIMD layers. Puttig an assert to check this sice the simulator does not have a fusion model now
                assert IBUF_access == 0
                assert OBUF_access == 0

                DRAM_access_filter = SIMDResult_inflayer.DRAM_access['filter']/(8*1024)
                DRAM_access_ifmap = SIMDResult_inflayer.DRAM_access['ifmap']/(8*1024)
                DRAM_access_psum = SIMDResult_inflayer.DRAM_access['intermediate']/(8*1024)
                DRAM_access_ofmap = SIMDResult_inflayer.DRAM_access['ofmap']/(8*1024)
                DRAM_access_bias = 0
                total_DRAM_access = DRAM_access_filter + DRAM_access_ifmap + DRAM_access_psum + DRAM_access_ofmap + DRAM_access_bias

                SA_compute_cycles = 0
                SA_stall_cycles =  0
                SIMD_compute_cycles = SIMDResult_inflayer.cycles['compute']
                SIMD_stall_cycles = SIMDResult_inflayer.cycles['DRAM_stall']
                total_cycles = SA_compute_cycles + SA_stall_cycles + SIMD_compute_cycles + SIMD_stall_cycles

                op_count = SIMDResult_inflayer.arithmetic['max'] + SIMDResult_inflayer.arithmetic['add'] + SIMDResult_inflayer.arithmetic['op_ScmnBN'] \
                            + (SIMDResult_inflayer.arithmetic['div'] * Hardware_param.div_cycles) + (SIMDResult_inflayer.arithmetic['exp'] * Hardware_param.exp_cycles) \
                            + (SIMDResult_inflayer.arithmetic['inv_sqrt'] * Hardware_param.inv_sqrt_cycles) \
                            + SIMDResult_inflayer.arithmetic['sub'] + SIMDResult_inflayer.arithmetic['mul'] \
                            + SIMDResult_inflayer.arithmetic['CondMove'] * Hardware_param.cond_move_cycles

                # additional statistics for the layers executed on SIMD array (1st four is zero data to be able to do the column-wise sum)
                total_SRAM_access_SA = 0
                total_DRAM_access_SA = 0
                total_cycles_SA = 0
                op_count_SA = 0
                total_SRAM_access_SIMD = VMEM_access + IMM_access + InsMem_access + IBUF_access + OBUF_access #IBUF & OBUF access are zero for SIMD layers for noFusion
                total_DRAM_access_SIMD = total_DRAM_access
                total_cycles_SIMD = SIMD_compute_cycles + SIMD_stall_cycles
                op_count_SIMD = op_count
            

            res_row_write = [Layer_name, WBUF_access, IBUF_access, OBUF_access, BBUF_access, VMEM_access, IMM_access, InsMem_access,\
                                DRAM_access_filter, DRAM_access_ifmap, DRAM_access_psum, DRAM_access_ofmap, DRAM_access_bias, total_DRAM_access, \
                                SA_compute_cycles, SA_stall_cycles, SIMD_compute_cycles, SIMD_stall_cycles, total_cycles, op_count]

            #print(res_row_write)
            writer.writerow(res_row_write)

            ####### This is to create a numpy array to add the results of all layers for a network
            row_for_array = [WBUF_access, IBUF_access, OBUF_access, BBUF_access, VMEM_access, IMM_access, InsMem_access,\
                                DRAM_access_filter, DRAM_access_ifmap, DRAM_access_psum, DRAM_access_ofmap, DRAM_access_bias, total_DRAM_access, \
                                SA_compute_cycles, SA_stall_cycles, SIMD_compute_cycles, SIMD_stall_cycles, total_cycles, op_count]

            #print("row_for_array:", row_for_array)
            #if layer_id == 254: #for testing a single layer
            if layer_id == 0:
                full_res_array = np.array(row_for_array)
            else:
                full_res_array = np.vstack([ full_res_array , row_for_array])

            ###### This is to create a numpy array to add the SA-SIMD breakdown results of all layers for a network
            row_for_SA_SIMD = [WBUF_access, IBUF_access, OBUF_access, BBUF_access, total_SRAM_access_SA, \
                                    VMEM_access, IMM_access, InsMem_access, total_SRAM_access_SIMD, \
                                    total_DRAM_access_SA, total_DRAM_access_SIMD, \
                                    SA_compute_cycles, SA_stall_cycles, total_cycles_SA, \
                                    SIMD_compute_cycles, SIMD_stall_cycles, total_cycles_SIMD, \
                                    op_count_SA, op_count_SIMD]

            #if layer_id == 254: #for testing a single layer
            if layer_id == 0:
                full_res_SA_SIMD = np.array(row_for_SA_SIMD)
            else:
                full_res_SA_SIMD = np.vstack([ full_res_SA_SIMD , row_for_SA_SIMD])
    
    ###The array containing result of all layer
    #print("full_res_array", full_res_array)
    Final_inf_or_singleiter_train_result = np.sum(full_res_array, axis = 0)  # summing the result across all layers, column-wise summation
    #print("Final_inf_or_singleiter_train_result:", Final_inf_or_singleiter_train_result) # for training this is the result for a single iteration

    # SA-SIMD breakdown result for the full network (inference or single iteration training)
    SA_SIMD_result_net = np.sum(full_res_SA_SIMD, axis = 0) # summing the result across all layers, column-wise summation
    
    layer_count, layer_count_extend = valid_layer_nos(DNNSpecNet)
    print(f'count_fused_layer = {count_fused_layer}, layer_nos = {layer_nos}, layer_count = {layer_count}, layer_count_extend = {layer_count_extend}')
    print(f'fusion_prcnt = {count_fused_layer/layer_count * 100}, fusion_prcnt_extend = {count_fused_layer/layer_count_extend * 100}')

    return Final_inf_or_singleiter_train_result, SA_SIMD_result_net  



























