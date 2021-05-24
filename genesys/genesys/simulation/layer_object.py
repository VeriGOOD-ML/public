### This file contains the parameters associated to all the layers

import logging
import math
import numpy as np
from data_objects import HardwareObject

class LayerObject(object):
    # This object stores all the parameters associated to a layer. This includes layer dimension, tiling info, fusion status etc
    # This object takes compiler output associated with each layer and the hardware object as input arguments
    def __init__(self, Hardware_param, CompOut_layer, next_layer, Optimal_WS_loop):
        # Ecah type of layer is read under seperate if condition. the dictionary keys and number of variables for different kind of layer is diferent in compiler output
        layer_op = CompOut_layer['operation']
        print("layer operation:", layer_op)

        #For layers where IC = OC in the function code, used OC parameters and IC paramters are ignored (used "None") in the simulator model
        #For example, for ReLU, IC, DTile_ic, STile_ic etc will not be used in their model's code (used "None" for these parameters)
        #For layers with no kernel, default is set as KH = KW = 1, Stride = 1, Pad = 0, none of these parameters are used. IH and IW are also not used for such layers
        #for SIMD, Technically, I do not need bw_psum since ifmap and psum (intermediate data) are same bitwidth for SIMD, but keeping it seperate if needed in future
        #for SIMD, psum means intermediate data and using it for intermediate DARM access if any, for example for softmax
        #for pooling do not use ic in the loop order, use only oc
        #use Conv/FC activation bitwidth from Hardware Object as SIMD ofmap bitwidth depending on whether the SIMD layer is before a conv or not

        ########################################################################### Convolution Layer
        if layer_op == "conv_bias" or layer_op == "conv":
            self.Layer_name = "Convolution"
            ##### Dimensions of the layer tensors
            #print(CompOut_layer['iterable_dimensions'])
            self.OW = CompOut_layer['iterable_dimensions']['OW']
            self.OH = CompOut_layer['iterable_dimensions']['OH']
            self.OC = CompOut_layer['iterable_dimensions']['OC']
            self.KW = CompOut_layer['iterable_dimensions']['KW']
            self.KH = CompOut_layer['iterable_dimensions']['KH']
            self.IC = CompOut_layer['iterable_dimensions']['IC']
            self.Batch = CompOut_layer['iterable_dimensions']['N']
            self.Stride = CompOut_layer['operation_parameters']['stride']
            #self.Pad = 0            
            self.IW = (self.OW - 1) * self.Stride + self.KW   #this will autometically give the padded ifmap height & width
            self.IH = (self.OH - 1) * self.Stride + self.KH

            ######Tiling parameters from DRAM to SRAM level 
            #print("weight:", CompOut_layer['inputs'][1]['tiling']['WBUF'])    # index 1 is weight in the list under outputs
            #print("ofmap:", CompOut_layer['outputs'][0]['tiling']['OBUF'])    # index 0 is ofmap in the list under outputs
            self.DTile_ow = CompOut_layer['outputs'][0]['tiling']['OBUF']['OW']
            self.DTile_oh = CompOut_layer['outputs'][0]['tiling']['OBUF']['OH']
            self.DTile_oc = CompOut_layer['outputs'][0]['tiling']['OBUF']['OC']
            self.DTile_kw = CompOut_layer['inputs'][1]['tiling']['WBUF']['KW']
            self.DTile_kh = CompOut_layer['inputs'][1]['tiling']['WBUF']['KH']
            self.DTile_ic = CompOut_layer['inputs'][1]['tiling']['WBUF']['IC']
            self.DTile_batch = CompOut_layer['outputs'][0]['tiling']['OBUF']['N']
            self.DTile_iw = (self.DTile_ow - 1) * self.Stride + self.DTile_kw
            self.DTile_ih = (self.DTile_oh - 1) * self.Stride + self.DTile_kh

            #Tiling parameters from SRAM to PE level
            #print("ifmap:", CompOut_layer['inputs'][0]['tiling']['pe_array'])    # index 0 is ifmap in the list under inputs
            #print("weight:", CompOut_layer['inputs'][1]['tiling']['pe_array'])    # index 1 is weight in the list under inputs
            #print("ofmap:", CompOut_layer['outputs'][0]['tiling']['pe_array'])    # index 0 is ofmap in the list under outputs
            self.Stile_ow = CompOut_layer['outputs'][0]['tiling']['pe_array']['OW']
            self.Stile_oh = CompOut_layer['outputs'][0]['tiling']['pe_array']['OH']
            self.Stile_oc = CompOut_layer['outputs'][0]['tiling']['pe_array']['OC']
            self.Stile_kw = CompOut_layer['inputs'][1]['tiling']['pe_array']['KW']
            self.Stile_kh = CompOut_layer['inputs'][1]['tiling']['pe_array']['KH']
            self.Stile_ic = CompOut_layer['inputs'][1]['tiling']['pe_array']['IC']
            self.Stile_batch = CompOut_layer['outputs'][0]['tiling']['pe_array']['N']
            self.Stile_iw = CompOut_layer['inputs'][0]['tiling']['pe_array']['IW']
            self.Stile_ih = CompOut_layer['inputs'][0]['tiling']['pe_array']['IH']

            #### Extracting loop order
            #print("Optimal_WS_loop:", Optimal_WS_loop)
            if Optimal_WS_loop == False:
                #print(CompOut_layer['iterable_dimensions'].keys())
                loop_keys = list(CompOut_layer['iterable_dimensions'].keys())
                loop_keys.reverse()     # in simulator code, the first one in the list is the innermost loop (opposite to the compiler output)
                #print(loop_keys)
                loop_keys_lower = [key.lower() for key in loop_keys]   # in simulator code, all the loop keys are lower case (compiler output is upper case)
                self.Loop_order = loop_keys_lower
                #self.Loop_order = ['ow', 'oh', 'kw', 'kh', 'ic', 'n', 'oc']   # the first one, i.e., ow is the inner most loop and the last one i.e., oc is the outermost loop
            elif Optimal_WS_loop == True:
                self.Loop_order = ['ow', 'oh', 'n', 'kw', 'kh', 'ic', 'oc']
            #print("Loop order:", self.Loop_order)

            #### Extracting fusion info. 
            #print(CompOut_layer['inputs'][0]['data_path'][0])
            #print(CompOut_layer['outputs'][0]['data_path'][-1])
            # for convolution layer, a nofusion means: ifmap comes from DRAM and ofmap goes to DRAM
            initial_ifmap_storage = CompOut_layer['inputs'][0]['data_path'][0]
            final_ofmap_storage = CompOut_layer['outputs'][0]['data_path'][-1]
            if initial_ifmap_storage == "DRAM" and final_ofmap_storage == "DRAM":
                self.fusion_status = "NoFusion"

            ### extracting the execution hardware based on the end location of first input
            input_end_location = CompOut_layer['inputs'][0]['data_path'][-1]  # end location of first input
            if input_end_location == "pe_array":
                self.Exe_Hardware = "Systolic"    
            elif input_end_location == "SIMD":
                self.Exe_Hardware = "SIMD"

            #print(CompOut_layer['inputs'][0]['dtype'])
            #print(CompOut_layer['inputs'][1]['dtype'])
            #print(CompOut_layer['inputs'][2]['dtype'])
            #print(CompOut_layer['outputs'][0]['dtype'])
            self.bw_ifmap = int(''.join(filter(str.isdigit, CompOut_layer['inputs'][0]['dtype'])))
            self.bw_filter = int(''.join(filter(str.isdigit, CompOut_layer['inputs'][1]['dtype'])))
            self.bw_ofmap = int(''.join(filter(str.isdigit, CompOut_layer['outputs'][0]['dtype'])))
            self.bw_psum = self.bw_ofmap  
            if layer_op == "conv_bias":
                self.bw_bias = int(''.join(filter(str.isdigit, CompOut_layer['inputs'][2]['dtype'])))
            else:
                self.bw_bias = self.bw_ofmap   # training conv layer do not have any bias, but systolic array need a bias to work, hence setting it to the bw_ofmap

        ################################################GEMM means fully connected layers
        elif layer_op == "gemm" or layer_op == "gemm_no_bias":
            self.Layer_name = "GEMM"
            ##### Dimensions of the layer tensors
            #print(CompOut_layer['iterable_dimensions'])
            self.OW = 1
            self.OH = 1
            self.OC = CompOut_layer['iterable_dimensions']['P']
            self.KW = 1
            self.KH = 1
            self.IC = CompOut_layer['iterable_dimensions']['N']
            self.Batch = CompOut_layer['iterable_dimensions']['M']
            self.Stride = 1
            #self.Pad = 0            
            self.IW = (self.OW - 1) * self.Stride + self.KW   #this will autometically give the padded ifmap height & width
            self.IH = (self.OH - 1) * self.Stride + self.KH

            
            ######Tiling parameters from DRAM to SRAM level 
            #print("weight:", CompOut_layer['inputs'][1]['tiling']['WBUF'])    # index 1 is weight in the list under outputs
            #print("ofmap:", CompOut_layer['outputs'][0]['tiling']['OBUF'])    # index 0 is ofmap in the list under outputs
            self.DTile_ow = 1
            self.DTile_oh = 1
            self.DTile_oc = CompOut_layer['outputs'][0]['tiling']['OBUF']['P']
            self.DTile_kw = 1
            self.DTile_kh = 1
            self.DTile_ic = CompOut_layer['inputs'][1]['tiling']['WBUF']['N']
            self.DTile_batch = CompOut_layer['outputs'][0]['tiling']['OBUF']['M']
            self.DTile_iw = (self.DTile_ow - 1) * self.Stride + self.DTile_kw
            self.DTile_ih = (self.DTile_oh - 1) * self.Stride + self.DTile_kh

            #Tiling parameters from SRAM to PE level
            #print("ifmap:", CompOut_layer['inputs'][0]['tiling']['pe_array'])    # index 0 is ifmap in the list under inputs
            #print("weight:", CompOut_layer['inputs'][1]['tiling']['pe_array'])    # index 1 is weight in the list under inputs
            #print("ofmap:", CompOut_layer['outputs'][0]['tiling']['pe_array'])    # index 0 is ofmap in the list under outputs
            self.Stile_ow = 1
            self.Stile_oh = 1
            self.Stile_oc = CompOut_layer['outputs'][0]['tiling']['pe_array']['P']
            self.Stile_kw = 1
            self.Stile_kh = 1
            self.Stile_ic = CompOut_layer['inputs'][1]['tiling']['pe_array']['N']
            self.Stile_batch = CompOut_layer['outputs'][0]['tiling']['pe_array']['M']
            self.Stile_iw = 1
            self.Stile_ih = 1
            
            #### Extracting loop order 
            #print(CompOut_layer['iterable_dimensions'].keys())
            loop_keys = list(CompOut_layer['iterable_dimensions'].keys())
            loop_keys.reverse()     # in simulator code, the first one in the list is the innermost loop (opposite to the compiler output)
            #print(loop_keys)
            key_conversion_dict = {'P':'oc', 'N':'ic', 'M':'n'}
            conv_loop_keys = []  
            for key in loop_keys:
                conv_loop_keys.append(key_conversion_dict[key])
            self.Loop_order = conv_loop_keys  #converted loop key
            #print("Loop order:", self.Loop_order)
            #self.Loop_order = ['ow', 'oh', 'kw', 'kh', 'ic', 'n', 'oc']   # the first one, i.e., ow is the inner most loop and the last one i.e., oc is the outermost loop

            #### Extracting fusion info. 
            #print(CompOut_layer['inputs'][0]['data_path'][0])
            #print(CompOut_layer['outputs'][0]['data_path'][-1])
            # for GEMM/FC layer, a nofusion means: ifmap comes from DRAM and ofmap goes to DRAM
            initial_ifmap_storage = CompOut_layer['inputs'][0]['data_path'][0]
            final_ofmap_storage = CompOut_layer['outputs'][0]['data_path'][-1]
            if initial_ifmap_storage == "DRAM" and final_ofmap_storage == "DRAM":
                self.fusion_status = "NoFusion"

            ### extracting the execution hardware based on the end location of first input 
            input_end_location = CompOut_layer['inputs'][0]['data_path'][-1]  # end location of first input
            if input_end_location == "pe_array":
                self.Exe_Hardware = "Systolic"    
            elif input_end_location == "SIMD":
                self.Exe_Hardware = "SIMD"

            #print(CompOut_layer['inputs'][0]['dtype'])
            #print(CompOut_layer['inputs'][1]['dtype'])
            #print(CompOut_layer['inputs'][2]['dtype'])
            #print(CompOut_layer['outputs'][0]['dtype'])
            self.bw_ifmap = int(''.join(filter(str.isdigit, CompOut_layer['inputs'][0]['dtype'])))
            self.bw_filter = int(''.join(filter(str.isdigit, CompOut_layer['inputs'][1]['dtype'])))
            self.bw_ofmap = int(''.join(filter(str.isdigit, CompOut_layer['outputs'][0]['dtype'])))
            self.bw_psum = self.bw_ofmap 
            if layer_op == "gemm":
                self.bw_bias = int(''.join(filter(str.isdigit, CompOut_layer['inputs'][2]['dtype'])))
            else:
                self.bw_bias = self.bw_ofmap

        ########################################################################### RelU Layer ###############
        elif layer_op == "relu":
            self.Layer_name = "ReLU"
            ##### Dimensions of the layer tensors
            #print(CompOut_layer['iterable_dimensions'])
            self.OW = CompOut_layer['iterable_dimensions']['W']
            self.OH = CompOut_layer['iterable_dimensions']['H']
            self.OC = CompOut_layer['iterable_dimensions']['C']
            self.KW = 1
            self.KH = 1
            self.IC = "None"
            self.Batch = CompOut_layer['iterable_dimensions']['N']
            self.Stride = 1
            #self.Pad = 0            
            self.IW = (self.OW - 1) * self.Stride + self.KW   #this will autometically give the padded ifmap height & width
            self.IH = (self.OH - 1) * self.Stride + self.KH
            
            ######Tiling parameters from DRAM to SRAM level 
            #print("ofmap:", CompOut_layer['outputs'][0]['tiling']['VMEM1'])    # index 0 is ofmap in the list under outputs
            self.DTile_ow = CompOut_layer['outputs'][0]['tiling']['VMEM1']['W']
            self.DTile_oh = CompOut_layer['outputs'][0]['tiling']['VMEM1']['H']
            self.DTile_oc = CompOut_layer['outputs'][0]['tiling']['VMEM1']['C']
            self.DTile_kw = 1
            self.DTile_kh = 1
            self.DTile_ic = "None"
            self.DTile_batch = CompOut_layer['outputs'][0]['tiling']['VMEM1']['N']
            self.DTile_iw = (self.DTile_ow - 1) * self.Stride + self.DTile_kw
            self.DTile_ih = (self.DTile_oh - 1) * self.Stride + self.DTile_kh

            #Tiling parameters from SRAM to SIMD level 
            #print("ifmap:", CompOut_layer['inputs'][0]['tiling']['SIMD'])    # index 0 is ifmap in the list under inputs
            #print("ofmap:", CompOut_layer['outputs'][0]['tiling']['SIMD'])    # index 0 is ofmap in the list under outputs
            self.Stile_ow = CompOut_layer['outputs'][0]['tiling']['SIMD']['W']
            self.Stile_oh = CompOut_layer['outputs'][0]['tiling']['SIMD']['H']
            self.Stile_oc = CompOut_layer['outputs'][0]['tiling']['SIMD']['C']
            self.Stile_kw = 1
            self.Stile_kh = 1
            self.Stile_ic = "None"
            self.Stile_batch = CompOut_layer['outputs'][0]['tiling']['SIMD']['N']
            self.Stile_iw = CompOut_layer['inputs'][0]['tiling']['SIMD']['W']
            self.Stile_ih = CompOut_layer['inputs'][0]['tiling']['SIMD']['H']

            #### Extracting loop order: loop order does not matter for ReLU layer
            #print(CompOut_layer['iterable_dimensions'].keys())
            loop_keys = list(CompOut_layer['iterable_dimensions'].keys())
            loop_keys.reverse()     # in simulator code, the first one in the list is the innermost loop (opposite to the compiler output)
            #print(loop_keys)
            key_conversion_dict = {'H':'oh', 'W':'ow', 'C':'oc', 'N':'n'}
            conv_loop_keys = []  
            for key in loop_keys:
                conv_loop_keys.append(key_conversion_dict[key])
            self.Loop_order = conv_loop_keys  #converted loop key
            #print("Loop order:", self.Loop_order)

            #### Extracting fusion info. 
            #print(CompOut_layer['inputs'][0]['data_path'][0])
            #print(CompOut_layer['outputs'][0]['data_path'][-1])
            # for SIMD layer, a nofusion means: ifmap comes from DRAM and ofmap goes to DRAM
            initial_ifmap_storage = CompOut_layer['inputs'][0]['data_path'][0]
            final_ofmap_storage = CompOut_layer['outputs'][0]['data_path'][-1]
            if initial_ifmap_storage == "DRAM" and final_ofmap_storage == "DRAM":
                self.fusion_status = "NoFusion"

            ### extracting the execution hardware based on the end location of first input 
            input_end_location = CompOut_layer['inputs'][0]['data_path'][-1]  # end location of first input
            if input_end_location == "pe_array":
                self.Exe_Hardware = "Systolic"    
            elif input_end_location == "SIMD":
                self.Exe_Hardware = "SIMD"

            #print(CompOut_layer['inputs'][0]['dtype'])
            #print(CompOut_layer['outputs'][0]['dtype'])
            self.bw_ifmap = int(''.join(filter(str.isdigit, CompOut_layer['inputs'][0]['dtype'])))
            self.bw_filter = "None"
            self.bw_bias = "None"
            self.bw_psum = self.bw_ifmap   # for SIMD, intermediate data bitwidth, bw_psum, is same as bw_ifmap (say, 32 bit) 
            if next_layer == "conv_bias" or next_layer == "conv" or next_layer == "gemm":
                self.bw_ofmap = Hardware_param.SAbw_ifmap  # if the next layer is in Systolic array then ofmap is quantized and so = bw_ifmap of SA operation
            else:
                self.bw_ofmap = int(''.join(filter(str.isdigit, CompOut_layer['outputs'][0]['dtype'])))
            #print("bw_ifmap:", self.bw_ifmap, "bw_ofmap:", self.bw_ofmap)


        ########################################################################### Element-wise Addition Layer ###############
        elif layer_op == "elem_add":
            self.Layer_name = "ElemAdd"
            ##### Dimensions of the layer tensors
            #print("iterable_dimensions:", CompOut_layer['iterable_dimensions'])
            self.OW = CompOut_layer['iterable_dimensions']['W']
            self.OH = CompOut_layer['iterable_dimensions']['H']
            self.OC = CompOut_layer['iterable_dimensions']['C']
            self.KW = 1
            self.KH = 1
            self.IC = "None"
            self.Batch = CompOut_layer['iterable_dimensions']['N']
            self.Stride = 1
            #self.Pad = 0            
            self.IW = (self.OW - 1) * self.Stride + self.KW   #this will autometically give the padded ifmap height & width
            self.IH = (self.OH - 1) * self.Stride + self.KH

            ######Tiling parameters from DRAM to SRAM level 
            #print("ofmap tiling:", CompOut_layer['outputs'][0]['tiling']['VMEM1'])    # index 0 is ofmap in the list under outputs
            self.DTile_ow = CompOut_layer['outputs'][0]['tiling']['VMEM1']['W']
            self.DTile_oh = CompOut_layer['outputs'][0]['tiling']['VMEM1']['H']
            self.DTile_oc = CompOut_layer['outputs'][0]['tiling']['VMEM1']['C']
            self.DTile_kw = 1
            self.DTile_kh = 1
            self.DTile_ic = "None"
            self.DTile_batch = CompOut_layer['outputs'][0]['tiling']['VMEM1']['N']
            self.DTile_iw = (self.DTile_ow - 1) * self.Stride + self.DTile_kw
            self.DTile_ih = (self.DTile_oh - 1) * self.Stride + self.DTile_kh

            #Tiling parameters from SRAM to SIMD level 
            #print("ifmap:", CompOut_layer['inputs'][0]['tiling']['SIMD'])    # index 0 is ifmap in the list under inputs
            #print("ofmap:", CompOut_layer['outputs'][0]['tiling']['SIMD'])    # index 0 is ofmap in the list under outputs
            self.Stile_ow = CompOut_layer['outputs'][0]['tiling']['SIMD']['W']
            self.Stile_oh = CompOut_layer['outputs'][0]['tiling']['SIMD']['H']
            self.Stile_oc = CompOut_layer['outputs'][0]['tiling']['SIMD']['C']
            self.Stile_kw = 1
            self.Stile_kh = 1
            self.Stile_ic = "None"
            self.Stile_batch = CompOut_layer['outputs'][0]['tiling']['SIMD']['N']
            self.Stile_iw = CompOut_layer['inputs'][0]['tiling']['SIMD']['W']
            self.Stile_ih = CompOut_layer['inputs'][0]['tiling']['SIMD']['H']

            #### Extracting loop order: loop order does not matter for Elem-Add layer
            #print(CompOut_layer['iterable_dimensions'].keys())
            loop_keys = list(CompOut_layer['iterable_dimensions'].keys())
            loop_keys.reverse()     # in simulator code, the first one in the list is the innermost loop (opposite to the compiler output)
            #print(loop_keys)
            key_conversion_dict = {'H':'oh', 'W':'ow', 'C':'oc', 'N':'n'}
            conv_loop_keys = []  
            for key in loop_keys:
                conv_loop_keys.append(key_conversion_dict[key])
            self.Loop_order = conv_loop_keys  #converted loop key
            #print("Loop order:", self.Loop_order)

            #### Extracting fusion info.
            #print(CompOut_layer['inputs'][0]['data_path'][0])
            #print(CompOut_layer['inputs'][1]['data_path'][0])
            #print(CompOut_layer['outputs'][0]['data_path'][-1])
            # for Elem-ADD layer, a nofusion means: both inputs comes from DRAM and ofmap goes to DRAM
            initial_ifmap1_storage = CompOut_layer['inputs'][0]['data_path'][0]
            initial_ifmap2_storage = CompOut_layer['inputs'][1]['data_path'][0]
            final_ofmap_storage = CompOut_layer['outputs'][0]['data_path'][-1]
            if initial_ifmap1_storage == "DRAM" and initial_ifmap2_storage == "DRAM" and final_ofmap_storage == "DRAM":
                self.fusion_status = "NoFusion"

            ### extracting the execution hardware based on the end location of first input CHECK THE VALIDITY OF THIS EXTRCATION FOR FUSION
            input_end_location = CompOut_layer['inputs'][0]['data_path'][-1]  # end location of first input
            if input_end_location == "pe_array":
                self.Exe_Hardware = "Systolic"    
            elif input_end_location == "SIMD":
                self.Exe_Hardware = "SIMD"

            print(CompOut_layer['inputs'][0]['dtype'])
            print(CompOut_layer['outputs'][0]['dtype'])
            self.bw_ifmap = int(''.join(filter(str.isdigit, CompOut_layer['inputs'][0]['dtype'])))
            self.bw_filter = "None"
            self.bw_bias = "None"
            self.bw_psum = self.bw_ifmap   # for SIMD, intermediate data bitwidth, bw_psum, is same as bw_ifmap (say, 32 bit) 
            if next_layer == "conv_bias" or next_layer == "conv" or next_layer == "gemm":
                self.bw_ofmap = Hardware_param.SAbw_ifmap  # if the next layer is in Systolic array then ofmap is quantized and so = bw_ifmap of SA operation
            else:
                self.bw_ofmap = int(''.join(filter(str.isdigit, CompOut_layer['outputs'][0]['dtype'])))
            print("bw_ifmap:", self.bw_ifmap, "bw_ofmap:", self.bw_ofmap)


        ########################################################################### MaxPool, AveragePool, and Global AveragePool Layer ###############
        elif layer_op == "max_pool" or layer_op == "avg_pool" or layer_op == "global_avg_pool":
            if (layer_op == "max_pool"):
                self.Layer_name = "MaxPool"
            else:
                self.Layer_name = "AvgPool" # the simulator treats the avg-pool and gloabl-avg-pool as one category of layer

            ##### Dimensions of the layer tensors
            #print("iterable_dimensions:", CompOut_layer['iterable_dimensions'])
            self.OW = CompOut_layer['iterable_dimensions']['OW']
            self.OH = CompOut_layer['iterable_dimensions']['OH']
            self.OC = CompOut_layer['iterable_dimensions']['C']
            self.IC = "None"
            self.Batch = CompOut_layer['iterable_dimensions']['N']

            if layer_op == "global_avg_pool":
                self.Stride = 1
                self.KW = CompOut_layer['iterable_dimensions']['IW']
                self.KH = CompOut_layer['iterable_dimensions']['IH']
            else:
                self.Stride = CompOut_layer['operation_parameters']['sx'] 
                self.KW = CompOut_layer['operation_parameters']['KW']
                self.KH = CompOut_layer['operation_parameters']['KH']
            #print("pool params:", self.Stride, self.KW, self.KH)
            #self.Pad = 0            
            self.IW = (self.OW - 1) * self.Stride + self.KW   #this will autometically give the padded ifmap height & width
            self.IH = (self.OH - 1) * self.Stride + self.KH
            
            ######Tiling parameters from DRAM to SRAM level 
            #print("ofmap:", CompOut_layer['outputs'][0]['tiling']['VMEM1'])    # index 0 is ofmap in the list under outputs
            self.DTile_ow = CompOut_layer['outputs'][0]['tiling']['VMEM2']['OW']
            self.DTile_oh = CompOut_layer['outputs'][0]['tiling']['VMEM2']['OH']
            self.DTile_oc = CompOut_layer['outputs'][0]['tiling']['VMEM2']['C']
            self.DTile_kw = self.KW  # this has to be true for global_avg_pool as well cause OW = 1, hence the only valid tiling is DTile_kw = KW = IW = DTtile_iw
            self.DTile_kh = self.KH  # to maintain the relationship between OW and IW
            self.DTile_ic = "None"
            self.DTile_batch = CompOut_layer['outputs'][0]['tiling']['VMEM2']['N']
            self.DTile_iw = (self.DTile_ow - 1) * self.Stride + self.DTile_kw
            self.DTile_ih = (self.DTile_oh - 1) * self.Stride + self.DTile_kh

            #Tiling parameters from SRAM to SIMD level
            #print("ifmap:", CompOut_layer['inputs'][0]['tiling']['SIMD'])    # index 0 is ifmap in the list under inputs
            #print("ofmap:", CompOut_layer['outputs'][0]['tiling']['SIMD'])    # index 0 is ofmap in the list under outputs
            self.Stile_ow = CompOut_layer['outputs'][0]['tiling']['SIMD']['OW']
            self.Stile_oh = CompOut_layer['outputs'][0]['tiling']['SIMD']['OH']
            self.Stile_oc = CompOut_layer['outputs'][0]['tiling']['SIMD']['C']
            self.Stile_kw = 1
            self.Stile_kh = 1
            self.Stile_ic = "None"
            self.Stile_batch = CompOut_layer['outputs'][0]['tiling']['SIMD']['N']
            self.Stile_iw = CompOut_layer['inputs'][0]['tiling']['SIMD']['IW']
            self.Stile_ih = CompOut_layer['inputs'][0]['tiling']['SIMD']['IH']

            #### Extracting loop order: 
            #print(CompOut_layer['iterable_dimensions'].keys())
            loop_keys = list(CompOut_layer['iterable_dimensions'].keys())
            loop_keys.reverse()     # in simulator code, the first one in the list is the innermost loop (opposite to the compiler output)
            #print(loop_keys)
            key_conversion_dict = {'OH':'oh', 'OW':'ow', 'C':'oc', 'N':'n', 'IH':'kh', 'IW': 'kw', 'KH':'kh', 'KW':'kw'}
            conv_loop_keys = []  
            for key in loop_keys:
                conv_loop_keys.append(key_conversion_dict[key])
            # this part is for max and avgpool layer where kh and kw loops are not given in the iterable dimension
            #print(conv_loop_keys)
            if 'kw' not in conv_loop_keys:
                conv_loop_keys.insert(2, 'kw')
            if 'kh' not in conv_loop_keys:
                conv_loop_keys.insert(3, 'kh')
            self.Loop_order = conv_loop_keys  #converted loop key
            #print("Loop order:", self.Loop_order)

            #### Extracting fusion info. 
            #print(CompOut_layer['inputs'][0]['data_path'][0])
            #print(CompOut_layer['outputs'][0]['data_path'][-1])
            # for SIMD layer, a nofusion means: ifmap comes from DRAM and ofmap goes to DRAM
            initial_ifmap_storage = CompOut_layer['inputs'][0]['data_path'][0]
            final_ofmap_storage = CompOut_layer['outputs'][0]['data_path'][-1]
            if initial_ifmap_storage == "DRAM" and final_ofmap_storage == "DRAM":
                self.fusion_status = "NoFusion"

            ### extracting the execution hardware based on the end location of first input 
            input_end_location = CompOut_layer['inputs'][0]['data_path'][-1]  # end location of first input
            if input_end_location == "pe_array":
                self.Exe_Hardware = "Systolic"    
            elif input_end_location == "SIMD":
                self.Exe_Hardware = "SIMD"

            #print(CompOut_layer['inputs'][0]['dtype'])
            #print(CompOut_layer['outputs'][0]['dtype'])
            self.bw_ifmap = int(''.join(filter(str.isdigit, CompOut_layer['inputs'][0]['dtype'])))
            self.bw_filter = "None"
            self.bw_bias = "None"
            self.bw_psum = self.bw_ifmap   # for SIMD, intermediate data bitwidth, bw_psum, is same as bw_ifmap (say, 32 bit) 
            if next_layer == "conv_bias" or next_layer == "conv" or next_layer == "gemm":
                self.bw_ofmap = Hardware_param.SAbw_ifmap  # if the next layer is in Systolic array then ofmap is quantized and so = bw_ifmap of SA operation
            else:
                self.bw_ofmap = int(''.join(filter(str.isdigit, CompOut_layer['outputs'][0]['dtype'])))
            #print("bw_ifmap:", self.bw_ifmap, "bw_ofmap:", self.bw_ofmap)
    

        ########################################################### Softmax Layer ##################  (compiler output not integrated yet for softmax)
        elif layer_op == "softmax":
            self.Layer_name = "Softmax"
            ##### Dimensions of the layer tensors
            #print(CompOut_layer['iterable_dimensions'])
            self.OW = 1
            self.OH = 1
            self.OC = 1000
            self.KW = 1
            self.KH = 1
            self.IC = "None"
            self.Batch = 128
            self.Stride = 1
            #self.Pad = 0            
            self.IW = (self.OW - 1) * self.Stride + self.KW   #this will autometically give the padded ifmap height & width
            self.IH = (self.OH - 1) * self.Stride + self.KH

            
            ######Tiling parameters from DRAM to SRAM level 
            #print("weight:", CompOut_layer['inputs'][1]['tiling']['WBUF'])    # index 1 is weight in the list under outputs
            #print("ofmap:", CompOut_layer['outputs'][0]['tiling']['OBUF'])    # index 0 is ofmap in the list under outputs
            self.DTile_ow = 1
            self.DTile_oh = 1
            self.DTile_oc = 64
            self.DTile_kw = 1
            self.DTile_kh = 1
            self.DTile_ic = "None"
            self.DTile_batch = 64
            self.DTile_iw = (self.DTile_ow - 1) * self.Stride + self.DTile_kw
            self.DTile_ih = (self.DTile_oh - 1) * self.Stride + self.DTile_kh

            #Tiling parameters from SRAM to PE level 
            #print("ifmap:", CompOut_layer['inputs'][0]['tiling']['pe_array'])    # index 0 is ifmap in the list under inputs
            #print("weight:", CompOut_layer['inputs'][1]['tiling']['pe_array'])    # index 1 is weight in the list under inputs
            #print("ofmap:", CompOut_layer['outputs'][0]['tiling']['pe_array'])    # index 0 is ofmap in the list under outputs
            self.Stile_ow = 1
            self.Stile_oh = 1
            self.Stile_oc = 1
            self.Stile_kw = 1
            self.Stile_kh = 1
            self.Stile_ic = "None"
            self.Stile_batch = Hardware_param.SysArray_col
            self.Stile_iw = 1
            self.Stile_ih = 1
            
            #### Extracting loop order 
            self.Loop_order = "DontCare"  

            #### Extracting fusion info. 
            self.fusion_status = "NoFusion"

            ### extracting the execution hardware based on the end location of first input 
            self.Exe_Hardware = "SIMD"

            #print(CompOut_layer['inputs'][0]['dtype'])
            #print(CompOut_layer['inputs'][1]['dtype'])
            #print(CompOut_layer['inputs'][2]['dtype'])
            #print(CompOut_layer['outputs'][0]['dtype'])
            self.bw_ifmap = 32
            self.bw_filter = "None"
            self.bw_bias = "None"
            self.bw_ofmap = 8
            self.bw_psum = 32  

        ########################################################### ROIAlign Layer ##################  (compiler output not integrated yet for RoIAlign)
        elif layer_op == "roialign":
            self.Layer_name = "ROIAlignPool"
            ## the Level projection part specific for FPN backbone is omitting now, later will add that either in the same function or as a seperate operation
            ##### Dimensions of the layer tensors; The parameters are the output dimensions after RoIAlign pool operation
            #print(CompOut_layer['iterable_dimensions'])
            self.OW = 7
            self.OH = 7
            self.OC = 256
            self.KW = "None"
            self.KH = "None"
            self.IC = "None"
            self.Batch = 4
            self.Stride = "None"
            #self.Pad = 0            
            self.IW = "DontCare"   
            self.IH = "DontCare"
            self.RoI = 1000   # additional parameter

            ######Tiling parameters from DRAM to SRAM level 
            #print("weight:", CompOut_layer['inputs'][1]['tiling']['WBUF'])    # index 1 is weight in the list under outputs
            #print("ofmap:", CompOut_layer['outputs'][0]['tiling']['OBUF'])    # index 0 is ofmap in the list under outputs
            self.DTile_ow = 4
            self.DTile_oh = 4
            self.DTile_oc = 64
            self.DTile_kw = "None"
            self.DTile_kh = "None"
            self.DTile_ic = "None"
            self.DTile_batch = 1   #batch serialized, so batch tile has to be 1
            self.DTile_iw = "DontCare"
            self.DTile_ih = "DontCare"
            self.DTile_roi = 1     #roi serialized, so roi tile has to be 1

            #Tiling parameters from SRAM to PE level 
            #print("ifmap:", CompOut_layer['inputs'][0]['tiling']['pe_array'])    # index 0 is ifmap in the list under inputs
            #print("weight:", CompOut_layer['inputs'][1]['tiling']['pe_array'])    # index 1 is weight in the list under inputs
            #print("ofmap:", CompOut_layer['outputs'][0]['tiling']['pe_array'])    # index 0 is ofmap in the list under outputs
            self.Stile_ow = 1
            self.Stile_oh = 1
            self.Stile_oc = Hardware_param.SysArray_col
            self.Stile_kw = "None"
            self.Stile_kh = "None"
            self.Stile_ic = "None"
            self.Stile_batch = 1        #batch serialized, so batch tile has to be 1
            self.Stile_iw = "DontCare"
            self.Stile_ih = "DontCare"
            self.Stile_roi = 1          #roi serialized, so roi tile has to be 1
            
            #### Extracting loop order 
            self.Loop_order = "DontCare"  

            #### Extracting fusion info. 
            self.fusion_status = "NoFusion"

            ### extracting the execution hardware based on the end location of first input 
            self.Exe_Hardware = "SIMD"

            #print(CompOut_layer['inputs'][0]['dtype'])
            #print(CompOut_layer['inputs'][1]['dtype'])
            #print(CompOut_layer['inputs'][2]['dtype'])
            #print(CompOut_layer['outputs'][0]['dtype'])
            self.bw_ifmap = 32
            self.bw_filter = "None"
            self.bw_bias = "None"
            self.bw_ofmap = 8
            self.bw_psum = 32


        ##########################################The common layer object for some training operations ##################  
        elif layer_op == "sgd4d" or layer_op == "sgd2d" or layer_op == "sgd1d" or layer_op == "relu_grad" or layer_op == "elem_add_grad" or layer_op == "reduce_sum":
            self.Layer_name = "Common_SIMD_Backward"
            self.CompilerOut_layer = CompOut_layer   #passing the full compiler output of the layer with the layer object for the common SIMD function

            self.fusion_status = "NoFusion"
            self.Exe_Hardware = "SIMD"

            #print(CompOut_layer['inputs'][0]['dtype'])
            #print(CompOut_layer['outputs'][0]['dtype'])
            self.bw_ifmap = int(''.join(filter(str.isdigit, CompOut_layer['inputs'][0]['dtype'])))
            self.bw_filter = "None"
            self.bw_bias = "None"
            self.bw_psum = self.bw_ifmap   # for SIMD, intermediate data bitwidth, bw_psum, is same as bw_ifmap (say, 32 bit) 
            if next_layer == "conv_bias" or next_layer == "conv" or next_layer == "gemm" or next_layer == "gemm_no_bias":
                self.bw_ofmap = Hardware_param.SAbw_ifmap  # if the next layer is in Systolic array then ofmap is quantized and so = bw_ifmap of SA operation
            else:
                self.bw_ofmap = int(''.join(filter(str.isdigit, CompOut_layer['outputs'][0]['dtype'])))
            #print("bw_ifmap:", self.bw_ifmap, "bw_ofmap:", self.bw_ofmap)

        ######################################### Batch Norm Forward Pass ##################  
        elif layer_op == "batch_norm":
            self.Layer_name = "BatchNorm"
            self.CompilerOut_layer = CompOut_layer   #passing the full compiler output of the layer with the layer object to apply batch norm equations
            self.fusion_status = "NoFusion"
            self.Exe_Hardware = "SIMD"

            #print(CompOut_layer['inputs'][0]['dtype'])
            #print(CompOut_layer['outputs'][0]['dtype'])
            self.bw_ifmap = int(''.join(filter(str.isdigit, CompOut_layer['inputs'][0]['dtype'])))
            self.bw_filter = "None"
            self.bw_bias = "None"
            self.bw_psum = self.bw_ifmap   # for SIMD, intermediate data bitwidth, bw_psum, is same as bw_ifmap (say, 32 bit) 
            if next_layer == "conv_bias" or next_layer == "conv" or next_layer == "gemm" or next_layer == "gemm_no_bias":
                self.bw_ofmap = Hardware_param.SAbw_ifmap  # if the next layer is in Systolic array then ofmap is quantized and so = bw_ifmap of SA operation
            else:
                self.bw_ofmap = int(''.join(filter(str.isdigit, CompOut_layer['outputs'][0]['dtype'])))
            #print("bw_ifmap:", self.bw_ifmap, "bw_ofmap:", self.bw_ofmap)

        ######################################### Batch Norm Backward Pass ##################  
        elif layer_op == "batchnorm_grad":
            self.Layer_name = "BatchNorm_Backward"
            self.CompilerOut_layer = CompOut_layer   #passing the full compiler output of the layer with the layer object to apply batch norm equations
            self.fusion_status = "NoFusion"
            self.Exe_Hardware = "SIMD"

            #print(CompOut_layer['inputs'][0]['dtype'])
            #print(CompOut_layer['outputs'][0]['dtype'])
            self.bw_ifmap = int(''.join(filter(str.isdigit, CompOut_layer['inputs'][0]['dtype'])))
            self.bw_filter = "None"
            self.bw_bias = "None"
            self.bw_psum = self.bw_ifmap   # for SIMD, intermediate data bitwidth, bw_psum, is same as bw_ifmap (say, 32 bit) 
            if next_layer == "conv_bias" or next_layer == "conv" or next_layer == "gemm" or next_layer == "gemm_no_bias":
                self.bw_ofmap = Hardware_param.SAbw_ifmap  # if the next layer is in Systolic array then ofmap is quantized and so = bw_ifmap of SA operation
            else:
                self.bw_ofmap = int(''.join(filter(str.isdigit, CompOut_layer['outputs'][0]['dtype'])))
            #print("bw_ifmap:", self.bw_ifmap, "bw_ofmap:", self.bw_ofmap)
        
        else:
            # This else condition is for the training layers for which the simualtor do not have a model yet
            self.Layer_name = layer_op + "_ignored"
            self.Exe_Hardware = "SIMD"
        






