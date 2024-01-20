### This file contains the parameters associated to all the layers

import logging
import math
import numpy as np
from data_objects import HardwareObject
from tiling_generator import LayerSpecTemplate, generate_tile_relu, generate_tile_elemadd, generate_tile_pool, generate_tile_conv, generate_tile_gemm
from tiling_generator import generate_tile_mean_istd, generate_tile_batchnorm_forward, generate_tile_batchnorm_backward, generate_tile_comnSIMDback
from tiling_generator import generate_tile_maxpool_grad, generate_tile_avgpool_grad

class TilingFlags(object):
    # This object sets the flags which decide whether to use the externally generated tiling (i.e., from a compiler) or tiling generated from the internal tiling generator
    # There are flag variable for each inference and training layer
    def __init__(self, TGflagAll, set_batch_size):
        # True: (i) use tiling generated from the internal tiling generator, 
        #       (ii) use bitwidth parameters from the hardware object instead of the DNN spec file
        #       (iii) use batch dimension from the outside top file
        #       (iv) layer specifications and reamining other parameters are used from the DNN spec file
        # False: use tiling & all other data from the DNN spec file

        # IMPORTANT NOTE: 
        # The tiling flag can be set for each layer individually. This means for a single execution point: layer x can use tiling generator while 
        # layer y can use the tiling from the DNN spec file. Therefore, during such execution when tiling is used both from tiling generator and DNN spec file, make sure 
        # the batch size is same in both cases. (i.e., in the DNN spec file and in the top file where the batch size is given)

        # Batch Override 
        self.Batch_override = set_batch_size   # this variable will override the batch size read from the DNN spec file while using the tiling generator

        # Inference + Training layers
        self.convolution = TGflagAll   # for both conv_bias and conv
        self.gemm_fc = TGflagAll      # for both gemm and gemm_no_bias
        self.relu = TGflagAll
        self.elem_add = TGflagAll
        self.max_pool = TGflagAll
        self.avg_pool = TGflagAll

        # Training only layers
        self.sgd1d = TGflagAll
        self.sgd2d = TGflagAll
        self.sgd4d = TGflagAll 
        self.relu_grad = TGflagAll
        self.reduce_sum = TGflagAll
        self.mean_var = TGflagAll
        self.batch_norm = TGflagAll
        self.batchnorm_grad = TGflagAll
        self.max_pool_grad = TGflagAll
        self.avg_pool_grad = TGflagAll


class LayerObject(object): 
    # This object stores all the parameters associated to a layer. This includes layer dimension, tiling info, fusion status etc
    # This object takes DNN spec associated with each layer and the hardware object as input arguments
    def __init__(self, Hardware_param, CompOut_layer, next_layer, Optimal_WS_loop, TGflag, fusion_flag, Mode):
        # Ecah type of layer is read under seperate if condition. the dictionary keys and number of variables for different kind of layers are diferent in DNN spec
        layer_op = CompOut_layer['operation']
        print("layer operation:", layer_op)

        #For layers where IC = OC in the function code, used OC parameters and IC paramters are ignored (used "None") in the simulator model
        #For example, for ReLU, IC, DTile_ic, STile_ic etc will not be used in their model's code (used "None" for these parameters)
        #For layers with no kernel, default is set as KH = KW = 1, Stride = 1, Pad = 0, none of these parameters are used. IH and IW are also not used for such layers
        #for SIMD, Technically, I do not need bw_psum since ifmap and psum (intermediate data) are same bitwidth for SIMD, but keeping it seperate if needed in future
        #for SIMD, psum means intermediate data and using it for intermediate DARM access if any, for example for softmax
        #for pooling do not use ic in the loop order, use only oc
        #use Conv/FC activation bitwidth from Hardware Object as SIMD ofmap bitwidth depending on whether the SIMD layer is before a conv or not

        self.fusion_flag = fusion_flag

        ########################################################################### Convolution Layer ######################################################
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
            
            print("TGflag.convolution:", TGflag.convolution)
            if TGflag.convolution == False:
                print("Using compiler output")
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

                #print(CompOut_layer['inputs'][0]['dtype'])
                #print(CompOut_layer['inputs'][1]['dtype'])
                #print(CompOut_layer['inputs'][2]['dtype'])
                #print(CompOut_layer['outputs'][0]['dtype'])
                self.bw_ifmap = int(''.join(filter(str.isdigit, CompOut_layer['inputs'][0]['dtype'])))
                self.bw_filter = int(''.join(filter(str.isdigit, CompOut_layer['inputs'][1]['dtype'])))
                self.bw_ofmap = int(''.join(filter(str.isdigit, CompOut_layer['outputs'][0]['dtype'])))
                self.bw_psum = self.bw_ofmap  #for conv, Sys Array has no quantization unit and hence bw_psum is same as bw_ofmap (say, 32 bit) 
                if layer_op == "conv_bias":
                    self.bw_bias = int(''.join(filter(str.isdigit, CompOut_layer['inputs'][2]['dtype'])))
                else:
                    self.bw_bias = self.bw_ofmap   # training conv layer do not have any bias, but systolic array need a bias to work, hence setting it to the bw_ofmap

            elif TGflag.convolution == True:  
                print("Using tiling generator")
                if Mode == "inference":
                    print("Running Inference: using network graph from compiler generated with any batch size")
                    # directly override the batch dimension
                    self.Batch = TGflag.Batch_override
                elif Mode == "training":
                    print("Running Training: using network graph from compiler generated with batch size = 1")
                    # use N == 1 logic to figure out which dimension to replace with the override batch data
                    if self.Batch == 1:   # this is for the forward pass conv or delL/delx computation conv
                        self.Batch = TGflag.Batch_override
                    else:
                        self.IC = TGflag.Batch_override  # this is for the delL/delw computation conv where IC diemnsion holds the value of batch dimension
                
                LSpec = LayerSpecTemplate()
                LSpec.OW, LSpec.OH, LSpec.OC, LSpec.KW, LSpec.KH, LSpec.IC, LSpec.Batch, LSpec.Stride, LSpec.IW, LSpec.IH = \
                                                            self.OW, self.OH, self.OC, self.KW, self.KH, self.IC, self.Batch, self.Stride, self.IW, self.IH 
                
                #print("LSpec before tiling_generator call:", vars(LSpec))
                generate_tile_conv(LSpec, Hardware_param)                
                #print("Padded OC:", LSpec.OC, "Final DTile_ow:", LSpec.DTile_ow)
                self.OC = LSpec.OC  # overriding the original OC, IC with padded OC, IC
                self.IC = LSpec.IC

                self.DTile_ow, self.DTile_oh, self.DTile_oc, self.DTile_kw, self.DTile_kh, self.DTile_ic, self.DTile_batch, self.DTile_iw, self.DTile_ih = \
                    LSpec.DTile_ow, LSpec.DTile_oh, LSpec.DTile_oc, LSpec.DTile_kw, LSpec.DTile_kh, LSpec.DTile_ic, LSpec.DTile_batch, LSpec.DTile_iw, LSpec.DTile_ih

                self.Stile_ow, self.Stile_oh, self.Stile_oc, self.Stile_kw, self.Stile_kh, self.Stile_ic, self.Stile_batch, self.Stile_iw, self.Stile_ih = \
                    LSpec.Stile_ow, LSpec.Stile_oh, LSpec.Stile_oc, LSpec.Stile_kw, LSpec.Stile_kh, LSpec.Stile_ic, LSpec.Stile_batch, LSpec.Stile_iw, LSpec.Stile_ih
                
                self.bw_ifmap = Hardware_param.SAbw_ifmap
                self.bw_filter = Hardware_param.SAbw_filter
                self.bw_psum = Hardware_param.SAbw_psum
                self.bw_ofmap = Hardware_param.SAbw_ofmap  
                self.bw_bias = Hardware_param.SAbw_bias   
            
            #### Extracting loop order
            #print("Optimal_WS_loop:", Optimal_WS_loop)
            if Optimal_WS_loop == False:
                #print(CompOut_layer['iterable_dimensions'].keys())
                loop_keys = list(CompOut_layer['iterable_dimensions'].keys())
                loop_keys.reverse()     # in simulator code, the first one in the list is the innermost loop (opposite to the DNN spec file)
                #print(loop_keys)
                loop_keys_lower = [key.lower() for key in loop_keys]   # in simulator code, all the loop keys are lower case 
                self.Loop_order = loop_keys_lower            
            elif Optimal_WS_loop == True:
                self.Loop_order = ['ow', 'oh', 'n', 'kw', 'kh', 'ic', 'oc'] # the first one, i.e., ow is the inner most loop and the last one i.e., oc is the outermost loop
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
            

        ################################################ GEMM means fully connected layers ##############################################################
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

            
            print("TGflag.gemm_fc:", TGflag.gemm_fc)
            if TGflag.gemm_fc == False:
                print("Using compiler output")
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

                #print(CompOut_layer['inputs'][0]['dtype'])
                #print(CompOut_layer['inputs'][1]['dtype'])
                #print(CompOut_layer['inputs'][2]['dtype'])
                #print(CompOut_layer['outputs'][0]['dtype'])
                self.bw_ifmap = int(''.join(filter(str.isdigit, CompOut_layer['inputs'][0]['dtype'])))
                self.bw_filter = int(''.join(filter(str.isdigit, CompOut_layer['inputs'][1]['dtype'])))
                self.bw_ofmap = int(''.join(filter(str.isdigit, CompOut_layer['outputs'][0]['dtype'])))
                self.bw_psum = self.bw_ofmap  #for gemm, Sys Array has no quantization unit and hence bw_psum is same as bw_ofmap (say, 32 bit);
                if layer_op == "gemm":
                    self.bw_bias = int(''.join(filter(str.isdigit, CompOut_layer['inputs'][2]['dtype'])))
                else:
                    self.bw_bias = self.bw_ofmap
            
            elif TGflag.gemm_fc == True:
                print("Using tiling generator")
                if Mode == "inference":
                    print("Running Inference: using network graph from compiler generated with any batch size")
                    # directly override the batch dimension
                    self.Batch = TGflag.Batch_override
                elif Mode == "training":
                    print("Running Training: using network graph from compiler generated with batch size = 1")
                    # use N == 1 logic to figure out which dimension to replace with the override batch data
                    if self.Batch == 1:   # this is for the forward pass conv or delL/delx computation conv
                        self.Batch = TGflag.Batch_override
                    else:
                        self.IC = TGflag.Batch_override  # this is for the delL/delw computation conv where IC diemnsion holds the value of batch dimension
                
                LSpec = LayerSpecTemplate()
                LSpec.OW, LSpec.OH, LSpec.OC, LSpec.KW, LSpec.KH, LSpec.IC, LSpec.Batch, LSpec.Stride, LSpec.IW, LSpec.IH = \
                                                            self.OW, self.OH, self.OC, self.KW, self.KH, self.IC, self.Batch, self.Stride, self.IW, self.IH 
                
                #print("LSpec before tiling_generator call:", vars(LSpec))
                generate_tile_gemm(LSpec, Hardware_param)            
                #print("Padded OC:", LSpec.OC, "Final DTile_ow:", LSpec.DTile_ow)
                self.OC = LSpec.OC  # overriding the original OC, IC with padded OC, IC
                self.IC = LSpec.IC

                self.DTile_ow, self.DTile_oh, self.DTile_oc, self.DTile_kw, self.DTile_kh, self.DTile_ic, self.DTile_batch, self.DTile_iw, self.DTile_ih = \
                    LSpec.DTile_ow, LSpec.DTile_oh, LSpec.DTile_oc, LSpec.DTile_kw, LSpec.DTile_kh, LSpec.DTile_ic, LSpec.DTile_batch, LSpec.DTile_iw, LSpec.DTile_ih

                self.Stile_ow, self.Stile_oh, self.Stile_oc, self.Stile_kw, self.Stile_kh, self.Stile_ic, self.Stile_batch, self.Stile_iw, self.Stile_ih = \
                    LSpec.Stile_ow, LSpec.Stile_oh, LSpec.Stile_oc, LSpec.Stile_kw, LSpec.Stile_kh, LSpec.Stile_ic, LSpec.Stile_batch, LSpec.Stile_iw, LSpec.Stile_ih
                
                self.bw_ifmap = Hardware_param.SAbw_ifmap
                self.bw_filter = Hardware_param.SAbw_filter
                self.bw_psum = Hardware_param.SAbw_psum
                self.bw_ofmap = Hardware_param.SAbw_ofmap  
                self.bw_bias = Hardware_param.SAbw_bias

            
            #### Extracting loop order 
            #print(CompOut_layer['iterable_dimensions'].keys())
            if Optimal_WS_loop == False:
                loop_keys = list(CompOut_layer['iterable_dimensions'].keys())
                loop_keys.reverse()     # in simulator code, the first one in the list is the innermost loop (opposite to the DNN spec file)
                #print(loop_keys)
                key_conversion_dict = {'P':'oc', 'N':'ic', 'M':'n'}
                conv_loop_keys = []  
                for key in loop_keys:
                    conv_loop_keys.append(key_conversion_dict[key])
                self.Loop_order = conv_loop_keys  #converted loop key
            elif Optimal_WS_loop == True:
                self.Loop_order = ['n', 'ic', 'oc']
            #print("Loop order:", self.Loop_order)

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

            
        ########################################################################### RelU Layer ###########################################################
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
            self.IW = (self.OW - 1) * self.Stride + self.KW   #this will automatically give the padded ifmap height & width
            self.IH = (self.OH - 1) * self.Stride + self.KH
            
            print("TGflag.relu:", TGflag.relu)
            if TGflag.relu == False:
                print("Using compiler output")
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

            elif TGflag.relu == True:
                print("Using tiling generator")
                self.Batch = TGflag.Batch_override
                LSpec = LayerSpecTemplate()
                LSpec.OW, LSpec.OH, LSpec.OC, LSpec.KW, LSpec.KH, LSpec.IC, LSpec.Batch, LSpec.Stride, LSpec.IW, LSpec.IH = \
                                                            self.OW, self.OH, self.OC, self.KW, self.KH, self.IC, self.Batch, self.Stride, self.IW, self.IH 
                
                #print("LSpec before tiling gen call:", vars(LSpec))
                generate_tile_relu(LSpec, Hardware_param)
                #print("Padded OC:", LSpec.OC, "Final DTile_ow:", LSpec.DTile_ow)
                self.OC = LSpec.OC  # overriding the original OC with padded OC

                self.DTile_ow, self.DTile_oh, self.DTile_oc, self.DTile_kw, self.DTile_kh, self.DTile_ic, self.DTile_batch, self.DTile_iw, self.DTile_ih = \
                    LSpec.DTile_ow, LSpec.DTile_oh, LSpec.DTile_oc, LSpec.DTile_kw, LSpec.DTile_kh, LSpec.DTile_ic, LSpec.DTile_batch, LSpec.DTile_iw, LSpec.DTile_ih

                self.Stile_ow, self.Stile_oh, self.Stile_oc, self.Stile_kw, self.Stile_kh, self.Stile_ic, self.Stile_batch, self.Stile_iw, self.Stile_ih = \
                    LSpec.Stile_ow, LSpec.Stile_oh, LSpec.Stile_oc, LSpec.Stile_kw, LSpec.Stile_kh, LSpec.Stile_ic, LSpec.Stile_batch, LSpec.Stile_iw, LSpec.Stile_ih
                
                #Setting the bitwidths from the hardware_param directly
                self.bw_ifmap = Hardware_param.SAbw_psum   # psum bitwidth of SA is same as SIMD input
                self.bw_filter = "None"
                self.bw_bias = "None"
                self.bw_psum = self.bw_ifmap   
                if next_layer == "conv_bias" or next_layer == "conv" or next_layer == "gemm" or next_layer == "gemm_no_bias":
                    self.bw_ofmap = Hardware_param.SAbw_ifmap  
                else:
                    self.bw_ofmap = Hardware_param.SAbw_psum
                #print("bw_ifmap:", self.bw_ifmap, "bw_ofmap:", self.bw_ofmap)

            #### Extracting loop order: loop order does not matter for ReLU layer
            #print(CompOut_layer['iterable_dimensions'].keys())
            loop_keys = list(CompOut_layer['iterable_dimensions'].keys())
            loop_keys.reverse()     # in my code, the first one in the list is the innermost loop (opposite to the DNN spec file)
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


        ########################################################################### Element-wise Addition Layer ##################################################
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

            print("TGflag.elem_add:", TGflag.elem_add)
            if TGflag.elem_add == False:
                print("Using compiler output")
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

            elif TGflag.elem_add == True:
                print("Using tiling generator")
                self.Batch = TGflag.Batch_override
                LSpec = LayerSpecTemplate()
                LSpec.OW, LSpec.OH, LSpec.OC, LSpec.KW, LSpec.KH, LSpec.IC, LSpec.Batch, LSpec.Stride, LSpec.IW, LSpec.IH = \
                                                            self.OW, self.OH, self.OC, self.KW, self.KH, self.IC, self.Batch, self.Stride, self.IW, self.IH 
                
                #print("LSpec before tiling gen call:", vars(LSpec))
                generate_tile_elemadd(LSpec, Hardware_param)
                #print("Padded OC:", LSpec.OC, "Final DTile_ow:", LSpec.DTile_ow)
                self.OC = LSpec.OC  # overriding the original OC with padded OC

                self.DTile_ow, self.DTile_oh, self.DTile_oc, self.DTile_kw, self.DTile_kh, self.DTile_ic, self.DTile_batch, self.DTile_iw, self.DTile_ih = \
                    LSpec.DTile_ow, LSpec.DTile_oh, LSpec.DTile_oc, LSpec.DTile_kw, LSpec.DTile_kh, LSpec.DTile_ic, LSpec.DTile_batch, LSpec.DTile_iw, LSpec.DTile_ih

                self.Stile_ow, self.Stile_oh, self.Stile_oc, self.Stile_kw, self.Stile_kh, self.Stile_ic, self.Stile_batch, self.Stile_iw, self.Stile_ih = \
                    LSpec.Stile_ow, LSpec.Stile_oh, LSpec.Stile_oc, LSpec.Stile_kw, LSpec.Stile_kh, LSpec.Stile_ic, LSpec.Stile_batch, LSpec.Stile_iw, LSpec.Stile_ih
                
                #Setting the bitwidths from the hardware_param directly
                self.bw_ifmap = Hardware_param.SAbw_psum   # psum bitwidth of SA is same as SIMD input
                self.bw_filter = "None"
                self.bw_bias = "None"
                self.bw_psum = self.bw_ifmap   
                if next_layer == "conv_bias" or next_layer == "conv" or next_layer == "gemm" or next_layer == "gemm_no_bias":
                    self.bw_ofmap = Hardware_param.SAbw_ifmap  
                else:
                    self.bw_ofmap = Hardware_param.SAbw_psum
                #print("bw_ifmap:", self.bw_ifmap, "bw_ofmap:", self.bw_ofmap)

            
            #### Extracting loop order: loop order does not matter for Elem-Add layer
            #print(CompOut_layer['iterable_dimensions'].keys())
            loop_keys = list(CompOut_layer['iterable_dimensions'].keys())
            loop_keys.reverse()     # in my code, the first one in the list is the innermost loop (opposite to the DNN spec file)
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

            ### extracting the execution hardware based on the end location of first input 
            input_end_location = CompOut_layer['inputs'][0]['data_path'][-1]  # end location of first input
            if input_end_location == "pe_array":
                self.Exe_Hardware = "Systolic"    
            elif input_end_location == "SIMD":
                self.Exe_Hardware = "SIMD"

            
        ########################################################################### MaxPool, AveragePool, and Global AveragePool Layer ##########################
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
                self.KW = CompOut_layer['iterable_dimensions']['KW']
                self.KH = CompOut_layer['iterable_dimensions']['KH']
            #print("pool params:", self.Stride, self.KW, self.KH)
            #self.Pad = 0            
            self.IW = (self.OW - 1) * self.Stride + self.KW   #this will autometically give the padded ifmap height & width
            self.IH = (self.OH - 1) * self.Stride + self.KH

            print("TGflag.max_pool:", TGflag.max_pool, "TGflag.avg_pool:", TGflag.avg_pool)
            if (self.Layer_name == "MaxPool" and TGflag.max_pool == False) or (self.Layer_name == "AvgPool" and TGflag.avg_pool == False):
                print("Using compiler output")            
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
            
            elif (self.Layer_name == "MaxPool" and TGflag.max_pool == True) or (self.Layer_name == "AvgPool" and TGflag.avg_pool == True):
                print("Using tiling generator")
                self.Batch = TGflag.Batch_override
                LSpec = LayerSpecTemplate()
                LSpec.OW, LSpec.OH, LSpec.OC, LSpec.KW, LSpec.KH, LSpec.IC, LSpec.Batch, LSpec.Stride, LSpec.IW, LSpec.IH = \
                                                            self.OW, self.OH, self.OC, self.KW, self.KH, self.IC, self.Batch, self.Stride, self.IW, self.IH 
                
                #print("LSpec before tiling gen call:", vars(LSpec))
                generate_tile_pool(LSpec, Hardware_param)
                #print("Padded OC:", LSpec.OC, "Final DTile_ow:", LSpec.DTile_ow)
                self.OC = LSpec.OC  # overriding the original OC with padded OC

                self.DTile_ow, self.DTile_oh, self.DTile_oc, self.DTile_kw, self.DTile_kh, self.DTile_ic, self.DTile_batch, self.DTile_iw, self.DTile_ih = \
                    LSpec.DTile_ow, LSpec.DTile_oh, LSpec.DTile_oc, LSpec.DTile_kw, LSpec.DTile_kh, LSpec.DTile_ic, LSpec.DTile_batch, LSpec.DTile_iw, LSpec.DTile_ih

                self.Stile_ow, self.Stile_oh, self.Stile_oc, self.Stile_kw, self.Stile_kh, self.Stile_ic, self.Stile_batch, self.Stile_iw, self.Stile_ih = \
                    LSpec.Stile_ow, LSpec.Stile_oh, LSpec.Stile_oc, LSpec.Stile_kw, LSpec.Stile_kh, LSpec.Stile_ic, LSpec.Stile_batch, LSpec.Stile_iw, LSpec.Stile_ih  
                

                #Setting the bitwidths from the hardware_param directly
                self.bw_ifmap = Hardware_param.SAbw_psum   # psum bitwidth of SA is same as SIMD input
                self.bw_filter = "None"
                self.bw_bias = "None"
                self.bw_psum = self.bw_ifmap   
                if next_layer == "conv_bias" or next_layer == "conv" or next_layer == "gemm" or next_layer == "gemm_no_bias":
                    self.bw_ofmap = Hardware_param.SAbw_ifmap  
                else:
                    self.bw_ofmap = Hardware_param.SAbw_psum
                #print("bw_ifmap:", self.bw_ifmap, "bw_ofmap:", self.bw_ofmap)          
            

            #### Extracting loop order: 
            #print(CompOut_layer['iterable_dimensions'].keys())
            loop_keys = list(CompOut_layer['iterable_dimensions'].keys())
            loop_keys.reverse()     # in my code, the first one in the list is the innermost loop (opposite to the DNN spec file)
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

        
        ########################################################### Softmax Layer ##################  (DNN spec file is not integrated yet for softmax)
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

        ########################################################### ROIAlign Layer ##################  (DNN spec file is not integrated yet for RoIAlign)
        elif layer_op == "roialign":
            self.Layer_name = "ROIAlignPool"
            ## the Level projection part specific for FPN backbone is omitting now, later may add that either in the same function or as a seperate operation
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


        ########################################## The common layer object for some training operations ############################################################ 
        elif layer_op == "sgd4d" or layer_op == "sgd2d" or layer_op == "sgd1d" or layer_op == "relu_grad" or layer_op == "elem_add_grad" or layer_op == "reduce_sum":
            self.Layer_name = "Common_SIMD_Backward"

            print("TGflag.sgd4d:", TGflag.sgd4d, "TGflag.sgd2d:", TGflag.sgd2d, "TGflag.sgd1d:", TGflag.sgd1d)
            print("TGflag.relu_grad:", TGflag.relu_grad, "TGflag.reduce_sum:", TGflag.reduce_sum)

            if (layer_op == "sgd4d" and TGflag.sgd4d == False) or (layer_op == "sgd2d" and TGflag.sgd2d == False) or (layer_op == "sgd1d" and TGflag.sgd1d == False) or \
                                (layer_op == "relu_grad" and TGflag.relu_grad == False) or (layer_op == "reduce_sum" and TGflag.reduce_sum == False):

                print("Using compiler output")
                self.CompilerOut_layer = CompOut_layer   #passing the full DNN spec of the layer with the layer object for the common SIMD function

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

            elif (layer_op == "sgd4d" and TGflag.sgd4d == True) or (layer_op == "sgd2d" and TGflag.sgd2d == True) or (layer_op == "sgd1d" and TGflag.sgd1d == True) or \
                                (layer_op == "relu_grad" and TGflag.relu_grad == True) or (layer_op == "reduce_sum" and TGflag.reduce_sum == True):
                print("Using tiling generator")

                Batch_override = TGflag.Batch_override
                CompilerOut_layer_TG = generate_tile_comnSIMDback(Hardware_param, Batch_override, CompOut_layer)
                self.CompilerOut_layer = CompilerOut_layer_TG   # passing the tiling output in the DNN spec format

                #Setting the bitwidths from the hardware_param directly
                self.bw_ifmap = Hardware_param.SAbw_psum   # psum bitwidth of SA is same as SIMD input
                self.bw_filter = "None"
                self.bw_bias = "None"
                self.bw_psum = self.bw_ifmap   
                if next_layer == "conv_bias" or next_layer == "conv" or next_layer == "gemm" or next_layer == "gemm_no_bias":
                    self.bw_ofmap = Hardware_param.SAbw_ifmap  
                else:
                    self.bw_ofmap = Hardware_param.SAbw_psum
                #print("bw_ifmap:", self.bw_ifmap, "bw_ofmap:", self.bw_ofmap)

            self.fusion_status = "NoFusion"
            self.Exe_Hardware = "SIMD"

            

        ######################################### Batch Norm Forward Pass & Backward Pass ####################################################  
        #extraction of DNN spec is exactly same for mean_var, batch_norm, and batchnorm_grad. Hence using a combined code
        elif layer_op == "mean_var" or layer_op == "batch_norm" or layer_op == "batchnorm_grad_x_mu":
            if layer_op == "mean_var":
                self.Layer_name = "MeanIstd"
            elif layer_op == "batch_norm":
                self.Layer_name = "BatchNorm"
            elif layer_op == "batchnorm_grad_x_mu":
                self.Layer_name = "BatchNorm_Backward"

            #for BatchNorm, the tiling and dimension of 4D input data and the 4D output data are same
            ##### Dimensions of the layer tensors
            print("iterable_dimensions:", CompOut_layer['iterable_dimensions'])
            self.IW = CompOut_layer['iterable_dimensions']['W']
            self.IH = CompOut_layer['iterable_dimensions']['H']
            self.OC = CompOut_layer['iterable_dimensions']['C']
            self.Batch = CompOut_layer['iterable_dimensions']['N']   
            
            print("TGflag.mean_var:", TGflag.mean_var, "TGflag.batch_norm:", TGflag.batch_norm, "TGflag.batchnorm_grad:", TGflag.batchnorm_grad)
            if (self.Layer_name == "MeanIstd" and TGflag.mean_var == False) or \
                                    (self.Layer_name == "BatchNorm" and TGflag.batch_norm == False) or \
                                    (self.Layer_name == "BatchNorm_Backward" and TGflag.batchnorm_grad == False):
                
                print("Using compiler output")
                ######Tiling parameters from DRAM to SRAM level 
                print("VMEM input tiling:", CompOut_layer['inputs'][0]['tiling']['VMEM1'])    # index 0 is input data in the list under inputs
                self.DTile_iw = CompOut_layer['inputs'][0]['tiling']['VMEM1']['W']
                self.DTile_ih = CompOut_layer['inputs'][0]['tiling']['VMEM1']['H']
                self.DTile_oc = CompOut_layer['inputs'][0]['tiling']['VMEM1']['C']
                self.DTile_batch = CompOut_layer['inputs'][0]['tiling']['VMEM1']['N']

                #Tiling parameters from SRAM to SIMD level 
                print("SIMD input tiling:", CompOut_layer['inputs'][0]['tiling']['SIMD'])    
                self.Stile_iw = CompOut_layer['inputs'][0]['tiling']['SIMD']['W']
                self.Stile_ih = CompOut_layer['inputs'][0]['tiling']['SIMD']['H']
                self.Stile_oc = CompOut_layer['inputs'][0]['tiling']['SIMD']['C']
                self.Stile_batch = CompOut_layer['inputs'][0]['tiling']['SIMD']['N']

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
                print("bw_ifmap:", self.bw_ifmap, "bw_ofmap:", self.bw_ofmap)

            elif (self.Layer_name == "MeanIstd" and TGflag.mean_var == True) or \
                                        (self.Layer_name == "BatchNorm" and TGflag.batch_norm == True) or \
                                        (self.Layer_name == "BatchNorm_Backward" and TGflag.batchnorm_grad == True): 
                
                print("Using tiling generator")
                self.Batch = TGflag.Batch_override
                LSpec = LayerSpecTemplate()
                LSpec.IW, LSpec.IH, LSpec.OC, LSpec.Batch = self.IW, self.IH, self.OC, self.Batch
                
                #print("LSpec before tiling gen call:", vars(LSpec))
                if self.Layer_name == "MeanIstd":
                    generate_tile_mean_istd(LSpec, Hardware_param)
                elif self.Layer_name == "BatchNorm":
                    generate_tile_batchnorm_forward(LSpec, Hardware_param)
                elif self.Layer_name == "BatchNorm_Backward":
                    generate_tile_batchnorm_backward(LSpec, Hardware_param)

                #print("Padded OC:", LSpec.OC, "Final DTile_ow:", LSpec.DTile_ow)
                self.OC = LSpec.OC  # overriding the original OC with padded OC
                self.DTile_iw, self.DTile_ih, self.DTile_oc, self.DTile_batch = LSpec.DTile_iw, LSpec.DTile_ih, LSpec.DTile_oc, LSpec.DTile_batch
                self.Stile_iw, self.Stile_ih, self.Stile_oc, self.Stile_batch = LSpec.Stile_iw, LSpec.Stile_ih, LSpec.Stile_oc, LSpec.Stile_batch  
        
                #Setting the bitwidths from the hardware_param directly
                self.bw_ifmap = Hardware_param.SAbw_psum   # psum bitwidth of SA is same as SIMD input
                self.bw_filter = "None"
                self.bw_bias = "None"
                self.bw_psum = self.bw_ifmap   
                if next_layer == "conv_bias" or next_layer == "conv" or next_layer == "gemm" or next_layer == "gemm_no_bias":
                    self.bw_ofmap = Hardware_param.SAbw_ifmap  
                else:
                    self.bw_ofmap = Hardware_param.SAbw_psum
                #print("bw_ifmap:", self.bw_ifmap, "bw_ofmap:", self.bw_ofmap)


            #### Extracting loop order:
            #print(CompOut_layer['iterable_dimensions'].keys())
            loop_keys = list(CompOut_layer['iterable_dimensions'].keys())
            loop_keys.reverse()     # in my code, the first one in the list is the innermost loop (opposite to the DNN spec file)
            #print(loop_keys)
            key_conversion_dict = {'H':'ih', 'W':'iw', 'C':'oc', 'N':'n'}
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
            #print("fusion_status:", self.fusion_status)

            ### extracting the execution hardware based on the end location of first input 
            input_end_location = CompOut_layer['inputs'][0]['data_path'][-1]  # end location of first input
            if input_end_location == "pe_array":
                self.Exe_Hardware = "Systolic"    
            elif input_end_location == "SIMD":
                self.Exe_Hardware = "SIMD"
            #print("execution hardware:", self.Exe_Hardware)

            
        ################################################################ Backward pass for Pooling Layer ############################################################  
        #extraction of DNN spec for gradient computation of max_pool, average_pool, and global_average pool
        elif layer_op == "max_pool_grad" or layer_op == "average_pool_grad" or layer_op == "global_average_pool_grad":
            if (layer_op == "max_pool_grad"):
                self.Layer_name = "MaxPool_Grad"
            else:
                self.Layer_name = "AvgPool_Grad" # the simulator treats the avg-pool-grad and gloabl-avg-pool-grad as one category of layer

            ##### Dimensions of the layer tensors
            #print("iterable_dimensions:", CompOut_layer['iterable_dimensions'])
            self.OW = CompOut_layer['iterable_dimensions']['OW']
            self.OH = CompOut_layer['iterable_dimensions']['OH']
            self.OC = CompOut_layer['iterable_dimensions']['C']
            self.IC = "None"
            self.Batch = CompOut_layer['iterable_dimensions']['N']

            if layer_op == "global_average_pool_grad":
                self.Stride = 1
                self.KW = CompOut_layer['iterable_dimensions']['IW']
                self.KH = CompOut_layer['iterable_dimensions']['IH']
            else:
                self.Stride = CompOut_layer['operation_parameters']['sx'] 
                self.KW = CompOut_layer['iterable_dimensions']['KW']
                self.KH = CompOut_layer['iterable_dimensions']['KH']
            #self.Pad = 0            
            self.IW = (self.OW - 1) * self.Stride + self.KW   #this will autometically give the padded ifmap height & width
            self.IH = (self.OH - 1) * self.Stride + self.KH


            print("TGflag.max_pool_grad:", TGflag.max_pool_grad, "TGflag.avg_pool_grad:", TGflag.avg_pool_grad)
            if (self.Layer_name == "MaxPool_Grad" and TGflag.max_pool_grad == False) or (self.Layer_name == "AvgPool_Grad" and TGflag.avg_pool_grad == False):
                print("Using compiler output") 
                ######Tiling parameters from DRAM to SRAM level 
                #print("grad for DRAM tiling:", CompOut_layer['inputs'][1]['tiling']['VMEM1'])    # index 1 is grad in the list under inputs
                self.DTile_ow = CompOut_layer['inputs'][1]['tiling']['VMEM1']['OW']
                self.DTile_oh = CompOut_layer['inputs'][1]['tiling']['VMEM1']['OH']
                self.DTile_oc = CompOut_layer['inputs'][1]['tiling']['VMEM1']['C']
                self.DTile_kw = self.KW  # this has to be true for global_avg_pool as well cause OW = 1, hence the only valid tiling is DTile_kw = KW = IW = DTtile_iw
                self.DTile_kh = self.KH  # to maintain the relationship between OW and IW
                self.DTile_ic = "None"
                self.DTile_batch = CompOut_layer['inputs'][1]['tiling']['VMEM1']['N']
                self.DTile_iw = (self.DTile_ow - 1) * self.Stride + self.DTile_kw
                self.DTile_ih = (self.DTile_oh - 1) * self.Stride + self.DTile_kh

                #Tiling parameters from SRAM to SIMD level 
                #print("grad for SRAM tiling:", CompOut_layer['inputs'][1]['tiling']['SIMD'])    # index 1 is grad in the list under inputs
                #print("data_out for SRAM tiling:", CompOut_layer['outputs'][0]['tiling']['SIMD'])    # index 0 is data_out in the list under outputs
                self.Stile_ow = CompOut_layer['inputs'][1]['tiling']['SIMD']['OW']
                self.Stile_oh = CompOut_layer['inputs'][1]['tiling']['SIMD']['OH']
                self.Stile_oc = CompOut_layer['inputs'][1]['tiling']['SIMD']['C']
                self.Stile_kw = 1
                self.Stile_kh = 1
                self.Stile_ic = "None"
                self.Stile_batch = CompOut_layer['inputs'][1]['tiling']['SIMD']['N']
                self.Stile_iw = CompOut_layer['outputs'][0]['tiling']['SIMD']['IW']
                self.Stile_ih = CompOut_layer['outputs'][0]['tiling']['SIMD']['IH']

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

            elif (self.Layer_name == "MaxPool_Grad" and TGflag.max_pool_grad == True) or (self.Layer_name == "AvgPool_Grad" and TGflag.avg_pool_grad == True):
                print("Using tiling generator")
                self.Batch = TGflag.Batch_override
                LSpec = LayerSpecTemplate()
                LSpec.OW, LSpec.OH, LSpec.OC, LSpec.KW, LSpec.KH, LSpec.IC, LSpec.Batch, LSpec.Stride, LSpec.IW, LSpec.IH = \
                                                            self.OW, self.OH, self.OC, self.KW, self.KH, self.IC, self.Batch, self.Stride, self.IW, self.IH 
                
                #print("LSpec before tiling gen call:", vars(LSpec))
                if self.Layer_name == "MaxPool_Grad":
                    generate_tile_maxpool_grad(LSpec, Hardware_param)
                elif self.Layer_name == "AvgPool_Grad":
                    generate_tile_avgpool_grad(LSpec, Hardware_param)

                #print("Padded OC:", LSpec.OC, "Final DTile_ow:", LSpec.DTile_ow)
                self.OC = LSpec.OC  # overriding the original OC with padded OC

                self.DTile_ow, self.DTile_oh, self.DTile_oc, self.DTile_kw, self.DTile_kh, self.DTile_ic, self.DTile_batch, self.DTile_iw, self.DTile_ih = \
                    LSpec.DTile_ow, LSpec.DTile_oh, LSpec.DTile_oc, LSpec.DTile_kw, LSpec.DTile_kh, LSpec.DTile_ic, LSpec.DTile_batch, LSpec.DTile_iw, LSpec.DTile_ih

                self.Stile_ow, self.Stile_oh, self.Stile_oc, self.Stile_kw, self.Stile_kh, self.Stile_ic, self.Stile_batch, self.Stile_iw, self.Stile_ih = \
                    LSpec.Stile_ow, LSpec.Stile_oh, LSpec.Stile_oc, LSpec.Stile_kw, LSpec.Stile_kh, LSpec.Stile_ic, LSpec.Stile_batch, LSpec.Stile_iw, LSpec.Stile_ih  
                
                #Setting the bitwidths from the hardware_param directly
                self.bw_ifmap = Hardware_param.SAbw_psum   # psum bitwidth of SA is same as SIMD input
                self.bw_filter = "None"
                self.bw_bias = "None"
                self.bw_psum = self.bw_ifmap   
                if next_layer == "conv_bias" or next_layer == "conv" or next_layer == "gemm" or next_layer == "gemm_no_bias":
                    self.bw_ofmap = Hardware_param.SAbw_ifmap  
                else:
                    self.bw_ofmap = Hardware_param.SAbw_psum
                #print("bw_ifmap:", self.bw_ifmap, "bw_ofmap:", self.bw_ofmap)

            #### Extracting loop order:
            #print(CompOut_layer['iterable_dimensions'].keys())
            loop_keys = list(CompOut_layer['iterable_dimensions'].keys())
            loop_keys.reverse()     # in my code, the first one in the list is the innermost loop (opposite to the DNN spec file)
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

            #### Extracting fusion info
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

            

        else:
            # This else condition is for the training layers for which the simualtor do not have a model yet
            self.Layer_name = layer_op + "_ignored"
            self.Exe_Hardware = "SIMD"

'''
        ######################################### Batch Norm Forward Pass ##################  
        elif layer_op == "batch_norm":
            self.Layer_name = "BatchNorm"
            self.CompilerOut_layer = CompOut_layer   #passing the full DNN spec of the layer with the layer object to apply batch norm equations
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
            self.CompilerOut_layer = CompOut_layer   #passing the full DNN spec of the layer with the layer object to apply batch norm equations
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
'''

        


        

        






