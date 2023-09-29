# Main Top file for the simulator

import logging
import math
import numpy as np
from simulation import simulate
import json
from pprint import pprint
from pathlib import Path
import csv
from layer_object import TilingFlags

############# Simulator Inputs
# For a DNN benchmark, the simulator takes two files as input: 
# (1): a .json file containing the hardware specifications
# (2): a .json file containing the DNN specifications
# examples of the hardware .json file format are in genesys/SimDIT/Hardware_Json_Inference/ and genesys/SimDIT/Hardware_Json_Training/ directories
# examples of the DNN specifications .json file format are in genesys/SimDIT/DNN_Spec_Inference/ and genesys/SimDIT/DNN_Spec_Training/ directories

############# Simulator Outputs
# (1): a .csv file containing the layer-wise breakdown of the performance statistics of a benchmark DNN (automatically saved in the working directory upon simulation)
# (2): a .csv file containing the performance statistics for the full DNN benchmark
# the output results from simulation are saved in genesys/SimDIT/Results_Inference/ and genesys/SimDIT/Results_Training/ directories

# Performance statistics provided by the simulator (for training the statistics are for a single iteration):
# (i) #of accesses for all the on-chip buffers for all the datatypes (in KB)
# (ii) #of accesses for the off-chip DRAM for all the data types (in KB)
# (iii) #of various arithmatic operations
# (iv) #of compute cycles
# (v) #of stall cycles
# (vi) total cycle counts


####### The following part of code feeds the simulator with example hardware & DNN specification pair

#Simulation_Phase is either "Inference" or "Training"
Simulation_Phase =  "Inference" 
#Simulation_Phase =  "Training"

DNN_benchmark = "ResNet50_"    #current DNN Spec directories contain example of DNN specification for ResNet50 and ResNet18
#DNN_benchmark = "ResNet18_"

hardware_directory_name = "Hardware_Json_" + Simulation_Phase + "/"
dnn_spec_dicrectory_name = "DNN_Spec_" + Simulation_Phase + "/"
result_directory_name = "Results_" + Simulation_Phase + "/"

# Various directory locations for inference/Training (SET THE CORRECT DIRECTORY PATH HERE BEFORE SIMULATION)
Hardware_directory = Path("/genesys/SimDIT/" + hardware_directory_name)
DNNSpec_directory = Path("/genesys/SimDIT/" + dnn_spec_dicrectory_name)
Result_directory = Path("/genesys/SimDIT/" + result_directory_name)


Result_header = ['Configuration name', 'WBUF access', 'IBUF access', 'OBUF access', 'BBUF access', 'VMEM access', 'IMM access', 'InsMem access', \
									   'DRAM access filter', 'DRAM access ifmap', 'DRAM access psum', 'DRAM access ofmap', 'DRAM access bias', 'total DRAM access', \
									   'SA compute cycles', 'SA stall cycles', 'SIMD compute cycles', 'SIMD stall cycles', 'total cycles', 'Op count']

SA_SIMD_header = ['Configuration name', 'WBUF access', 'IBUF access', 'OBUF access', 'BBUF access', 'total SRAM access SA', \
										'VMEM access', 'IMM access', 'InsMem access', 'total SRAM access SIMD', \
									   	'total DRAM access SA', 'total DRAM access SIMD', \
									    'SA compute cycles', 'SA stall cycles', 'total cycles SA', \
									    'SIMD compute cycles', 'SIMD stall cycles', 'total cycles SIMD', \
									    'Op count SA', 'Op count SIMD']

Optimal_WS_loop = True  # this flag is to bypass the loop order in the DNN spec file and to perform simulation using optimal weight stationary loop order for the convolution layers

######### Set the TGflagAll = True to use the internal tiling generator. For more detail see the TilingFlags object in layer_object.py
# set the batch size to use. This batch size is used by the internal tiling generator. Training should be run with a batch size greater than 1
set_batch_size = 1  
TGflagAll = True  
TGflag = TilingFlags(TGflagAll, set_batch_size)

Batch_size = set_batch_size  

if Batch_size == 1:
		Batching = "SingleB_"
elif Batch_size > 1:
		Batching = "MultiB_"


#######Formatting depending on the simulation phase
if Simulation_Phase == "Inference":
	Compute_Phase =  "Inference_"
	#hardware_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '11', '12', '13', '14']  #list of design hardware for inference
	hardware_list = ['01'] # to run one hardware design point

	#name of the stored result file for all inference design points with the same batch size
	Result_file_name = DNN_benchmark + "result_Inference_" + Batching +  str(Batch_size) +".csv"
	#print(Result_file_name)
	SA_SIMD_file_name = DNN_benchmark + "SA_SIMD_Inference_" + Batching +  str(Batch_size) +".csv"

elif Simulation_Phase == "Training":
	Compute_Phase =  "Train_"
	#hardware_list = ['01', '02', '05', '06', '08', '09', '11', '12']  #list of design hardware for Training
	hardware_list = ['01']  #to run one hardware design point
	
	#name of the stored result file for all training design points
	Result_file_name = DNN_benchmark + "result_Training_single_iteration.csv"
	SA_SIMD_file_name = DNN_benchmark + "SA_SIMD_Training_single_iteration.csv"


####### Executing simulator for inference or training (single iteration)
with open(Result_directory/Result_file_name, "w") as csvFile1, open(Result_directory/SA_SIMD_file_name, "w") as csvFile2: 
	writer1 = csv.writer(csvFile1)
	writer1.writerow(Result_header) 

	writer2 = csv.writer(csvFile2)
	writer2.writerow(SA_SIMD_header) 

	for hrd_design_no in hardware_list:
		HD_json_name = "genesys_params_design" + hrd_design_no + '.json'

		if TGflagAll == True:    #Using the internal tiling generator
			if Simulation_Phase == "Inference":
				dnn_spec_file_name = DNN_benchmark + "inference_Spec_Locked.json"  
			elif Simulation_Phase == "Training":
				dnn_spec_file_name = DNN_benchmark + "training_Spec_Locked.json"
		elif TGflagAll == False:
			print("provide external dnn spec and tiling in the DNN specifications .json file format")
			DNNSpec_file_name = DNN_benchmark + Compute_Phase + Batching + "Design" + hrd_design_no
			dnn_spec_file_name = "external_" + DNNSpec_file_name + ".json"

					
		# Loading the hardware config file
		with open(Hardware_directory/HD_json_name) as f2:
			Hardware_config = json.load(f2)
		#print(Hardware_config)

		# Loading the dnn spec file
		with open(DNNSpec_directory/dnn_spec_file_name) as f:
			DNNSpecNet = json.load(f)
		#print(json.dumps(DNNSpecNet, indent = 4))

		## All DATA Access Results are in KB
		Final_result, SA_SIMD_result_net = simulate(DNNSpecNet, Hardware_config, Optimal_WS_loop, TGflag)
		print("Final_result:", Final_result)

		# name of the design point in the result file
		result_design_point_name = DNN_benchmark + Compute_Phase + Batching + "Design" + hrd_design_no

		DNN_mode_hardware = np.array([result_design_point_name])
		DNN_result = np.hstack([ DNN_mode_hardware , Final_result])

		print(HD_json_name)
		print(dnn_spec_file_name)
		print(result_design_point_name)

		writer1.writerow(DNN_result)

		#writing the SA-SIMD breakdown results
		DNN_result_SA_SIMD = np.hstack([ DNN_mode_hardware , SA_SIMD_result_net])
		writer2.writerow(DNN_result_SA_SIMD)






		




