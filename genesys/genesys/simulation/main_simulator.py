# Main Top file for the simulator

import logging
import math
import numpy as np
from simulation import simulate
import json
from pprint import pprint
from pathlib import Path
import csv

############# Simulator Inputs
# For a DNN benchmark, the simulator takes two files as input: 
# (1): a .json file containing the GeneSys hardware parameterization
# (2): a .json file containing the compiler output generated at the specified hardware configuration
# examples of the hardware .json file format are in genesys/simulation/Hardware_Json_Inference/ and genesys/simulation/Hardware_Json_Training/ directories
# examples of the compiler output .json file format are in genesys/simulation/Compiler_Output_Inference/ and genesys/simulation/Compiler_Output_Training/ directories

############# Simulator Outputs
# (1): a .csv file containing the layer-wise breakdown of the performance statistics of a benchmark DNN (automatically saved in the working directory upon simulation)
# (2): a .csv file containing the performance statistics for the whole DNN benchmark
# the output results from simulation are saved in genesys/simulation/Results_Inference/ and genesys/simulation/Results_Training/ directories

# Performance statistics provided by the simulator (for training the statistics are for a single iteration):
# (i) #of accesses for all the on-chip buffers for all the datatypes (in KB)
# (ii) #of accesses for the off-chip DRAM for all the data types (in KB)
# (iii) #of various arithmatic computations
# (iv) #of compute cycles
# (v) #of stall cycles
# (vi) total cycle counts


####### The following part of code feeds the simulator with example hardware & compiler output pair

#Simulation_Phase is either "Inference" or "Training"
Simulation_Phase =  "Inference" 
#Simulation_Phase =  "Training"

DNN_benchmark = "ResNet50_"    #the current compiler output directories contain example of compiler outputs for ResNet50 for the example hardware configuration points

hardware_directory_name = "Hardware_Json_" + Simulation_Phase + "/"
comp_out_dicrectory_name = "Compiler_Output_" + Simulation_Phase + "/"
result_directory_name = "Results_" + Simulation_Phase + "/"

# Various directory locations for inference/Training (SET THE CORRECT DIRECTORY PATH HERE BEFORE SIMULATION)
Hardware_directory = Path("genesys/simulation/" + hardware_directory_name)
CompilerOut_directory = Path("genesys/simulation/" + comp_out_dicrectory_name)
Result_directory = Path("genesys/simulation/" + result_directory_name)


Result_header = ['Configuration name', 'WBUF access', 'IBUF access', 'OBUF access', 'BUBF access', 'VMEM access', 'IMM access', 'InsMem access', \
									   'DARM access filter', 'DRAM access ifmap', 'DRAM access psum', 'DRAM access ofmap', 'DRAM access bias', 'total DRAM access', \
									   'SA compute cycles', 'SA stall cycles', 'SIMD compute cycles', 'SIMD stall cycles', 'total cycles', 'Op count']

Optimal_WS_loop = True  # this variable is to bypass compiler loop order and to perform simulation using optimal weight stationary loop order for the convolution layers


if Simulation_Phase == "Inference":
	Compute_Phase =  "Inference_"
	#list of example hrd_design_no in Hardware_Json_Inference directory:['01', '02', '03', '04', '05', '06', '07', '08', '09', '11', '12', '13', '14']
	hrd_design_no = '01'

	Batch_size = 1  #the example compiler outputs are for either a Batch_size of 1 or Batch size of 24576 during inference

	if Batch_size == 1:
			Batching = "SingleB_"
	elif Batch_size > 1:
			Batching = "MultiB_"

	#name of the stored result file for all inference design points with the same batch size
	Result_file_name = DNN_benchmark + "result_Inference_" + Batching +  str(Batch_size) +".csv"
	#print(Result_file_name)

	with open(Result_directory/Result_file_name, "w") as csvFile:   
		writer = csv.writer(csvFile)
		writer.writerow(Result_header) 
		
		HD_json_name = "genesys_params_design" + hrd_design_no + '.json'
		CompOutput_file_name = DNN_benchmark + Compute_Phase + Batching + "Design" + hrd_design_no
		#print(CompOutput_file_name)
		compiler_output_file_name = "compiled_" + CompOutput_file_name + "_nop" + ".json"
					
		# Loading the hardware config file
		with open(Hardware_directory/HD_json_name) as f2:
			Hardware_config = json.load(f2)
		#print(Hardware_config)

		# Loading the corresponding compiler output file
		with open(CompilerOut_directory/compiler_output_file_name) as f:
			CompilerOutput = json.load(f)
		#print(json.dumps(CompilerOutput, indent = 4))

		## All DATA Access Results are in KB
		Final_result = simulate(CompilerOutput, Hardware_config, Optimal_WS_loop)
		print("Final_result:", Final_result)

		# name of the design point in the result file
		result_design_point_name = DNN_benchmark + Compute_Phase + Batching + "Design" + hrd_design_no

		DNN_mode_hardware = np.array([result_design_point_name])
		DNN_result = np.hstack([ DNN_mode_hardware , Final_result])

		print(HD_json_name)
		print(compiler_output_file_name)
		print(result_design_point_name)

		writer.writerow(DNN_result)


elif Simulation_Phase == "Training":
	Compute_Phase =  "Train_"
	#list of example hrd_design_no in Hardware_Json_Training directory:['01', '02', '05', '06', '08', '09', '11', '12']
	hrd_design_no = '01'
	
	Batching = "MultiB_"   # the example compiler output is for a minibatch of 256 for training

	#name of the stored result file for all training design points
	Result_file_name = DNN_benchmark + "result_Training_single_iteration.csv"

	with open(Result_directory/Result_file_name, "w") as csvFile: 
		writer = csv.writer(csvFile)
		writer.writerow(Result_header) 

		HD_json_name = "genesys_params_design" + hrd_design_no + '.json'
		CompOutput_file_name = DNN_benchmark + Compute_Phase + Batching + "Design" + hrd_design_no
		#print(CompOutput_file_name)
		compiler_output_file_name = "compiled_" + CompOutput_file_name + "_nop" + ".json"
					
		# Loading the hardware config file
		with open(Hardware_directory/HD_json_name) as f2:
			Hardware_config = json.load(f2)
		#print(Hardware_config)

		# Loading the corresponding compiler output file
		with open(CompilerOut_directory/compiler_output_file_name) as f:
			CompilerOutput = json.load(f)
		#print(json.dumps(CompilerOutput, indent = 4))

		## All DATA Access Results are in KB
		Final_result = simulate(CompilerOutput, Hardware_config, Optimal_WS_loop)
		print("Final_result:", Final_result)

		# name of the design point in the result file
		result_design_point_name = DNN_benchmark + Compute_Phase + Batching + "Design" + hrd_design_no

		DNN_mode_hardware = np.array([result_design_point_name])
		DNN_result = np.hstack([ DNN_mode_hardware , Final_result])

		print(HD_json_name)
		print(compiler_output_file_name)
		print(result_design_point_name)

		writer.writerow(DNN_result)


		




