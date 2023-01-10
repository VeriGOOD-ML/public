# SimDIT: A Simulation Framework for DNN Inference and Training on ASIC Accelerator Platforms

* A comprehensive simulation framework for fast performance analysis of DNN hardware
* Models convolution and a diverse set of non-convolution operations to cover DNN inference and training

If you use any part of SimDIT for your work please cite: https://github.com/VeriGOOD-ML/public/tree/main/genesys/SimDIT

### Simulate a DNN benchmark using SimDIT
Using SimDIT, you can run a software simulation of a DNN benchmark. This directory contains all the source code of SimDIT. 

#### Inputs to SimDIT:
For a DNN benchmark, SimDIT takes two files as input: 
(1) a .json file containing the hardware specifications and
(2) a .json file containing the DNN specifications.
Examples of the hardware .json file format can be found in `genesys/SimDIT/Hardware_Json_Inference/` and `genesys/SimDIT/Hardware_Json_Training/` directories. Examples of the DNN specifications .json file format can be found in `genesys/SimDIT/DNN_Spec_Inference/` and `genesys/SimDIT/DNN_Spec_Training/` directories.

#### Outputs of SimDIT:
(1) a .csv file containing the layer-wise breakdown of the performance statistics of a benchmark DNN (stored in the working directory) and
(2) a .csv file containing the performance statistics for the full DNN benchmark.
The output results (2) from simulation get stored in `genesys/SimDIT/Results_Inference/` and `genesys/SimDIT/Results_Training/` directories.

#### Performance statistics provided by SimDIT (for training the statistics are for a single iteration):
(i) #of accesses for all the on-chip buffers for all the data types (in KB);
(ii) #of accesses for the off-chip DRAM for all the data types (in KB);
(iii) #of various arithmetic operations;
(iv) #of compute cycles; (v) #of stall cycles; (vi) total cycle counts.

Use the following command to run simulation:
```console
 python3 main_simulator.py
```
Please follow the specific instructions provided in the main_simulator.py file to simulate Inference or Training for a DNN benchmark.

