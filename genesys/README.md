
# Welcome to GeneSys!

The GeneSys compiler uses an embedded architecture description language to create a Hierarchical Architecture Graph for flexibly compiling _mg_-DFGs to different architectures. The GeneSys architecture uses a systolic array centric approach that forms the core convolution engine for implementing DNN algorithms, and can be customized to  run a range of standard DNN topologies.

GeneSys has three components: 
1.) RTL
2.) Compiler
3.) Software Simulator

This document will help you get up and running. 

## RTL

To generate hardware for RTL simulation or FPGA, please follow the below instructions

### Steps

```
$ cd public/genesys/GeneSys-RTL/ (This is a git submodule that links the Genesys RTL repo (https://github.com/actlab-genesys/GeneSys.git)).
$ Follow the steps in the repo to either run tests on RTL simulation or FPGA.
```


## COMPILER AND SOFTWARE SIMULATOR

### Step 0: Check prerequisites
The following dependencies must be met by your system:
  * python >= 3.7 (For [PEP 560](https://www.python.org/dev/peps/pep-0560/) support)


### Step 1: Clone the VeriGOOD-ML source code
  ```console
  $ git clone --recurse-submodules https://github.com/VeriGOOD-ML/public
  $ cd public/genesys
  ```


### Step 2: Create a [Python virtualenv](https://docs.python.org/3/tutorial/venv.html)
Note: You may choose to skip this step if you are doing a system-wide install for multiple users.
      Please DO NOT skip this step if you are installing for personal use and/or you are a developer.
```console
$ python -m venv general
$ source general/bin/activate
$ python -m pip install pip --upgrade
```

### Step 3: Install GeneSys
If you already have a working installation of Python 3.7 or Python 3.8, the easiest way to install GeneSys is:
```console
$ pip install -e .
```

### Step 4: Compile a benchmark using GeneSys
You can compile a GeneSys benchmark by running the following commands, where <model_name> is one of  `resnet18`, or `resnet50` (you can try compiling with other onnx models you find as well!)  and <config_name> is the name of one of the configs located in `genesys/examples/genesys/configs`:
```console
$ cd genesys
$ python tools/benchmark_compilation.py --model <model_name> --config <config_file>
```

Compiled output will be stored in the `genesys/tools/compilation_output/` directory.

### Step 5: Simulate a benchmark using GeneSys
After compiling the benchmark, you can run a software simulation of the benchmark. The directory `genesys/simulation/` contains the source code of the simulator. 

#### Simulator Inputs:
For a DNN benchmark, the simulator takes two files as input: 
(1) a .json file containing the GeneSys hardware parameterization and
(2) a .json file containing the compiler output generated at the specified hardware configuration.
Examples of the hardware .json file format can be found in `genesys/simulation/Hardware_Json_Inference/` and `genesys/simulation/Hardware_Json_Training/` directories. Examples of the compiler output .json file format can be found in `genesys/simulation/Compiler_Output_Inference/` and `genesys/simulation/Compiler_Output_Training/` directories.

#### Simulator Outputs:
(1) a .csv file containing the layer-wise breakdown of the performance statistics of a benchmark DNN (stored in the working directory) and
(2) a .csv file containing the performance statistics for the full DNN benchmark.
The output results (2) from simulation get stored in `genesys/simulation/Results_Inference/` and `genesys/simulation/Results_Training/` directories.

#### Performance statistics provided by the simulator (for training the statistics are for a single iteration):
(i) #of accesses for all the on-chip buffers for all the data types (in KB);
(ii) #of accesses for the off-chip DRAM for all the data types (in KB);
(iii) #of various arithmetic operations;
(iv) #of compute cycles; (v) #of stall cycles; (vi) total cycle counts.

Use the following command to run simulation:
```console
 python3 genesys/simulation/MainSimulator.py
```
Please follow the specific instructions in the genesys/simulation/MainSimulator.py to simulate Inference or Training for a DNN benchmark.

