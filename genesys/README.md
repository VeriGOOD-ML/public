
# Welcome to GeneSys!

The GeneSys compiler uses an embedded architecture description language to create a Hierarchical Architecture Graph for flexibly compiling _sr_-DFGs to different architectures. The GeneSys architecture uses a systolic array centric approach that forms the core convolution engine for implementing DNN algorithms, and can be customized to  run a range of standard DNN topologies.

This document will help you get up and running.  

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
You can compile a GeneSys benchmark by running the following commands, where <benchmark_name> is one of  `resnet18`, `resnet50`, or `maskrcnn` and <output_type> is either "simulation" or "instructions", depending on whether or not the goal is to simulate the compiled output or generate executable instructions:
```console
$ python genesys/benchmarks/run_benchmark.py --benchmark <benchmark_name> --output_type <output_type>
```

Compiled output will be stored in the `genesys/benchmarks/compilation_output/` directory.

### Step 5: Simulate a benchmark using GeneSys
After compiling the benchmark, you can run a software simulation of the benchmark by running the following command:
```console
$ python benchmarks/simulate_benchmark.py --benchmark <benchmark_name>
```
