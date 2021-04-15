# Welcome to Axiline!
Axiline is a non-template-based framework for synthesizing small ML algorithms, creating efficient Verilog-based implementations by directly synthesizing the _mg_-DFG.
This document will help you get up and running.  

### Step 0: Check prerequisites
The following dependencies must be met by your system:
  * python >= 3.7 (For [PEP 560](https://www.python.org/dev/peps/pep-0560/) support)


### Step 1: Clone the VeriGOOD-ML source code
  ```console
  $ git clone --recurse-submodules https://github.com/VeriGOOD-ML/public
  $ cd public/axiline
  ```


### Step 2: Create a [Python virtualenv](https://docs.python.org/3/tutorial/venv.html)
Note: You may choose to skip this step if you are doing a system-wide install for multiple users.
      Please DO NOT skip this step if you are installing for personal use and/or you are a developer.
```console
$ python -m venv general
$ source general/bin/activate
$ python -m pip install pip --upgrade
```

### Step 3: Install Axiline
If you already have a working installation of Python 3.7 or Python 3.8, the way to install Axiline is:
```console
$ pip install -e .
```

### Step 4: Compile a benchmark using Axiline
```console
$ python axiline/axiline/run_benchmark.py --benchmark <benchmark_name/onnx_directory> --congig <config_directory><optional>  --output <output_path>
```
