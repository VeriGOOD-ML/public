
# Welcome to Tabla!

Tabla is an innovative framework that accelerates a class of statistical machine learning algorithms. It consists of the accelerator template, domain-specific language, and model compiler.

This document will help you get up and running.  


### Step 0: Check prerequisites
The following dependencies must be met by your system:
  * python >= 3.7 (For [PEP 560](https://www.python.org/dev/peps/pep-0560/) support)


### Step 1: Clone the VeriGOOD-ML source code
  ```console
  $ git clone https://github.com/VeriGOOD-ML/public
  $ cd public/tabla
  ```


### Step 2: Create a [Python virtualenv](https://docs.python.org/3/tutorial/venv.html)
Note: You may choose to skip this step if you are doing a system-wide install for multiple users.
      Please DO NOT skip this step if you are installing for personal use and/or you are a developer.
```console
$ python -m venv general
$ source general/bin/activate
$ python -m pip install pip --upgrade
```

### Step 3: Install TABLA
If you already have a working installation of Python 3.7 or Python 3.8, the easiest way to install TABLA is:
```console
$ pip install -e .
```

### Step 4: Compile a benchmark using TABLA
You can compile a TABLA benchmark by running the following commands, where <benchmark_name> is one of  `backprop`, `linear`, `logistic`, `reco`, `svm`, `svm_wifi` and <feature_size> corresponds to the feature size of the target benchmark:
```console
$ python benchmarks/run_benchmark.py --benchmark <benchmark_name> --feature_size <feature_size>
```

Compiled output will be stored in the `tabla/benchmarks/compilation_output/` directory.

### Step 5: Simulate a benchmark using TABLA
After compiling the benchmark, you can run a software simulation of the benchmark by running the following command:
```console
$ python benchmarks/simulate_benchmark.py --benchmark <benchmark_name> --feature_size <feature_size>
```
