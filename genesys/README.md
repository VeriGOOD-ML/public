
# Welcome to GeneSys!

The GeneSys compiler uses an embedded architecture description language to create a Hierarchical Architecture Graph for flexibly compiling _mg_-DFGs to different architectures. The GeneSys architecture uses a systolic array centric approach that forms the core convolution engine for implementing DNN algorithms, and can be customized to  run a range of standard DNN topologies.

This document will help you get up and running.  

### Step 0: Check prerequisites
The following dependencies must be met by your system:
  * python >= 3.7 (For [PEP 560](https://www.python.org/dev/peps/pep-0560/) support)


### Step 1: Clone the VeriGOOD-ML source code
  ```console
  $ git clone --recurse-submodules https://github.com/VeriGOOD-ML/public
  $ cd public
  ```


### Step 2: Create a [Python virtualenv](https://docs.python.org/3/tutorial/venv.html)
Note: You may choose to skip this step if you are doing a system-wide install for multiple users.
      Please DO NOT skip this step if you are installing for personal use and/or you are a developer.
```console
$ python -m venv ~/.venv/verigood_ml
$ source ~/.venv/verigood_ml/bin/activate
$ python -m pip install pip setuptools wheel --upgrade
```

### Step 3: Install PolyMath and the Codelets Compiler
If you already have a working installation of Python 3.7 or Python 3.8, the easiest way to install PolyMath and the Codelets compiler is:
```console
$ cd polymath && python -m pip install -r requirements.txt && python setup.py build install
$ cd ../codelets.src && python -m pip install -r requirements.txt && python setup.py build install && cd ..
```

### Step 4: Compile a benchmark using GeneSys
You can compile a GeneSys benchmark by running the following commands, where <model_name> is one of  `resnet18`, or `resnet50` (you can try compiling with other onnx models you find as well!)  and <config_name> is the name of one of the configs located in `genesys/examples/genesys/configs`:
```console
$ python compile_benchmark.py --model <model_name> --config <config_file>
```

Compiled output will be stored in the `compilation_output/` directory.

