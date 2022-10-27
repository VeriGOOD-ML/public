# Welcome to Axiline!

Axiline is a framework for synthesizing small ML algorithms, creating efficient Verilog-based implementations by directly synthesizing the _sr_-DFG. Axiline support multiple modes that can transfer the small ML algorithm into RTL including non-template-based directly transferring, and templated based pipeline design.
This document will help you get up and running. 
### Step 0: Check prerequisites
The following dependencies must be met by your system:
  * python >= 3.7 (For [PEP 560](https://www.python.org/dev/peps/pep-0560/) support)
  * Polymath compiler is required to transfer the small ML algorithm from .onnx into sr-DFG

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

### Step 3A: Run Axiline for Matrix/vector multiplication-based  ML algorithm
Axiline has three mode of transferring small matrix/vector multiplication-based  ML algorithm into Verilog:
Typical small matrix/vector multiplication-based ML algorithms include `logistic regression`, `linear regression`,`SVM`, `recommender systems`, `backpropagations`(less or equivalent to 2 FC layer) 
1. Directly transfer DFG into combinational Verilog
2. Use predefined pipeline architectures to implement the ML algorithm (inference/training)
3. Template-based design

Call "axiline_compiler" to initiate Verilog generation and select one mode out of three.
 ```console
  axiline_compiler(<mode>, <onnx_path>, <bandwidth>, <template_path>, <output_path>)
  or
  python3 axiline_compiler.py -m mode -o <onnx_path> -b <bandwidth> -tp <template_path> -op <output_path>
```

`<mode>` should be a integer 1,2,3 corresponding to 3 listed modes
`<bandwidth>` should be a integer used to determined parallelism, which is in number of bit per cycle.
`<template_path>` should be a string of path to the folder containing Verilog for all template, only optional for template design. If not specified, compiler will use default templates.
`<output_path>` should be a string of path for output files.

### Step 3B: Run Axiline for Decision tree algorithm
Axiline is able to transfer decision tree inference algorithms into pipelined designs, supporting user-defined parallelism and number of pipeline stages.
Call "axiline_dt_compiler" to run the scheduling for decision tree.
 ```console
  axiline_dt_compiler(<decision_tree>, <num_unit>, <output_path>)
```
`<output_path>` should be a path of string to output directory.
`<num_unit>` indicate number of unit which should be an integer or a list of integers. Integer means all pipeline stage has the same units, 
A list of integers indicate the number of units in stages. If number of unit more than needed will be ignored. 
If there is not enough stages/units, the compiler will exit with an Error.
In current version, Axiline decision tree compiler only support an interface to sklearn DecisionTreeClassifier. 
`<decision_tree>` should be a sklearn.tree.DecisionTreeClassifier objective in current version.


* Some testing examples are in `/axiline/test/test_compiler*` and `/axiline/test/test_dt`
* Generated Verilog for `logistic regression`, `linear regression`,`SVM`, `recommender systems` are in `/axiline/hardware` 
* Some script used to generate RTL are in `/axiline/tools`. E.g, sigmoid.py are used to generate Verilog module (Fixed-point, LUT-based) for sigmoid function.