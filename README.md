
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

# VeriGOOD-ML: Verilog Generator, Optimized for Designs for Machine Learning

The objective of the VeriGOOD-ML project is to develop open-source Verilog-based compiler for RTML hardware.  The software in this repo translates an ONNX description of an ML algorithm to Verilog hardware, with no human in the loop.

Our approach is based on the following components:
* _PolyMath_: [PolyMath](https://github.com/he-actlab/polymath) is a compilation stack for multi-acceleration, and uses a recursively-defined intermediate representation allowing simultaneous access to all levels of operation granularity, called an _mg_-DFG. The flexible nature of the _mg_-DFG allows for straightforward translation to different types of Verilog implementation. The PolyMath framework also includes a translator from the [ONNX](https://github.com/onnx/onnx) format to _mg_-DFG format, which enables high-level ML models to be compiled to different Verilog hardware.

* Using the _mg_-DFG, we propose three core target engines:
    * The TABLA platform for non-DNN algorithms can be customized to perform training and inference for a variety of machine learning algorithms, including linear regression, logistic regression, support vector machines, recommender systems, and backpropagation.
    * The GeneSys platform uses a systolic array centric approach that forms the core convolution engine for implementing DNN algorithms, and can be customized to run a range of standard DNN topologies.
    * Axiline is a non-template-based framework for synthesizing small ML algorithms, creating efficient Verilog-based implementations by directly synthesizing the _mg_-DFG.


## Inputs

* An [ONNX](https://github.com/onnx/onnx) model file

## Outputs

* Executable Binary for one of TABLA, GeneSys, OR
* An efficient Verilog-based implementation (Axiline)

## Getting started

Installation and usage instructions are found in the following links:

* [TABLA](tabla)
* [GeneSys](genesys)
* [Axiline](axiline)

## Citing us
If you use this work, please cite one of our papers, PolyMath, published in the 2021 IEEE International Symposium on High Performance Computer Architecture (HPCA).

```
S. Kinzer, J.K. Kim, S. Ghodrati, B. Yatham, A. Althoff, D. Mahajan, S. Lerner, and H. Esmailzadeh, "A Computational Stack for Cross-Domain Acceleration", in the IEEE International Symposium on High Performance Computer Architecture (HPCA), 2021.
```
