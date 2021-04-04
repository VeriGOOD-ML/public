
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

# VeriGOOD-ML: Verilog Generator, Optimized for Designs for Machine Learning

The objective of the VeriGOOD-ML project is to translate an ONNX description of an ML
algorithm to Verilog hardware, and to demonstrate the implementation of the hardware to GDSII.
The project is therefore designed to produce the following outcomes:
* Open-source software for a no-human-in-the-loop, Verilog-based compiler for RTML
hardware.
* Hardware in the form of a test chip for a chiplet-based ecosystem, taking the output of the
Verilog generator to GDSII layout for a chip that will be fabricated, packaged, tested, and
characterized.

Our approach is based on the following components:
* _PolyMath_: [PolyMath](https://github.com/he-actlab/polymath) is a compilation stack for multi-acceleration, and uses a recursively-defined intermediate representation allowing simultaneous access to all levels of operation granu- larity, called an _sr_-DFG. The flexible nature of the _sr_-DFG allows for straightforward translation to different types of Verilog implementation. The PolyMath framework also includes a translator from the [ONNX](https://github.com/onnx/onnx) format to _sr_-DFG format, which enables high-level ML models to be compiled to different Verilog hardware.

* Using the _sr_-DFG, we propose three core target engines:
    * The TABLA platform for non-DNN algorithms can be customized to perform training and inference for a variety of machine learning algorithms, including linear regression, logistic regression, support vector machines, recommender systems, and backpropagation.
    * The GeneSys platform us3es a systolic array centric approach that forms the core convolution engine for implementing DNN algorithms, and can be customized to run a range of standard DNN topologies.
    * Acciline is a non-template-based framework for synthesizing small ML algorithms, creating efficient Verilog-based implementations by directly synthesizing the _sr_-DFG.


## Inputs

* An [ONNX](https://github.com/onnx/onnx) model file

## Outputs

* Executable Binary for one of TABLA, GeneSys, OR
* An efficient Verilog-based implementation (Acciline)

## Getting started

Installation and usage instructions are found in the following links:

* [TABLA](tabla)
* [GeneSys](genesys)
* [Acciline](acciline)
