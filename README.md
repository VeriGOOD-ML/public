
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
If you use any part of this project, please cite one of our papers:


* H. Esmaeilzadeh, S. Ghodrati, J. Gu, S. Guo, A. B. Kahng, J. K. Kim, S. Kinzer, R. Mahapatra, S. D. Manasi, E. Mascarenhas, S. S. Sapatnekar, R. Varadarajan, Z. Wang, H. Xu, B. R. Yatham, and Z. Zeng, "VeriGOOD-ML: An Open-Source Flow for Automated ML Hardware Synthesis," IEEE/ACM International Conference On Computer Aided Design (ICCAD), 2021. DOI: 10.1109/ICCAD51958.2021.9643449.
* S. Kinzer, J.K. Kim, S. Ghodrati, B. Yatham, A. Althoff, D. Mahajan, S. Lerner, and H. Esmailzadeh, "A Computational Stack for Cross-Domain Acceleration", in the IEEE International Symposium on High Performance Computer Architecture (HPCA), 2021.
* A. B. Kahng, R. Varadarajan and Z. Wang, “RTL-MP: Toward Practical, Human-Quality Chip Planning and Macro Placement", Proceedings of the ACM/IEEE International Symposium on Physical Design, 2022. DOI: 10.1145/3505170.3506731.
* H. Esmaeilzadeh, S. Ghodrati, A. B. Kahng, J. K. Kim, S. Kinzer, S. Kundu, R. Mahapatra, S. D. Manasi, S. S. Sapatnekar, Z. Wang, Z. Zeng, “Physically Accurate Learning-based Performance Prediction of Hardware-accelerated ML Algorithms,” Proceedings of the ACM/IEEE Workshop on Machine Learning for CAD (MLCAD), 2022. DOI: 10.1109/MLCAD55463.2022.9900090.
* Z. Zeng and S. S. Sapatnekar, "Energy-efficient Hardware Acceleration of Shallow Machine Learning Applications," Proceedings of the Design, Automation & Test in Europe Conference & Exhibition (DATE), 2023 DOI: 10.23919/DATE56975.2023.10137232.

