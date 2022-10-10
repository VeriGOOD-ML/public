# PPA Prediction models for VeriGood-ML
We proposed a ML based framework for PPA prediction of ML accelerators. We validated our framework using VeriGOOD-ML and [VTA](https://github.com/pasqoc/incubator-tvm-vta). Here we provide the code for our framework. For details you can check our paper “[Physically Accurate Learning-based Performance Prediction of Hardware-accelerated ML Algorithms](https://vlsicad.ucsd.edu/Publications/Conferences/391/c391.pdf)”.

For ASIC  implementation we use GF12 enablement. We can not publish the SP&R data of the ASIC implementation of VeriGOOD-ML and VTA designs due to NDA. In the following section we provide descriptions of the available functions that can be used to train models using [H2O](https://h2o.ai/).

## Function details
We provide two main files: 
- [rtml_ppa.py](./rtml_ppa.py): Contains all the model initialization, hyperparameter tuning, training and testing related functions.
- [rtml_helper.py](./rtml_helper.py): Contains the functions related to data, feature and metrics loading. 

Also we provide two example codes: [train_model_genesys_unseen_asic.py](./train_model_genesys_unseen_asic.py) and [train_model_tabla_mixed_asic.py](./train_model_tabla_mixed_asic.py) to show how the functions are used to load the data, trained the models. Also, we show how we choose base learners for the stacked ensemble model.

We are working on publishing the training and testing data.

## How to Cite
If you use this work, please cite our paper. ([pdf](https://vlsicad.ucsd.edu/Publications/Conferences/391/c391.pdf))
```
@inproceedings{10.1145/3551901.3556489,
author = {Esmaeilzadeh, Hadi and Ghodrati, Soroush and Kahng, Andrew B. and Kim, Joon Kyung and Kinzer, Sean and Kundu, Sayak and Mahapatra, Rohan and Manasi, Susmita Dey and Sapatnekar, Sachin S. and Wang, Zhiang and Zeng, Ziqing},
title = {Physically Accurate Learning-Based Performance Prediction of Hardware-Accelerated ML Algorithms},
year = {2022},
isbn = {9781450394864},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3551901.3556489},
doi = {10.1145/3551901.3556489},
booktitle = {Proceedings of the 2022 ACM/IEEE Workshop on Machine Learning for CAD},
pages = {119–126},
numpages = {8},
keywords = {ML accelerator, PPA prediction, design space exploration},
location = {Virtual Event, China},
series = {MLCAD '22}
}
```
