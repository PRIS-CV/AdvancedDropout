# AdvancedDropout
Code release for Advanced Dropout: A Model-free Methodology for Bayesian Dropout Optimization [ArXiv](https://arxiv.org/abs/2010.05244 "ArXiv")

## Code List
+ main.py
	+ Main file for running
+ mlp.py
	+ Fully connected (FC) layers with advanced dropout
+ variationalBayesDropout.py
	+ Advanced dropout

## Dataset
### CIFAR-10 (and others)

## Requirements
- python >= 3.6
- PyTorch >= 1.1.0
- torchvision >= 0.3.0
- GPU memory >= 3500MiB (GTX 1080Ti)

## Training
- Download datasets
- Train and evaluate: `python main.py` or use nohup `nohup python main.py >1.out 2>&1 &`

## Citation
If you find this paper useful in your research, please consider citing:
```
@misc{xie2020advanced, 
author={Jiyang Xie and Zhanyu Ma and Guoqiang Zhang and Jing-Hao Xue and Zheng-Hua Tan and Jun Guo}, 
title={Advanced Dropout: A Model-free Methodology for Bayesian Dropout Optimization}, 
eprint={2010.05244}, 
archivePrefix={arXiv}, 
primaryClass={cs.LG}, 
year={2020}} 
```
