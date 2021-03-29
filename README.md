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
@ARTICLE{xie2020advanced, 
author={Jiyang Xie and Zhanyu Ma and Guoqiang Zhang and Jing-Hao Xue and Zheng-Hua Tan and Jun Guo}, 
journal={Arxiv preprint, arXiv:2010.05244}, 
title={Advanced Dropout: A Model-free Methodology for Bayesian Dropout Optimization}, 
year={2020}} 
```

## Contact
Thanks for your attention!
If you have any suggestion or question, you can leave a message here or contact us directly:
- xiejiyang2013@bupt.edu.cn
- mazhanyu@bupt.edu.cn
