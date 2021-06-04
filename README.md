# AdvancedDropout
Advanced Dropout: A Model-free Methodology for Bayesian Dropout Optimization (IEEE TPAMI 2021) [IEEE Xplore](https://ieeexplore.ieee.org/document/9439951 "IEEE Xplore") or [ArXiv](https://arxiv.org/abs/2010.05244 "ArXiv")

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
@ARTICLE{9439951,
  author={Xie, Jiyang and Ma, Zhanyu and Lei, Jianjun and Zhang, Guoqiang and Xue, Jing-Hao and Tan, Zheng-Hua and Guo, Jun},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Advanced Dropout: A Model-free Methodology for Bayesian Dropout Optimization}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TPAMI.2021.3083089}}
```
