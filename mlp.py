import numpy as np
import torch
import torch.nn as nn
from variationalBayesDropout import AdvancedDropout

class MLP(nn.Module):
    def __init__(self, node_list):
        '''
        params:
        node_list (int list): n elements where the first element is the input layers' node number, 
                                the middle n-1 elements are hidden layers' node numbers, 
                                and the last element is the output layers' node number
        '''
        super(MLP, self).__init__()
        self.classifier = self._make_layers(node_list)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out

    def _make_layers(self, node_list):
        layers = []
        in_dim = node_list[0]
        for idx in range(1, len(node_list) - 1):
            num = node_list[idx]
            layers += [AdvancedDropout(in_dim),
                        nn.Linear(in_dim, num),
                        nn.ReLU(inplace=True)]
            in_dim = num
        layers += [AdvancedDropout(in_dim),
                    nn.Linear(in_dim, node_list[-1])]
        return nn.Sequential(*layers)
