import torch, numpy as np
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, Dropout, Parameter
from torch_geometric.nn.conv  import MessagePassing
from torch_geometric.utils    import softmax as tg_softmax
from torch_geometric.nn.inits import glorot, zeros
import torch_geometric
from torch_geometric.nn import (
    Set2Set,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
    GCNConv,
    DiffGroupNorm
)
from torch_scatter import scatter_mean, scatter_add, scatter_max, scatter
import torch
from torch import nn


class MLP(nn.Module):

    def __init__(self, hs, act=None):
        super().__init__()
        self.hs = hs
        self.act = act
        
        num_layers = len(hs)

        layers = []
        for i in range(num_layers-1):
            layers += [nn.Linear(hs[i], hs[i+1])]
            if (act is not None) and (i < num_layers-2):
                layers += [act]

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
    
    def __repr__(self):
        return f'{self.__class__.__name__}(hs={self.hs}, act={self.act})'



