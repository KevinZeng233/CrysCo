import torch, numpy as np
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, Dropout, Parameter
from math import sqrt
from itertools import product
from collections import namedtuple
import torch
import torch.nn.functional as F
from torch import nn, einsum
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
import torch.nn as nn
import numpy as np
import pandas as pd
import math
##General imports
import csv
import os
import time
from datetime import datetime
import shutil
import copy
import numpy as np
from functools import partial
import platform

##Torch imports
import torch.nn.functional as F
import torch
from torch_geometric.data import DataLoader, Dataset
from torch_geometric.nn import DataParallel
import torch_geometric.transforms as T
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp
import process

class EGAT_att(torch.nn.Module):
    def __init__(self, dim, act, batch_norm, batch_track_stats, dropout_rate, fc_layers=2):
        super(EGAT_att, self).__init__()

        self.act       = act
        self.fc_layers = fc_layers
        if batch_track_stats      == "False":
            self.batch_track_stats = False 
        else:
            self.batch_track_stats = True 

        self.batch_norm   = batch_norm
        self.dropout_rate = dropout_rate

        self.global_mlp   = torch.nn.ModuleList()
        self.bn_list      = torch.nn.ModuleList()

        assert fc_layers > 1, "Need at least 2 fc layer"   

        for i in range(self.fc_layers + 1):
            if i == 0:
                lin    = torch.nn.Linear(dim+108, dim)
                self.global_mlp.append(lin)       
            else: 
                if i != self.fc_layers :
                    lin = torch.nn.Linear(dim, dim)
                else:
                    lin = torch.nn.Linear(dim, 1)
                self.global_mlp.append(lin)   

            if self.batch_norm == "True":
                #bn = BatchNorm1d(dim, track_running_stats=self.batch_track_stats)
                bn = DiffGroupNorm(dim, 10, track_running_stats=self.batch_track_stats)
                self.bn_list.append(bn)     

    def forward(self, x, batch, glbl_x ):
        out   = torch.cat([x,glbl_x],dim=-1)
        for i in range(0, len(self.global_mlp)):
            if i   != len(self.global_mlp) -1:
                out = self.global_mlp[i](out)
                out = getattr(F, self.act)(out)    
            else:
                out = self.global_mlp[i](out)   
                out = tg_softmax(out,batch)                
        return out
        for degree, t in features.items():
            # Compute the norms and normalized features
            norm = t.norm(dim = -1, keepdim = True).clamp(min = self.eps)
            phase = t / norm

            # Transform on norms
            parameters = self.transform[degree]
            gate_weights, scale = parameters['w_gate'], parameters['scale']

            transformed = rearrange(norm, '... () -> ...')

            if not exists(scale):
                scale = einsum('b n d, d e -> b n e', transformed, gate_weights)

            transformed = self.nonlin(transformed * scale)
            transformed = rearrange(transformed, '... -> ... ()')

            # Nonlinearity on norm
            output[degree] = (transformed * phase).view(*t.shape)
        x           = getattr(F, self.act)(self.node_layer1(chunk))
        x           = self.atten_layer(x)
        out         = tg_softmax(x,batch)
        return out


class EGAT_LAYER(MessagePassing):
    def __init__(self, dim, act, batch_norm, batch_track_stats, dropout_rate, fc_layers=2, **kwargs):
        super(EGAT_LAYER, self).__init__(aggr='add',flow='target_to_source', **kwargs)

        self.act          = act
        self.fc_layers    = fc_layers
        if batch_track_stats      == "False":
            self.batch_track_stats = False 
        else:
            self.batch_track_stats = True 

        self.batch_norm   = batch_norm
        self.dropout_rate = dropout_rate
 
        # FIXED-lines ------------------------------------------------------------
        self.heads             = 4
        self.add_bias          = True
        self.neg_slope         = 0.2

        self.bn1               = nn.BatchNorm1d(self.heads)
        self.W                 = Parameter(torch.Tensor(dim*2,self.heads*dim))
        self.att               = Parameter(torch.Tensor(1,self.heads,2*dim))
        self.dim               = dim

        if self.add_bias  : self.bias = Parameter(torch.Tensor(dim))
        else              : self.register_parameter('bias', None)
        self.reset_parameters()
        # FIXED-lines -------------------------------------------------------------

    def reset_parameters(self):
        glorot(self.W)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x,edge_attr=edge_attr)

    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr): 
        out_i   = torch.cat([x_i,edge_attr],dim=-1)
        out_j   = torch.cat([x_j,edge_attr],dim=-1)
        
        out_i   = getattr(F, self.act)(torch.matmul(out_i,self.W))
        out_j   = getattr(F, self.act)(torch.matmul(out_j,self.W))
        out_i   = out_i.view(-1, self.heads, self.dim)
        out_j   = out_j.view(-1, self.heads, self.dim)

        alpha   = getattr(F, self.act)((torch.cat([out_i, out_j], dim=-1)*self.att).sum(dim=-1))
        alpha   = getattr(F, self.act)(self.bn1(alpha))
        alpha   = tg_softmax(alpha,edge_index_i)

        alpha   = F.dropout(alpha, p=self.dropout_rate, training=self.training)
        out_j     = (out_j * alpha.view(-1, self.heads, 1)).transpose(0,1)
        return out_j

    def update(self, aggr_out):
        out = aggr_out.mean(dim=0)
        if self.bias is not None:  out = out + self.bias
        return out
    
class Radial(nn.Module):
    def __init__(
        self,
        degree_in,
        nc_in,
        degree_out,
        nc_out,
        edge_dim = 0,
        radial_hidden_dim = 64
    ):
        super().__init__()
        self.degree_in = degree_in
        self.degree_out = degree_out
        self.nc_in = nc_in
        self.nc_out = nc_out

        self.d_out = to_order(degree_out)
        self.edge_dim = edge_dim

        mid_dim = radial_hidden_dim
        edge_dim = default(edge_dim, 0)

        self.rp = nn.Sequential(
            nn.Linear(edge_dim + 1, mid_dim),
            nn.SiLU(),
            LayerNorm(mid_dim),
            nn.Linear(mid_dim, mid_dim),
            nn.SiLU(),
            LayerNorm(mid_dim),
            nn.Linear(mid_dim, nc_in * nc_out),
            Rearrange('... (lo li) -> ... lo li', li = nc_in, lo = nc_out)
        )

    def forward(self, feat):
        return self.rp(feat)

class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.heads = heads
        dim_inner = dim_head * heads
        self.to_qkv = nn.Linear(dim, dim_inner * 3)

    def forward(self, x, mask = None):
        has_degree_m_dim = x.ndim == 4

        if has_degree_m_dim:
            x = rearrange(x, '... 1 -> ...')

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        if exists(mask):
            mask = rearrange(mask, 'b n -> b 1 n 1')
            k = k.masked_fill(~mask, -torch.finfo(q.dtype).max)
            v = v.masked_fill(~mask, 0.)

        k = k.softmax(dim = -2)
        q = q.softmax(dim = -1)

        kv = einsum(k, v, 'b h n d, b h n e -> b h d e')
        out = einsum(kv, q, 'b h d e, b h n d -> b h n e')
        out = rearrange(out, 'b h n d -> b n (h d)')

        if has_degree_m_dim:
            out = rearrange(out, '... -> ... 1')

        return out
class ResidualSE3(nn.Module):
    def forward(self, x, res):
        out = {}
        for degree, tensor in x.items():
            degree = str(degree)
            out[degree] = tensor
            if degree in res:
                out[degree] = out[degree] + res[degree]
        return out

class LinearSE3(nn.Module):
    def __init__(
        self,
        fiber_in,
        fiber_out
    ):
        super().__init__()
        self.weights = nn.ParameterDict()

        for (degree, dim_in, dim_out) in (fiber_in & fiber_out):
            key = str(degree)
            self.weights[key]  = nn.Parameter(torch.randn(dim_in, dim_out) / sqrt(dim_in))

    def forward(self, x):
        out = {}
        for degree, weight in self.weights.items():
            out[degree] = einsum('b n d m, d e -> b n e m', x[degree], weight)
        return out

class NormSE3(nn.Module):

    def __init__(
        self,
        fiber,
        nonlin = nn.GELU(),
        gated_scale = False,
        eps = 1e-12,
    ):
        super().__init__()
        self.fiber = fiber
        self.nonlin = nonlin
        self.eps = eps

        # Norm mappings: 1 per feature type
        self.transform = nn.ModuleDict()
        for degree, chan in fiber:
            self.transform[str(degree)] = nn.ParameterDict({
                'scale': nn.Parameter(torch.ones(1, 1, chan)) if not gated_scale else None,
                'w_gate': nn.Parameter(rand_uniform((chan, chan), -1e-3, 1e-3)) if gated_scale else None
            })

    def forward(self, features):
        output = {}
        for degree, t in features.items():
            # Compute the norms and normalized features
            norm = t.norm(dim = -1, keepdim = True).clamp(min = self.eps)
            phase = t / norm

            # Transform on norms
            parameters = self.transform[degree]
            gate_weights, scale = parameters['w_gate'], parameters['scale']

            transformed = rearrange(norm, '... () -> ...')

            if not exists(scale):
                scale = einsum('b n d, d e -> b n e', transformed, gate_weights)

            transformed = self.nonlin(transformed * scale)
            transformed = rearrange(transformed, '... -> ... ()')

            # Nonlinearity on norm
            output[degree] = (transformed * phase).view(*t.shape)

        return output
# %%
RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)
data_type_torch = torch.float32


# %%
class ResidualNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_dims):
        super(ResidualNN, self).__init__()
        dims = [input_dim] + hidden_layer_dims
        self.fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)])
        self.res_fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1], bias=False) if (dims[i] != dims[i+1]) else nn.Identity() for i in range(len(dims)-1)])
        self.acts = nn.ModuleList([nn.LeakyReLU() for _ in range(len(dims)-1)])
        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, fea):
        for fc, res_fc, act in zip(self.fcs, self.res_fcs, self.acts):
            fea = act(fc(fea)) + res_fc(fea)
        return self.fc_out(fea)

    def __repr__(self):
        return f'{self.__class__.__name__}'
    
class ElementEncoder(nn.Module):
    def __init__(self, d_model, compute_device=None):
        super().__init__()
        self.d_model = d_model
        self.compute_device = compute_device

        mat2vec = 'mat2vec.csv'  # element embedding
        cbfv = pd.read_csv(mat2vec, index_col=0).values
        feat_size = cbfv.shape[-1]
        self.fc_mat2vec = nn.Linear(feat_size, d_model).to(self.compute_device)
        zeros = np.zeros((1, feat_size))
        cat_array = np.concatenate([zeros, cbfv])
        cat_array = torch.as_tensor(cat_array, dtype=data_type_torch)
        self.cbfv = nn.Embedding.from_pretrained(cat_array).to(self.compute_device, dtype=data_type_torch)

    def forward(self, src):
        mat2vec_emb = self.cbfv(src)
        x_emb = self.fc_mat2vec(mat2vec_emb)
        return x_emb


class PositionalEncoder(nn.Module):

    def __init__(self,
                 d_model,
                 resolution=100,
                 log10=False,
                 compute_device=None):
        super().__init__()
        self.d_model = d_model//2
        self.resolution = resolution
        self.log10 = log10
        self.compute_device = compute_device

        x = torch.linspace(0, self.resolution - 1,
                           self.resolution,
                           requires_grad=False) \
            .view(self.resolution, 1)
        fraction = torch.linspace(0, self.d_model - 1,
                                  self.d_model,
                                  requires_grad=False) \
            .view(1, self.d_model).repeat(self.resolution, 1)

        pe = torch.zeros(self.resolution, self.d_model)
        pe[:, 0::2] = torch.sin(x /torch.pow(
            50,2 * fraction[:, 0::2] / self.d_model))
        pe[:, 1::2] = torch.cos(x / torch.pow(
            50, 2 * fraction[:, 1::2] / self.d_model))
        pe = self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.clone()
        if self.log10:
            x = 0.0025 * (torch.log2(x))**2
            # clamp x[x > 1] = 1
            x = torch.clamp(x, max=1)
            # x = 1 - x  # for sinusoidal encoding at x=0
        # clamp x[x < 1/self.resolution] = 1/self.resolution
        x = torch.clamp(x, min=1/self.resolution)
        frac_idx = torch.round(x * (self.resolution)).to(dtype=torch.long) - 1
        out = self.pe[frac_idx]

        return out


class Transformer(nn.Module):
    def __init__(self, d_model, N, heads, frac=False, attn=True, compute_device=None):
        super().__init__()
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.fractional = frac
        self.attention = attn
        self.compute_device = compute_device
        self.embed = ElementEncoder(d_model=self.d_model, compute_device=self.compute_device)
        self.pe = PositionalEncoder(self.d_model, resolution=5000, log10=False)
        self.ple = PositionalEncoder(self.d_model, resolution=5000, log10=True)

        self.emb_scaler = nn.Parameter(torch.tensor([1.]))
        self.pos_scaler = nn.Parameter(torch.tensor([1.]))
        self.pos_scaler_log = nn.Parameter(torch.tensor([1.]))

        if self.attention:
            encoder_layer = nn.TransformerEncoderLayer(self.d_model, nhead=self.heads, dim_feedforward=2048, dropout=0.1)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.N)

    def forward(self, src, frac):
        x = self.embed(src) * 2 ** self.emb_scaler
        mask = frac.unsqueeze(dim=-1)
        mask = torch.matmul(mask, mask.transpose(-2, -1))
        mask[mask != 0] = 1
        src_mask = mask[:, 0] != 1

        pe = torch.zeros_like(x)
        ple = torch.zeros_like(x)
        pe_scaler = 2 ** (1 - self.pos_scaler) ** 2
        ple_scaler = 2 ** (1 - self.pos_scaler_log) ** 2
        pe[:, :, :self.d_model // 2] = self.pe(frac) * pe_scaler
        ple[:, :, self.d_model // 2:] = self.ple(frac) * ple_scaler

        if self.attention:
            x_src = x + pe + ple
            x_src = x_src.transpose(0, 1)
            x = self.transformer_encoder(x_src, src_key_padding_mask=src_mask)
            x = x.transpose(0, 1)

        if self.fractional:
            x = x * frac.unsqueeze(2).repeat(1, 1, self.d_model)

        hmask = mask[:, :, 0:1].repeat(1, 1, self.d_model)
        if mask is not None:
            x = x.masked_fill(hmask == 0, 0)

        return x
    
class CrysCo(torch.nn.Module):
    def __init__(self, data,
        out_dims=64,
        d_model=512,
        N=3,
        heads=4,
        compute_device=None,
        dim1=64,
        dim2=150,
        fully_connected=1,
        EGAT_count=5,
        res_fcount=1,
        pool="global_add_pool",
        pool_order="early",
        batch_norm="True",
        batch_track_stats="True",
        act="softplus",
        dropout_rate=0.0,
        **kwargs
    ):
        super(CrysCo, self).__init__()
        
        self.batch_track_stats = batch_track_stats == "True"
        self.batch_norm = batch_norm
        self.pool = pool
        self.act = act
        self.pool_order = pool_order
        self.dropout_rate = dropout_rate
        self.avg = False
        self.out_dims = out_dims
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.compute_device = compute_device
        self.edge_att = EGAT_att(dim1, act, batch_norm, batch_track_stats, dropout_rate)
        self.encoder = Transformer(d_model=self.d_model, N=self.N, heads=self.heads, compute_device=self.compute_device)
        self.out_hidden = [1024, 512, 256, 128]
        self.output_nn = ResidualNN(self.d_model, self.out_dims, self.out_hidden)

        assert EGAT_count > 0, "Need at least 1 gat layer"
        gc_dim = data.num_features if fully_connected == 0 else dim1
        post_fc_dim = data.num_features if fully_connected == 0 else dim1
        output_dim = 1 if data[0].y.ndim == 0 else len(data[0].y[0])
        
        if fully_connected > 0:
            self.pre_lin_list_E = torch.nn.ModuleList()  
            self.pre_lin_list_N = torch.nn.ModuleList()  

            for i in range(fully_connected):
                lin_N = torch.nn.Linear(data.num_features if i == 0 else dim1, dim1)
                self.pre_lin_list_N.append(lin_N)
                lin_E = torch.nn.Linear(data.num_edge_features if i == 0 else dim1, dim1)
                self.pre_lin_list_E.append(lin_E)    

        self.conv_list = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        for i in range(EGAT_count):
            conv = EGAT_LAYER(dim1, act, batch_norm, batch_track_stats, dropout_rate)
            self.conv_list.append(conv)
            if self.batch_norm == "True":
                bn = DiffGroupNorm(gc_dim, 10, track_running_stats=self.batch_track_stats)
                self.bn_list.append(bn)

        if res_fcount > 0:
            self.post_lin_list = torch.nn.ModuleList()
            for i in range(res_fcount):
                lin = torch.nn.Linear(post_fc_dim * 2 if i == 0 and pool_order == "early" and pool == "set2set" else post_fc_dim, dim2)
                self.post_lin_list.append(lin)
            self.lin_out = torch.nn.Linear(dim2, output_dim)

        if pool_order == "early" and pool == "set2set":
            self.set2set = Set2Set(post_fc_dim, processing_steps=3)

    def forward(self, data):        
        for i in range(len(self.pre_lin_list_N)):
            out_x = self.pre_lin_list_N[i](data.x if i == 0 else out_x)
            out_x = getattr(F, 'leaky_relu')(out_x, 0.2)
            out_e = self.pre_lin_list_E[i](data.edge_attr if i == 0 else out_e)
            out_e = getattr(F, 'leaky_relu')(out_e, 0.2)       
        prev_out_x = out_x

        for i in range(len(self.conv_list)):
            if not self.pre_lin_list_N and i == 0:
                out_x = self.conv_list[i](data.x, data.edge_index, data.edge_attr)
                out_x = self.bn_list[i](out_x) if self.batch_norm == "True" else out_x
            else:
                out_x = self.conv_list[i](out_x, data.edge_index, out_e)
                out_x = self.bn_list[i](out_x) if self.batch_norm == "True" else out_x
            out_x = torch.add(out_x, prev_out_x)
            out_x = F.dropout(out_x, p=self.dropout_rate, training=self.training)
            prev_out_x = out_x
        
        out_a = self.edge_att(out_x, data.batch, data.glob_feat)
        out_x = out_x * out_a

        output = self.encoder(data.src.to(dtype=torch.long, non_blocking=True), data.frac.to(dtype=torch.float, non_blocking=True))
        mask = (data.src == 0).unsqueeze(-1).repeat(1, 1, self.out_dims)
        output = self.output_nn(output)
        output = output.masked_fill(mask, 0)
        output = output.sum(dim=1) / (~mask).sum(dim=1)

        if self.pool_order == "early":
            if self.pool == "set2set":
                out_x = self.set2set(out_x, data.batch)
            else:
                out_x = getattr(torch_geometric.nn, self.pool)(out_x, data.batch)
            for i in range(len(self.post_lin_list)):
                out_x = self.post_lin_list[i](out_x)
                out_x = getattr(F, self.act)(out_x)
            out_x = out_x + output
            out = self.lin_out(out_x)

        if out.shape[1] == 1:
            return out.view(-1)
        else:
            return out

