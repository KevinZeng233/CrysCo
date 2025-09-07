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
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch, numpy as np
from torch import Tensor
import torch.nn as nn
class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=16,
        heads=4
    ):
        super().__init__()
        self.heads = heads
        dim_inner = dim_head * heads
        self.to_qkv = nn.Linear(dim, dim_inner * 3)

    def forward(self, x, mask=None):
        has_degree_m_dim = x.ndim == 4

        if has_degree_m_dim:
            x = rearrange(x, '... 1 -> ...')

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        if mask is not None:
            mask = rearrange(mask, 'b n -> b 1 n 1')
            k = k.masked_fill(~mask, -torch.finfo(q.dtype).max)
            v = v.masked_fill(~mask, 0.)

        k = k.softmax(dim=-2)
        q = q.softmax(dim=-1)

        kv = torch.einsum('bhnd, bhne -> bhde', k, v)
        out = torch.einsum('bhde, bhnd -> bhne', kv, q)
        out = rearrange(out, 'b h n d -> b n (h d)')

        if has_degree_m_dim:
            out = rearrange(out, '... -> ... 1')

        return out

class GatedGCN(MessagePassing):

    def __init__(self, node_dim=64, edge_dim=64, epsilon=1e-5):
        super().__init__(aggr='add')
        self.W_src  = nn.Linear(node_dim, node_dim)
        self.W_dst  = nn.Linear(node_dim, node_dim)
        self.W_A    = nn.Linear(node_dim, edge_dim)
        self.W_B    = nn.Linear(node_dim, edge_dim)
        self.W_C    = nn.Linear(edge_dim, edge_dim)
        self.act    = nn.SiLU()
        self.sigma  = nn.Sigmoid()
        self.norm_x = nn.LayerNorm([node_dim])
        self.norm_e = nn.LayerNorm([edge_dim])
        self.eps    = epsilon

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.W_src.weight); self.W_src.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.W_dst.weight); self.W_dst.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.W_A.weight);   self.W_A.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.W_B.weight);   self.W_B.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.W_C.weight);   self.W_C.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_attr):
        i, j = edge_index

        # Calculate gated edges
        sigma_e = self.sigma(edge_attr)
        e_sum   = scatter(src=sigma_e, index=i , dim=0)
        e_gated = sigma_e / (e_sum[i] + self.eps)

        # Update the nodes (this utilizes the gated edges)
        out = self.propagate(edge_index, x=x, e_gated=e_gated)
        out = self.W_src(x) + out
        out = x + self.act(self.norm_x(out))

        # Update the edges
        edge_attr = edge_attr + self.act(self.norm_e(self.W_A(x[i]) \
                                                   + self.W_B(x[j]) \
                                                   + self.W_C(edge_attr)))

        return out, edge_attr

    def message(self, x_j, e_gated):
        return e_gated * self.W_dst(x_j)
        

#this code is obtained from  https://www.cell.com/patterns/pdf/S2666-3899(22)00076-9.pdf      
class EGAT_att(torch.nn.Module):
    def __init__(self, dim, act, batch_norm, dropout_rate,fc_layers=2):
        super(EGAT_att, self).__init__()

        self.act       = act
        self.fc_layers = fc_layers
        self.global_mlp   = torch.nn.ModuleList()
        self.bn_list      = torch.nn.ModuleList()   

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
            bn = BatchNorm1d(dim)
            self.bn_list.append(bn)     

    def forward(self, x, batch, glbl_x):
        out   = torch.cat([x,glbl_x],dim=-1)
        for i in range(0, len(self.global_mlp)):
            if i   != len(self.global_mlp) -1:
                out = self.global_mlp[i](out)
                out = getattr(F, self.act)(out)    
            else:
                out = self.global_mlp[i](out)   
                out = tg_softmax(out,batch)                
        return out

        x           = getattr(F, self.act)(self.node_layer1(chunk))
        x           = self.atten_layer(x)
        out         = tg_softmax(x,batch)
        return out


class EGAT_LAYER(MessagePassing):
    def __init__(self, dim, activation, use_batch_norm, dropout, fc_layers=2, **kwargs):
        super().__init__(aggr='add', flow='target_to_source', **kwargs)
        self.activation_func = getattr(F, activation)
        self.dropout = dropout
        self.dim = dim
        self.heads = 4
        self.weight = Parameter(torch.Tensor(dim * 2, self.heads * dim))
        self.attention = Parameter(torch.Tensor(1, self.heads, 2 * dim))
        self.bias = Parameter(torch.Tensor(dim)) if kwargs.get('add_bias', True) else None
        self.bn = nn.BatchNorm1d(self.heads)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.attention)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr):
        combined_x_i = self.activation_func(torch.matmul(torch.cat([x_i, edge_attr], dim=-1), self.weight)).view(-1, self.heads, self.dim)
        combined_x_j = self.activation_func(torch.matmul(torch.cat([x_j, edge_attr], dim=-1), self.weight)).view(-1, self.heads, self.dim)
        alpha = self.activation_func((torch.cat([combined_x_i, combined_x_j], dim=-1) * self.attention).sum(dim=-1))
        if self.bn:
            alpha = self.activation_func(self.bn(alpha))
        alpha = tg_softmax(alpha, edge_index_i)
        return (combined_x_j * F.dropout(alpha, p=self.dropout, training=self.training).view(-1, self.heads, 1)).transpose(0, 1)

    def update(self, aggr_out):
        return aggr_out.mean(dim=0) + (self.bias if self.bias is not None else 0)
    
class EGATs_attention(MessagePassing):
    def __init__(self, dim, activation='relu', use_batch_norm=False, dropout_rate=0.5, 
                 num_heads=4, add_bias=True, num_fc_layers=2, edge_dim=None, **kwargs):
        super().__init__(aggr='add', **kwargs)  # 聚合方式为 'add' (求和)
        self.dim = dim # 节点特征和边特征的维度
        self.activation = activation
        self.use_batch_norm = use_batch_norm # 是否在注意力系数上使用批归一化
        self.dropout_rate = dropout_rate # 注意力系数的 dropout 率
        self.num_heads = num_heads # 多头注意力的头数
        self.add_bias = add_bias # 是否在输出时添加可学习的偏置项
        self.num_fc_layers = num_fc_layers # 后续扩展预留
        self.edge_dim = edge_dim if edge_dim is not None else dim # 边特征的维度

        self.bn1 = nn.BatchNorm1d(num_heads) if use_batch_norm else None # 注意力系数的BN层
        self.W = Parameter(torch.Tensor(dim * 2, num_heads * dim)) # 变换节点和边特征的权重矩阵 它为每个注意力头准备变换后的特征。
        self.att = Parameter(torch.Tensor(1, num_heads, 2 * dim)) # 注意力机制的参数向量 用于计算每个头、每条边的注意力分数。
        if add_bias:
            self.bias = Parameter(torch.Tensor(dim)) # 输出偏置
        else:
            self.bias = None

        self.edge_transform = nn.Linear(1, self.edge_dim) # 用于更新边特征的线性层 一个线性层，将标量注意力分数（均值）映射到新的 edge_dim 维边特征

        self.reset_parameters() # 初始化参数

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        nn.init.xavier_uniform_(self.edge_transform.weight)
        if self.edge_transform.bias is not None:
            nn.init.zeros_(self.edge_transform.bias)
    
    # 消息传递过程 (forward, message, update)
    # 消息传递的三个核心步骤是：propagate() -> message() -> update()。
    
    # 第1步：forward - 启动传播
    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr is None:
            edge_attr = torch.zeros((edge_index.size(1), self.edge_dim), device=x.device)
        # 调用 self.propagate(...) 启动消息传递过程。PyG 会自动调用 message 和 update 方法。
        out, edge_attr_updated = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        # 返回更新后的节点表示和更新后的边特征
        return out, edge_attr_updated

    # 第2步：message - 构建消息 & 计算注意力
    # 这是最复杂也是最重要的方法，它为图中的每条边计算要发送的消息。
    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr):
        # a) 准备输入：
        out_i = torch.cat([x_i, edge_attr], dim=-1) # 拼接目标节点（消息接收方）i和边特征
        out_j = torch.cat([x_j, edge_attr], dim=-1) # 拼接源节点（消息发送方）j和边特征

        # b) 线性变换与激活：将拼接后的特征通过权重矩阵 self.W 进行变换，并应用激活函数
        # 此时 out_i 和 out_j 的形状为 [num_edges, num_heads * dim]
        act_func = getattr(F, self.activation)
        out_i = act_func(torch.matmul(out_i, self.W)) # 变换目标节点+边
        out_j = act_func(torch.matmul(out_j, self.W)) # 变换源节点+边

        # c) 重塑为多头： 将变换后的特征重塑，使每个注意力头的特征分离。
        out_i = out_i.view(-1, self.num_heads, self.dim)
        out_j = out_j.view(-1, self.num_heads, self.dim)

        # d) 计算注意力系数：
        # 将变换后的目标节点和源节点特征再次拼接。
        # 与注意力向量 self.att 做元素乘法后求和，这等价于一个简化版的点积注意力，计算每条边、每个头上的注意力分数 alpha。
        alpha = (torch.cat([out_i, out_j], dim=-1) * self.att).sum(dim=-1)
        
        # e) 规范化注意力：
        # tg_softmax(alpha, edge_index_i)：这是关键一步。它确保对于同一个目标节点 i 的所有入边，其注意力系数之和为 1。这是标准 GAT 的做法。
        if self.use_batch_norm and self.bn1 is not None:
            alpha = self.bn1(alpha)
        alpha = act_func(alpha)
        alpha = tg_softmax(alpha, edge_index_i)
        alpha = F.dropout(alpha, p=self.dropout_rate, training=self.training)
        
        # f) 更新边特征：
        alpha_avg = alpha.mean(dim=1, keepdim=True) # [E, 1] 对多头注意力取平均，得到一个标量
        self.edge_attr_updated = self.edge_transform(alpha_avg) # [E, edge_dim] 将标量映射为新的边特征

        # g) 加权消息：
        # 用计算好的注意力系数 alpha 对源节点变换后的特征 out_j 进行加权。这就是要传递的“消息”。
        out_j = (out_j * alpha.view(-1, self.num_heads, 1)).transpose(0, 1)
        return out_j

    # aggr_out：已经通过 aggr='add' 方式聚合好的消息，形状为 [num_heads, num_nodes, dim]。对于每个节点，其所有邻居的消息已经求和。
    # out = aggr_out.mean(dim=0)：将多个注意力头的输出进行平均，得到最终的节点表示。
    # 最后返回更新后的节点特征和之前在 message 中计算好的更新后的边特征。
    def update(self, aggr_out, edge_attr):
        out = aggr_out.mean(dim=0)
        if self.bias is not None:
            out = out + self.bias
        return out, self.edge_attr_updated
    
