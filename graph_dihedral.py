#this code has been obtained from ALIGNN-D
import torch
from torch_geometric.data import Data
import torch
from torch.linalg import cross


def bond_angles(bond_vec, edge_index_bnd_ang):
    bond_vec /= torch.linalg.norm(bond_vec, dim=-1, keepdim=True)
    i = edge_index_bnd_ang[0]
    j = edge_index_bnd_ang[1]
    cos_ang = (bond_vec[i] * bond_vec[j]).sum(dim=-1, keepdim=True).clamp(min=-1., max=1.)
    sin_ang = cos_ang.acos().sin()
    return torch.hstack([cos_ang, sin_ang])


def dihedral_angles(pos, edge_index_bnd, edge_index_dih_ang):
    """Does not account for periodic boundaries"""
    dih_idx = edge_index_bnd.T[edge_index_dih_ang.T].reshape(-1, 4)
    dih_idx = dih_idx.T
    i, j, k, l = dih_idx[0], dih_idx[1], dih_idx[3], dih_idx[2]
    u1 = pos[j] - pos[i]
    u2 = pos[k] - pos[j]
    u3 = pos[l] - pos[k]
    u1 /= torch.linalg.norm(u1, dim=-1, keepdim=True)
    u2 /= torch.linalg.norm(u2, dim=-1, keepdim=True)
    u3 /= torch.linalg.norm(u3, dim=-1, keepdim=True)
    cos_ang = (cross(u1, u2) * cross(u2, u3)).sum(dim=-1, keepdim=True).clamp(min=-1., max=1.)
    sin_ang = (u1 * cross(u2, u3)).sum(dim=-1, keepdim=True).clamp(min=-1., max=1.)
    return torch.hstack([cos_ang, sin_ang])


class MolData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key in {'edge_index', 'bnd_index', 'aux_bnd_index'}:
            return self.x_atm.size(0)
        if key in {'ang_index', 'bnd_ang_index', 'dih_ang_index'}:
            return self.x_bnd.size(0)
        return super().__inc__(key, value, *args, **kwargs)

    def bond_features(self, bond_index_name='bnd_index'):
        """Does not account for periodic boundaries"""
        i, j = self.__getattr__(bond_index_name)
        bond_vec = self.pos[j] - self.pos[i]
        bond_len = bond_vec.norm(dim=-1, keepdim=True)
        return torch.hstack([bond_vec, bond_len])

    def bond_angle_features(self, bond_feature_name='x_bnd', bond_angle_index_name='bnd_ang_index'):
        x_bnd = self.__getattr__(bond_feature_name)
        bnd_ang_index = self.__getattr__(bond_angle_index_name)
        return bond_angles(x_bnd[:, :3], bnd_ang_index)
    
    def dihedral_angle_features(self, bond_index_name='bnd_index', dihehdral_angle_index_name='dih_ang_index'):
        """Does not account for periodict boundaries"""
        bnd_index = self.__getattr__(bond_index_name)
        dih_ang_index = self.__getattr__(dihehdral_angle_index_name)
        return dihedral_angles(self.pos, bnd_index, dih_ang_index)
    
    def concat_features_with_onehot(self, x1_name='x_bnd', x2_name='x_aux_bnd'):
        x1 = self.__getattr__(x1_name)
        x2 = self.__getattr__(x2_name)
        mask = torch.cat([
            torch.zeros(x1.size(0), dtype=torch.long, device=self.pos.device),
            torch.ones(x2.size(0),  dtype=torch.long, device=self.pos.device),
        ])
        onehot = torch.nn.functional.one_hot(mask)
        return torch.hstack([torch.vstack([x1, x2]), onehot])
    
    def get_bnd_ang_vals(self):
        cos_ang, _ = self.x_bnd_ang[:, :2].T
        return cos_ang.arccos().rad2deg()
    
    def get_dih_ang_vals(self):
        cos_ang, sin_ang = self.x_dih_ang[:, :2].T
        return torch.atan2(sin_ang, cos_ang).rad2deg()


# In[7]:


import torch
import numpy as np

from ase.neighborlist import primitive_neighbor_list


def ase_radius_graph(pos, cutoff, numbers=None, cell=np.diag([1.,1.,1.]), pbc=[False]*3):
    """Computes graph edges based on a cutoff radius for 3D structure data with periodic boundaries.
    Returns the edge indices `edge_index` and the edge vectors `edge_vec`.

    This implementation uses ASE's neighbor list algorithm, which accounts for periodic boundaries.
    """
    pos_  = pos.clone().detach().cpu().numpy()
    i, j, S = primitive_neighbor_list('ijS', positions=pos_, cell=cell, cutoff=cutoff, pbc=pbc, numbers=numbers)
    i = torch.tensor(i, dtype=torch.long,  device=pos.device)
    j = torch.tensor(j, dtype=torch.long,  device=pos.device)
    S = torch.tensor(S, dtype=torch.float, device=pos.device)
    edge_index = torch.stack([i, j])
    edge_vec = pos[j] - pos[i] + S@cell
    return edge_index, edge_vec


# In[8]:


import numpy as np
import pandas as pd


mask2index = lambda mask: np.flatnonzero(mask)


def index2mask(idx_arr, n):
    mask = np.zeros(n, dtype=int)
    mask[idx_arr] = 1
    return mask.astype(np.bool)


def np_groupby(arr, groups):
    """Numpy implementation of `groupby` operation (a common method in pandas).
    """
    arr, groups = np.array(arr), np.array(groups)
    sort_idx = groups.argsort()
    arr = arr[sort_idx]
    groups = groups[sort_idx]
    return np.split(arr, np.unique(groups, return_index=True)[1])[1:]


def np_scatter(src, index, func):
    """Generalization of the `torch_scatter.scatter` operation for any reduce function.
    See https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html for how `scatter` works.

    Args:
        src (array): The source array.
        index (array of int): The indices of elements to scatter.
        func (function): Reduce function (e.g., mean, sum) that operates on elements with the same indices.

    :rtype: generator
    """
    return (func(g) for g in np_groupby(src, index))


# In[9]:


import numpy as np
import itertools
from functools import partial


permute_2 = partial(itertools.permutations, r=2)
def line_graph(edge_index_G):
    """Return the (angular) line graph of the input graph.

    Args:
        edge_index_G (ndarray): Input graph in COO format.
    """
    src_G, dst_G = edge_index_G
    edge_index_A = [
        (u, v)
        for edge_pairs in np_scatter(np.arange(len(dst_G)), dst_G, permute_2)
        for u, v in edge_pairs
    ]
    return np.array(edge_index_A).T


def dihedral_graph(edge_index_G):
    """Return the "dihedral angle line graph" of the input graph.

    Args:
        edge_index_G (ndarray): Input graph in COO format.
    """
    src, dst = edge_index_G
    edge_index_A = [
        (u, v)
        for i, j in edge_index_G.T
        for u in np.flatnonzero((dst == i) & (src != j))
        for v in np.flatnonzero((dst == j) & (src != i))
    ]
    return np.array(edge_index_A).T

def atoms2molgraph(atoms, cutoff):
    """Returns edge indices of atomic bonds according to a cutoff criteria.  
    """
    i, j = neighbor_list('ij', atoms, cutoff=3)
    return np.stack((i, j))

def atoms2pygdata(atoms):
    """Converts ASE `atoms` into a PyG graph data holding the molecular graph (G) and the angular graph (A).
    The angular graph holds both bond and dihedral angles.
    """
    encoder = OneHotEncoder()

    # Reshape the atomic numbers array for encoding
    numbers_reshaped = atoms.numbers.reshape(-1, 1)

    # Fit and transform the atomic numbers
    x_atm = encoder.fit_transform(numbers_reshaped).toarray() 
    edge_index_bnd = atoms2molgraph(atoms, cutoff=3)
    edge_index_bnd_ang = line_graph(edge_index_bnd)
    edge_index_dih_ang = dihedral_graph(edge_index_bnd)

    data = MolData(
        pos                = torch.tensor(atoms.positions,    dtype=torch.float),
        x_atm              = torch.tensor(x_atm,              dtype=torch.float),
        edge_index_bnd     = torch.tensor(edge_index_bnd,     dtype=torch.long),
        edge_index_bnd_ang = torch.tensor(edge_index_bnd_ang, dtype=torch.long),
        edge_index_dih_ang = torch.tensor(edge_index_dih_ang, dtype=torch.long),
    )
    return data

@torch.no_grad()
def update_bonds_and_angles(data, batch=None):

    data.x_bnd = data.bond_features('edge_index_bnd')
    data.x_bnd_ang = data.bond_angle_features('x_bnd', 'edge_index_bnd_ang')
    data.x_dih_ang = data.dihedral_angle_features('edge_index_bnd', 'edge_index_dih_ang')
    data.x_ang = data.concat_features_with_onehot('x_bnd_ang', 'x_dih_ang')
    data.edge_index_ang = torch.hstack([data.edge_index_bnd_ang, data.edge_index_dih_ang])
    return data

