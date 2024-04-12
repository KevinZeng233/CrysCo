from ase.io import read
import pandas as pd
import os
import csv
import argparse
from utils import EDM_CsvLoader, create_global_feat, threshold_sort, dense_to_sparse, add_self_loops, get_dictionary, GaussianSmearing, NormalizeEdge, OneHotDegree, Cleanup,dihedral_angles, bond_angles, process_data,create_global_feat,GetY
import numpy as np
import torch
from graph_dihedral import Graph,load_graphs_targets,load_dataset,CrystalGraphDataset
from ase.io import read
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import torch
from ase.neighborlist import neighbor_list
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os
import csv
import numpy as np
from extracted_features import human_features
import csv
import os
from time import time
import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.functional as F
from torch.utils.data import Dataset
from itertools import combinations
from pymatgen.io.cif import CifParser
from pymatgen.core.periodic_table import Element


def main(data_dir, csv_data, target_property_file,output):
    tc = os.path.join(data_dir, csv_data)

    data_loaders = EDM_CsvLoader(csv_data=tc, batch_size=1)
    data_loader = data_loaders.get_data_loaders()

    data_l = []
    for X, y, formula in data_loader:
        src, frac = X.squeeze(-1).chunk(2, dim=1)
        frac = frac * (1 + (torch.randn_like(frac)) * 0.02)
        frac = torch.clamp(frac, 0, 1)
        frac[src == 0] = 0
        frac = frac / frac.sum(dim=1).unsqueeze(1).repeat(1, frac.shape[-1])
        data_l.append([y, [formula], frac, src])

    #structural part   
    #target_property_file = 'new_Eh_st.csv'
    data_list3 = human_features(data_dir, target_property_file)
    dataset = load_dataset(data_dir,target_property_file, neighbors=12, rcut=8, delta=1)
    crystal_dataset1 = CrystalGraphDataset(dataset, neighbors=12, rcut=8, delta=1)
    with open(target_property_file) as f:
        reader = csv.reader(f)
        target_data = [row for row in reader][1:]

    data_list = []
    for index, target_row in enumerate(target_data):
        structure_id = target_row[0]
        data = Data()

        ase_crystal = read(structure_id + ".cif")
        data.ase = ase_crystal

        if index == 0:
            length = [len(ase_crystal)]
            elements = [list(set(ase_crystal.get_chemical_symbols()))]
        else:
            length.append(len(ase_crystal))
            elements.append(list(set(ase_crystal.get_chemical_symbols())))

        distance_matrix = ase_crystal.get_all_distances(mic=True)
        distance_matrix_trimmed = threshold_sort(distance_matrix, 8, 12, adj=False)

        distance_matrix_trimmed = torch.Tensor(distance_matrix_trimmed)
        edge_index, edge_weight = dense_to_sparse(distance_matrix_trimmed)

        self_loops = True
        if self_loops:
            edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=len(ase_crystal), fill_value=0)
            distance_matrix_mask = (distance_matrix_trimmed.fill_diagonal_(1) != 0).int()
        else:
            distance_matrix_mask = (distance_matrix_trimmed != 0).int()

        data.edge_index = edge_index
        data.edge_weight = edge_weight
        data.edge_descriptor = {"distance": edge_weight, "mask": distance_matrix_mask}

        target = target_row[1:]
        y = torch.Tensor(np.array([target], dtype=np.float32))
        data.y = y

        atoms_index = ase_crystal.get_atomic_numbers()
        gatgnn_glob_feat = create_global_feat(atoms_index)
        gatgnn_glob_feat = np.repeat(gatgnn_glob_feat, len(atoms_index), axis=0)
        data.glob_feat = torch.Tensor(gatgnn_glob_feat).float()

        z = torch.LongTensor(atoms_index)
        data.z = z

        u = torch.zeros((3,))
        data.u = u.unsqueeze(0)

        data.structure_id = [[structure_id] * len(data.y)]

        if (index + 1) % 500 == 0 or (index + 1) == len(target_data):
            print("Data processed:", index + 1, "out of", len(target_data))

        data_list.append(data)

    def bond_angles(bond_vec, edge_index_bnd_ang):
        bond_vec /= torch.linalg.norm(bond_vec, dim=-1, keepdim=True)
        i, j = edge_index_bnd_ang
        cos_ang = (bond_vec[i] * bond_vec[j]).sum(dim=-1, keepdim=True).clamp(min=-1., max=1.)
        sin_ang = cos_ang.acos().sin()
        return torch.hstack([cos_ang, sin_ang])


    def dihedral_angles(pos, edge_index_bnd, edge_index_dih_ang):
        dih_idx = edge_index_bnd.T[edge_index_dih_ang.T].reshape(-1, 4).T
        i, j, k, l = dih_idx
        u1, u2, u3 = pos[j] - pos[i], pos[k] - pos[j], pos[l] - pos[k]
        u1, u2, u3 = u1 / torch.linalg.norm(u1, dim=-1, keepdim=True), u2 / torch.linalg.norm(u2, dim=-1, keepdim=True), u3 / torch.linalg.norm(u3, dim=-1, keepdim=True)
        cos_ang = (cross(u1, u2) * cross(u2, u3)).sum(dim=-1, keepdim=True).clamp(min=-1., max=1.)
        sin_ang = (u1 * cross(u2, u3)).sum(dim=-1, keepdim=True).clamp(min=-1., max=1.)
        return torch.hstack([cos_ang, sin_ang])
    n_atoms_max = max(length)
    species = sorted(set(sum(elements, [])))
    num_species = len(species)

    print("Max structure size:", n_atoms_max, "Max number of elements:", num_species)
    print("Unique species:", species)

    crystal_length = len(ase_crystal)
    data.length = torch.LongTensor([crystal_length])

    atom_dictionary = get_dictionary("dictionary_default.json")
    for data in data_list:
        atom_fea = np.vstack([atom_dictionary[str(atom.item())] for atom in data.z]).astype(float)

        data.x = torch.Tensor(atom_fea)

    for data in data_list:
        data = OneHotDegree(data, 12 + 1)

    distance_gaussian = GaussianSmearing(0, 1, 50, 0.2)
    NormalizeEdge(data_list, "distance")

    for data in data_list:
        data.edge_attr = distance_gaussian(data.edge_descriptor["distance"])

        if (index + 1) % 500 == 0 or (index + 1) == len(target_data):
            print("Edge processed:", index + 1, "out of", len(target_data))

    Cleanup(data_list, ["ase", "edge_descriptor"])
    for i in range(len(data_list)):
        data_list[i].Y = data_l[i][0]
        data_list[i].formula=data_l[i][1]
        data_list[i].frac=data_l[i][2]
        data_list[i].src=data_l[i][3]
        data_list[i].human_d = data_list3[i]
        data_list[i].angle_feat=crystal_dataset1[i][2]
        data_list[i].dihedral_angle_feat=crystal_dataset1[i][3]

    def check_and_remove_data_points(data_list):
        filtered_data_list = []
        for data in data_list:
            if data.angle_feat.size(0) == data.x.size(0):
                filtered_data_list.append(data)
        return filtered_data_list
    data_list1 = check_and_remove_data_points(data_list)
    for data in data_list1:
        data.angle_fea = torch.cat((data.angle_feat.view(-1, 144), data.dihedral_angle_feat), dim=1)
    dat, slices = InMemoryDataset.collate(data_list1)
    torch.save((dat, slices), output)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process data for the Jupyter notebook.')
    parser.add_argument('data_dir', type=str, help='Path to the directory containing the data files')
    parser.add_argument('csv_data', type=str, help='Path to the CSV file containing compositional data')
    parser.add_argument('target_property_file', type=str, help='Path to the CSV file containing target property data')
    parser.add_argument('output', type=str, help='Path to the .pt file to generate output')
    args = parser.parse_args()

    main(args.data_dir, args.csv_data, args.target_property_file,args.output)
