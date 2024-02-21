import os
import sys
import time
import csv
import json
import warnings
import numpy as np
import ase
import glob
from ase import io
from scipy.stats import rankdata
from scipy import interpolate
from torch_geometric.data import DataLoader, Dataset
##torch imports
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Dataset, Data, InMemoryDataset
from torch_geometric.utils import dense_to_sparse, degree, add_self_loops
import torch_geometric.transforms as T
from torch_geometric.utils import degree
import process

##basic train, val, test split
def split_data(
    dataset,
    train_ratio,
    val_ratio,
    test_ratio,
    seed=np.random.randint(1, 1e6),
    save=False,
):
    dataset_size = len(dataset)
    if (train_ratio + val_ratio + test_ratio) <= 1:
        train_length = int(dataset_size * train_ratio)
        val_length = int(dataset_size * val_ratio)
        test_length = int(dataset_size * test_ratio)
        unused_length = dataset_size - train_length - val_length - test_length
        (
            train_dataset,
            val_dataset,
            test_dataset,
            unused_dataset,
        ) = torch.utils.data.random_split(
            dataset,
            [train_length, val_length, test_length, unused_length],
            generator=torch.Generator().manual_seed(seed),
        )
        print(
            "train length:",
            train_length,
            "val length:",
            val_length,
            "test length:",
            test_length,
            "unused length:",
            unused_length,
            "seed :",
            seed,
        )
        return train_dataset, val_dataset, test_dataset
    else:
        print("invalid ratios")


def get_dataset(data_path, target_index, reprocess="False", processing_args=None):
    if processing_args == None:
        processed_path = "processed"
    else:
        processed_path = processing_args.get("processed_path", "processed")

    transforms = GetY(index=target_index)

    if os.path.exists(data_path) == False:
        print("Data not found in:", data_path)
        sys.exit()

    if reprocess == "True":
        os.system("rm -rf " + os.path.join(data_path, processed_path))
        process_data(data_path, processed_path, processing_args)

    if os.path.exists(os.path.join(data_path, processed_path, "Eh_new.pt")) == True:
        dataset = StructureDataset(
            data_path,
            processed_path,
            transforms,
        )
    elif os.path.exists(os.path.join(data_path, processed_path, "data0.pt")) == True:
        dataset = StructureDataset_large(
            data_path,
            processed_path,
            transforms,
        )
    else:
        process_data(data_path, processed_path, processing_args)
        if os.path.exists(os.path.join(data_path, processed_path, "Eh_new.pt")) == True:
            dataset = StructureDataset(
                data_path,
                processed_path,
                transforms,
            )
        elif os.path.exists(os.path.join(data_path, processed_path, "data0.pt")) == True:
            dataset = StructureDataset_large(
                data_path,
                processed_path,
                transforms,
            )        
    return dataset


##Dataset class from pytorch/pytorch geometric; inmemory case
class StructureDataset(InMemoryDataset):
    def __init__(
        self, data_path, processed_path="processed", transform=None, pre_transform=None
    ):
        self.data_path = data_path
        self.processed_path = processed_path
        super(StructureDataset, self).__init__(data_path, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_dir(self):
        return os.path.join(self.data_path, self.processed_path)

    @property
    def processed_file_names(self):
        file_names = ["Eh_new.pt"]
        return file_names



        return data
##Get specified y index from data.y
class GetY(object):
    def __init__(self, index=0):
        self.index = index

    def __call__(self, data):
        # Specify target.
        if self.index != -1:
            data.y = data.y[0][self.index]
        return data

def loader_setup(
    train_ratio,
    val_ratio,
    test_ratio,
    batch_size,
    dataset,
    rank,
    seed,
    world_size=0,
    num_workers=0,
):
    ##Split datasets
    train_dataset, val_dataset, test_dataset = process.split_data(
        dataset, train_ratio, val_ratio, test_ratio, seed
    )

    ##DDP
    if rank not in ("cpu", "cuda"):
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank
        )
    elif rank in ("cpu", "cuda"):
        train_sampler = None

    ##Load data
    train_loader = val_loader = test_loader = None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    # may scale down batch size if memory is an issue
    if rank in (0, "cpu", "cuda"):
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
        if len(test_dataset) > 0:
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
    return (
        train_loader,
        val_loader,
        test_loader,
        train_sampler,
        train_dataset,
        val_dataset,
        test_dataset,
    )



