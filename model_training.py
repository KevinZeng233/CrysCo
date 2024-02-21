
import csv
import os
import time
from datetime import datetime
import shutil
import copy
import numpy as np
from functools import partial
import platform
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
#from dataload import loader_setup, get_dataset
from CrysCo import CrysCo
import argparse  # Add this line to import argparse


model_parameters = {  
       "out_dims":4,
        "d_model":512,
        "N":3,
        "heads":4
        ,"dim1": 64
        ,"dim2": 4
        ,"fully_connected": 1
        ,"EGAT_count": 24
        ,"res_fcount": 1
        ,"pool": "global_add_pool"
        ,"pool_order": "early"
        ,"batch_norm": "True"
        ,"batch_track_stats": "True"
        ,"act": "softplus"
        ,"model": "CrysCo"
        ,"dropout_rate": 0.0
        ,"epochs": 400
        ,"lr": 0.006
        ,"batch_size": 100
        ,"optimizer": "AdamW"
        ,"optimizer_args": {}
        ,"scheduler": "ReduceLROnPlateau"
        ,"scheduler_args": {"mode":"min", "factor":0.8, "patience":10, "min_lr":0.00001, "threshold":0.0002}}
training_parameters = { 
    "target_index": 0
    #Loss functions (from pytorch) examples: l1_loss, mse_loss, binary_cross_entropy
    ,"loss": "l1_loss"       
    #Ratios for train/val/test split out of a total of 1  
    ,"train_ratio": 0.8
    ,"val_ratio": 0.05
    ,"test_ratio": 0.15
    #Training print out frequency (print per n number of epochs)
    ,"verbosity": 1}
job_parameters= { 
        "job_name": "my_train_job"
        ,"load_model": "False"
        ,"save_model": "True"
        ,"model_path": "my_model.pth"
        ,"write_output": "True"
        ,"parallel": "True"
        ,"seed": 42

        ,"cv_folds": 5}
Predict= {
        "job_name": "my_predict_job"   
        ,"model_path": "my_model.pth"
        ,"write_output": "True"
        ,"seed": 0}


def model_summary(model):
    model_params_list = list(model.named_parameters())
    print("--------------------------------------------------------------------------")
    line_new = "{:>30}  {:>20} {:>20}".format(
        "Layer.Parameter", "Param Tensor Shape", "Param #"
    )
    print(line_new)
    print("--------------------------------------------------------------------------")
    for elem in model_params_list:
        p_name = elem[0]
        p_shape = list(elem[1].size())
        p_count = torch.tensor(elem[1].size()).prod().item()
        line_new = "{:>30}  {:>20} {:>20}".format(p_name, str(p_shape), str(p_count))
        print(line_new)
    print("--------------------------------------------------------------------------")
    total_params = sum([param.nelement() for param in model.parameters()])
    print("Total params:", total_params)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params:", num_trainable_params)
    print("Non-trainable params:", total_params - num_trainable_params)
    
def ddp_setup(rank, world_size):
    if rank in ("cpu", "cuda"):
        return
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    if platform.system() == 'Windows':
        dist.init_process_group("gloo", rank=rank, world_size=world_size)    
    else:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = True
ddp_setup('cuda', 0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def get_dataset(data_path, filename="Eh_new.pt", target_index=0):
    processed_path = "processed"
    transforms = GetY(index=target_index)

    if os.path.exists(os.path.join(data_path, processed_path, filename)):
        dataset = StructureDataset(
            data_path,
            processed_path,
            filename=filename,
            transform=transforms,
        )
        return dataset


##Dataset class from pytorch/pytorch geometric; inmemory case
class StructureDataset(InMemoryDataset):
    def __init__(
        self, data_path, processed_path="processed", filename="Eh_new.pt", transform=None, pre_transform=None
    ):
        self.data_path = data_path
        self.processed_path = processed_path
        self.filename = filename  # Add this line
        super(StructureDataset, self).__init__(data_path, transform, pre_transform)
        self.data, self.slices = torch.load(os.path.join(self.processed_dir, self.filename))

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_dir(self):
        return os.path.join(self.data_path, self.processed_path)

    @property
    def processed_file_names(self):
        return [self.filename]  #



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


def model_setup(
    rank,
    model_name,
    model_params,
    dataset,
    load_model=False,
    model_path=None,
    print_model=True,
):
    model = CrysCo(
        data=dataset, **(model_params if model_params is not None else {})
    ).to(rank)
    if load_model == "True":
        assert os.path.exists(model_path), "Saved model not found"
        if str(rank) in ("cpu"):
            saved = torch.load(model_path, map_location=torch.device("cpu"))
        else:
            saved = torch.load(model_path)
        model.load_state_dict(saved["model_state_dict"])
        # optimizer.load_state_dict(saved['optimizer_state_dict'])

    # DDP
    if rank not in ("cpu", "cuda"):
        model = DistributedDataParallel(
            model, device_ids=[rank], find_unused_parameters=True
        )

    return model


def train(model, optimizer, loader, loss_method, rank):
    model.train()
    loss_all = 0
    count = 0
    for data in loader:
        data = data.to(rank)
        optimizer.zero_grad()
        out_put = model(data)
        # print(data.y.shape, output.shape)
        loss = getattr(F, loss_method)(out_put, data.y)
        loss.backward()
        loss_all += loss.detach() * out_put.size(0)

        # clip = 10
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

        optimizer.step()
        count = count + out_put.size(0)

    loss_all = loss_all / count
    return loss_all


def trainer(
    rank,
    world_size,
    model,
    optimizer,
    scheduler,
    loss,
    train_loader,
    val_loader,
    train_sampler,
    epochs,
    verbosity,
    filename = "my_model_temp.pth",
):

    train_error = val_error = test_error = epoch_time = float("NaN")
    train_start = time.time()
    best_val_error = 1e10
    model_best = model
    ##Start training over epochs loop
    for epoch in range(1, epochs + 1):

        lr = scheduler.optimizer.param_groups[0]["lr"]
        if rank not in ("cpu", "cuda"):
            train_sampler.set_epoch(epoch)
        ##Train model
        train_error = train(model, optimizer, train_loader, loss, rank=rank)
        if rank not in ("cpu", "cuda"):
            torch.distributed.reduce(train_error, dst=0)
            train_error = train_error / world_size

        ##Get validation performance
        if rank not in ("cpu", "cuda"):
            dist.barrier()
        if val_loader != None and rank in (0, "cpu", "cuda"):
            if rank not in ("cpu", "cuda"):
                val_error = evaluate(
                    val_loader, model.module, loss, rank=rank, out=False
                )
            else:
                val_error = evaluate(val_loader, model, loss, rank=rank, out=False)

        ##Train loop timings
        epoch_time = time.time() - train_start
        train_start = time.time()

        ##remember the best val error and save model and checkpoint        
        if val_loader != None and rank in (0, "cpu", "cuda"):
            if val_error == float("NaN") or val_error < best_val_error:
                if rank not in ("cpu", "cuda"):
                    model_best = copy.deepcopy(model.module)
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "full_model": model,
                        },
                        filename,
                    )
                else:
                    model_best = copy.deepcopy(model)
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "full_model": model,
                        },
                        filename,
                    )
            best_val_error = min(val_error, best_val_error)
        elif val_loader == None and rank in (0, "cpu", "cuda"):
            if rank not in ("cpu", "cuda"):
                model_best = copy.deepcopy(model.module)
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "full_model": model,
                    },
                    filename,
                )
            else:
                model_best = copy.deepcopy(model)
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "full_model": model,
                    },
                    filename,
                )

        ##scheduler on train error
        scheduler.step(train_error)

        ##Print performance
        if epoch % verbosity == 0:
            if rank in (0, "cpu", "cuda"):
                print(
                    "Epoch: {:04d}, Learning Rate: {:.6f}, Training Error: {:.5f}, Val Error: {:.5f}, Time per epoch (s): {:.5f}".format(
                        epoch, lr, train_error, val_error, epoch_time
                    )
                )

    if rank not in ("cpu", "cuda"):
        dist.barrier()

    return model_best

def evaluate(loader, model, loss_method, rank, out=False):
    model.eval()
    loss_all = 0
    count = 0
    for data in loader:
        data = data.to(rank)
        with torch.no_grad():
            output = model(data)
            loss = getattr(F, loss_method)(output, data.y)
            loss_all += loss * output.size(0)
            if out == True:
                if count == 0:
                    ids = [item for sublist in data.structure_id for item in sublist]
                    ids = [item for sublist in ids for item in sublist]
                    predict = output.data.cpu().numpy()
                    target = data.y.cpu().numpy()
                else:
                    ids_temp = [
                        item for sublist in data.structure_id for item in sublist
                    ]
                    ids_temp = [item for sublist in ids_temp for item in sublist]
                    ids = ids + ids_temp
                    predict = np.concatenate(
                        (predict, output.data.cpu().numpy()), axis=0
                    )
                    target = np.concatenate((target, data.y.cpu().numpy()), axis=0)
            count = count + output.size(0)

    loss_all = loss_all / count

    if out == True:
        test_out = np.column_stack((ids, target, predict))
        return loss_all, test_out
    elif out == False:
        return loss_all


##Write results to csv file
def write_results(output, filename):
    shape = output.shape
    with open(filename, "w") as f:
        csvwriter = csv.writer(f)
        for i in range(0, len(output)):
            if i == 0:
                csvwriter.writerow(
                    ["ids"]
                    + ["target"] * int((shape[1] - 1) / 2)
                    + ["prediction"] * int((shape[1] - 1) / 2)
                )
            elif i > 0:
                csvwriter.writerow(output[i - 1, :])

# In[38]:


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the dataset")
    parser.add_argument("--data", type=str, required=True, help=" dataset")
    args = parser.parse_args()

    # Assuming 'data_elasticity.pt' is in the specified data directory
    database_path = os.path.join(args.data_dir, args.data)
    
    # Setup the dataset
    database = get_dataset(args.data_dir, args.data, 0)

    # Setup loaders
    (
        train_loader,
        val_loader,
        test_loader,
        train_sampler,
        train_dataset,
        _,
        _,
    ) = loader_setup(
        0.85,
        0.05,
        0.10,
        90,
        database,
        'cuda',
        42,
        0,
    )

    # Setup the model
    model = model_setup(
        'cuda',
        'CrysCo',
        model_parameters,
        database,
    )

    optimizer = getattr(torch.optim, model_parameters["optimizer"])(
        model.parameters(),
        lr=model_parameters["lr"],
        **model_parameters["optimizer_args"]
    )
    scheduler = getattr(torch.optim.lr_scheduler, model_parameters["scheduler"])(
        optimizer, **model_parameters["scheduler_args"]
    )

    trainer(
            'cuda',
            0,
            model,
            optimizer,
            scheduler,
            training_parameters["loss"],
            train_loader,
            val_loader,
            train_sampler,
            model_parameters["epochs"],
            training_parameters["verbosity"],
            "my_model_temp.pth",)


if __name__ == "__main__":
    main()


