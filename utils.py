import numpy as np
from scipy import io
from torch.utils.data import TensorDataset, DataLoader, Subset, SubsetRandomSampler
from sklearn.model_selection import KFold
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def h_i(num_nodes):
    l1 = []
    l2 = []
    for j in range(1, num_nodes):
        for i in range(j):
            l1.append(i)
            l2.append(j)

    return np.array(l1), np.array(l2)


def anti_vectorize(arr_2d, num_nodes=160, h=True):
    batch_size = arr_2d.shape[0]  # Extract the batch size
    adj_matrices = np.zeros((batch_size, num_nodes, num_nodes))  # Create an array to store the adjacency matrices

    for i in range(batch_size):
        arr_1d = arr_2d[i]  # Extract the 1D array for this batch element
        adj_matrix = np.zeros((num_nodes, num_nodes))  # Create a matrix to store the adjacency matrix

        if h:
            cv, cb = h_i(num_nodes)
            adj_matrix[cv, cb] = arr_1d

        else:
            adj_matrix[np.triu_indices(n=num_nodes, k=1)] = arr_1d  # Assign the values to the upper triangular part
        adj_matrix += adj_matrix.T  # Add the transpose to make the matrix symmetric
        np.fill_diagonal(adj_matrix, 1)  # Set diagonal values to 1

        adj_matrices[i] = adj_matrix  # Add the adjacency matrix to the output array

    return adj_matrices




class GraphDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DiffusionGraphDataset(Dataset):
    def __init__(self, data):
        self.data_functional = data[0]
        self.data_morphological = data[1]

    def __len__(self):
        return len(self.data_functional)

    def __getitem__(self, idx):
        return [self.data_functional[idx], self.data_morphological[idx]]




class GraphDataModule(pl.LightningDataModule):
    def __init__(self, data_dict, batch_size):
        super(GraphDataModule, self).__init__()
        self.data_dict = data_dict
        self.batch_size = batch_size
        self.setup()

    def setup(self, stage=None):
        self.datasets = {}
        for fold, (train_data, test_data) in self.data_dict.items():
            train_dataset = GraphDataset(train_data)
            test_dataset = GraphDataset(test_data)
            self.datasets[fold] = {'train': train_dataset, 'test': test_dataset}

    def train_dataloader(self):
        dataloaders = []
        for fold in self.datasets:
            dataloaders.append(DataLoader(self.datasets[fold]['train'], batch_size=self.batch_size))
        return dataloaders

    def val_dataloader(self):
        dataloaders = []
        for fold in self.datasets:
            dataloaders.append(DataLoader(self.datasets[fold]['test'], batch_size=self.batch_size))
        return dataloaders


class DiffusionGraphDataModule(pl.LightningDataModule):
    def __init__(self, data_dict, batch_size):
        super(DiffusionGraphDataModule, self).__init__()
        self.data_dict = data_dict
        self.batch_size = batch_size
        self.setup()

    def setup(self, stage=None):
        self.datasets = {}
        for fold, (train_data, test_data) in self.data_dict.items():
            train_dataset = DiffusionGraphDataset(train_data)
            test_dataset = DiffusionGraphDataset(test_data)
            self.datasets[fold] = {'train': train_dataset, 'test': test_dataset}

    def train_dataloader(self):
        dataloaders = []
        for fold in self.datasets:
            dataloaders.append(DataLoader(self.datasets[fold]['train'], batch_size=self.batch_size))
        return dataloaders

    def val_dataloader(self):
        dataloaders = []
        for fold in self.datasets:
            dataloaders.append(DataLoader(self.datasets[fold]['test'], batch_size=self.batch_size))
        return dataloaders

def connectomes_data_loader(source_data, target_data, batch_size=32):

    if source_data is None and target_data is None:
        #create random numpy data
        source_data = np.random.rand(279, 595)
        target_data = np.random.rand(279, 12720)

    else:
        source_data = io.loadmat(source_data)['morph_thickness_data_35_rh']
        target_data = io.loadmat(target_data)['LR']

    source_data = anti_vectorize(source_data, h=False, num_nodes=35)
    target_data = anti_vectorize(target_data, h=True, num_nodes=160)

    source_data = torch.from_numpy(source_data).float()
    target_data = torch.from_numpy(target_data).float()
    source_data[source_data < 0] = 0
    target_data[target_data < 0] = 0
    all_source_x = source_data
    all_target_x = target_data
    all_ds = TensorDataset(all_target_x, all_source_x)
    num_folds = 3
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=config['Seed']['seed'])
    indices = list(range(len(all_ds)))
    i_l = list(kf.split(indices))
    train_loaders = []
    val_loaders = []
    for i in range(num_folds):
        train_idx, val_idx = i_l[i]
        train_ds = Subset(all_ds, train_idx)
        val_ds = Subset(all_ds, val_idx)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

    return train_loaders, val_loaders

