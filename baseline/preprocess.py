import numpy as np
from scipy.io import loadmat
from torch_geometric.data import Data
import torch
from config import N_TARGET_NODES_F, N_SOURCE_NODES_F,N_TARGET_NODES,N_SOURCE_NODES
from itertools import combinations
import torch_geometric


def convert_vector_to_graph_RH(data):
    """
        convert subject vector to adjacency matrix then use it to create a graph
        edge_index:
        edge_attr:
        x:
    """

    data.reshape(1, N_SOURCE_NODES_F)
    # create adjacency matrix
    tri = np.zeros((N_SOURCE_NODES, N_SOURCE_NODES))
    tri[np.triu_indices(N_SOURCE_NODES, 1)] = data
    tri = tri + tri.T
    tri[np.diag_indices(N_SOURCE_NODES)] = 1

    edge_attr = torch.Tensor(tri).view(N_SOURCE_NODES**2, 1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    counter = 0
    pos_counter = 0
    neg_counter = 0
    N_ROI = N_SOURCE_NODES

    pos_edge_index = torch.zeros(2, N_ROI * N_ROI)
    neg_edge_indexe = []
    # pos_edge_indexe = []
    for i in range(N_ROI):
        for j in range(N_ROI):
            pos_edge_index[:, counter] = torch.tensor([i, j])
            counter += 1

        # xx = torch.ones(160, 160, dtype=torch.float)

        x = torch.tensor(tri, dtype=torch.float)
        pos_edge_index = torch.tensor(pos_edge_index, dtype=torch.long)

        return Data(x=x, pos_edge_index=pos_edge_index, edge_attr=edge_attr)
def convert_vector_to_graph_HHR(data):
    """
        convert subject vector to adjacency matrix then use it to create a graph
        edge_index:
        edge_attr:
        x:
    """

    data.reshape(1, 35778)
    # create adjacency matrix
    tri = np.zeros((268, 268))
    tri[np.triu_indices(268, 1)] = data
    tri = tri + tri.T
    tri[np.diag_indices(268)] = 1

    edge_attr = torch.Tensor(tri).view(71824, 1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    counter = 0
    pos_counter = 0
    neg_counter = 0
    N_ROI = 268

    pos_edge_index = torch.zeros(2, N_ROI * N_ROI)
    neg_edge_indexe = []
    # pos_edge_indexe = []
    for i in range(N_ROI):
        for j in range(N_ROI):
            pos_edge_index[:, counter] = torch.tensor([i, j])
            counter += 1

        # xx = torch.ones(268, 268, dtype=torch.float)

        x = torch.tensor(tri, dtype=torch.float)
        pos_edge_index = torch.tensor(pos_edge_index, dtype=torch.long)

        return Data(x=x, pos_edge_index=pos_edge_index, edge_attr=edge_attr)

def h_i(num_nodes):
  l1 = []
  l2 = []
  for j in range(1,num_nodes):
    for i in range(j):
      l1.append(i)
      l2.append(j)

  return np.array(l1), np.array(l2)


def convert_vector_to_graph_FC(data):
    """
        convert subject vector to adjacency matrix then use it to create a graph
        edge_index:
        edge_attr:
        x:
    """

    data.reshape(1, N_TARGET_NODES_F)
    # create adjacency matrix
    tri = np.zeros((N_TARGET_NODES, N_TARGET_NODES))
    tri[h_i(num_nodes=N_TARGET_NODES)] = data
    tri = tri + tri.T
    tri[np.diag_indices(N_TARGET_NODES)] = 1

    edge_attr = torch.Tensor(tri).view(N_TARGET_NODES**2, 1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    counter = 0
    pos_counter = 0
    neg_counter = 0
    N_ROI = N_TARGET_NODES

    pos_edge_index = torch.zeros(2, N_ROI * N_ROI)
    neg_edge_indexe = []
    # pos_edge_indexe = []
    for i in range(N_ROI):
        for j in range(N_ROI):
            pos_edge_index[:, counter] = torch.tensor([i, j])
            counter += 1

        # xx = torch.ones(160, 160, dtype=torch.float)

        x = torch.tensor(tri, dtype=torch.float)
        pos_edge_index = torch.tensor(pos_edge_index, dtype=torch.long)


    return Data(x=x, pos_edge_index=pos_edge_index, edge_attr=edge_attr)

def cast_data_vector_RH(dataset):
    """
        convert subject vectors to graph and append it in a list
    """

    dataset_g = []

    for subj in range(dataset.shape[0]):
        dataset_g.append(convert_vector_to_graph_RH(dataset[subj]))

    return dataset_g
def cast_data_vector_HHR(dataset):
    """
        convert subject vectors to graph and append it in a list
    """

    dataset_g = []

    for subj in range(dataset.shape[0]):
        dataset_g.append(convert_vector_to_graph_HHR(dataset[subj]))

    return dataset_g

def cast_data_vector_FC(dataset):
    """
        convert subject vectors to graph and append it in a list
    """

    dataset_g = []

    for subj in range(dataset.shape[0]):
        dataset_g.append(convert_vector_to_graph_FC(dataset[subj]))

    return dataset_g
def convert_generated_to_graph_268(data1):
    """
        convert generated output from G to a graph
    """

    dataset = []

    for data in data1:
        counter = 0
        N_ROI = 268
        pos_edge_index = torch.zeros(2, N_ROI * N_ROI, dtype=torch.long)
        for i in range(N_ROI):
            for j in range(N_ROI):
                pos_edge_index[:, counter] = torch.tensor([i, j])
                counter += 1

        x = data
        pos_edge_index = torch.tensor(pos_edge_index, dtype=torch.long)
        data = Data(x=x, pos_edge_index= pos_edge_index, edge_attr=data.view(71824, 1))
        dataset.append(data)

    return dataset
def convert_generated_to_graph(data):
    """
        convert generated output from G to a graph
    """

    dataset = []

# for data in data1:
    counter = 0
    N_ROI = N_TARGET_NODES
    pos_edge_index = torch.zeros(2, N_ROI * N_ROI, dtype=torch.long)
    for i in range(N_ROI):
        for j in range(N_ROI):
            pos_edge_index[:, counter] = torch.tensor([i, j])
            counter += 1

    x = data
    pos_edge_index = torch.tensor(pos_edge_index, dtype=torch.long)
    data = Data(x=x, pos_edge_index= pos_edge_index, edge_attr=data.view(N_TARGET_NODES**2, 1))
    dataset.append(data)

    return dataset

def convert_generated_to_graph_Al(data1):
    """
        convert generated output from G to a graph
    """









    dataset = []

    # for data in data1:
    counter = 0
    N_ROI = N_SOURCE_NODES
    pos_edge_index = torch.zeros(2, N_ROI * N_ROI, dtype=torch.long)
    for i in range(N_ROI):
        for j in range(N_ROI):
            pos_edge_index[:, counter] = torch.tensor([i, j])
            counter += 1

    # x = data
    pos_edge_index = torch.tensor(pos_edge_index, dtype=torch.long)
    data = Data(x=data1, pos_edge_index=pos_edge_index, edge_attr=data1.view(N_SOURCE_NODES*N_SOURCE_NODES, 1))
    dataset.append(data)

    return dataset


def anti_vectorize(arr_2d, num_nodes=160, h=True):
    batch_size = arr_2d.shape[0]  # Extract the batch size
    adj_matrices = np.zeros((batch_size, num_nodes, num_nodes)) # Create an array to store the adjacency matrices

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

    return torch.from_numpy(adj_matrices)


def anti_vectorize_tensor(arr_2d, num_nodes=160, h=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = arr_2d.shape[0]  # Extract the batch size
    adj_matrices = torch.zeros((batch_size, num_nodes, num_nodes)) # Create an array to store the adjacency matrices

    for i in range(batch_size):
        arr_1d = arr_2d[i].to(device)  # Extract the 1D array for this batch element
        adj_matrix = torch.zeros((num_nodes, num_nodes)).to(device)  # Create a matrix to store the adjacency matrix

        if h:
            cv, cb = h_i(num_nodes)
            adj_matrix[cv, cb] = arr_1d

        else:
            adj_matrix[np.triu_indices(n=num_nodes, k=1)] = arr_1d  # Assign the values to the upper triangular part
        adj_matrix_t = torch.transpose(adj_matrix,0,1)
        adj_matrix = (adj_matrix_t + adj_matrix)/2 # Add the transpose to make the matrix symmetric
        adj_matrix.fill_diagonal_(1)  # Set diagonal values to 1

        adj_matrices[i] = adj_matrix  # Add the adjacency matrix to the output array

    return adj_matrices


def convert_func_data_to_graph_list(arr_2d, num_nodes=160, h=True):
    batch_size = arr_2d.shape[0]
    dataset = []# Create an array to store the adjacency matrices

    for i in range(batch_size):
        arr_1d = arr_2d[i]  # Extract the 1D array for this batch element
        adj_matrix = np.zeros((num_nodes, num_nodes))  # Create a matrix to store the adjacency matrix

        if h:
            cv, cb = h_i(num_nodes)
            adj_matrix[cv, cb] = arr_1d

        else:
            adj_matrix[np.triu_indices(n=num_nodes, k=1)] = arr_1d  # Assign the values to the upper triangular part
        adj_matrix += adj_matrix.T  # Add the transpose to make the matrix symmetric
        np.fill_diagonal(adj_matrix, 1)

        x = torch.from_numpy(adj_matrix).type(torch.float32)
        x[x < 0] = 0

        dense_adj = torch.ones((num_nodes, num_nodes))

        # Convert the dense adjacency matrix to a sparse edge index
        edge_index, _ = torch_geometric.utils.dense_to_sparse(dense_adj)

        edge_attribute = x.view(num_nodes*num_nodes, 1)

        data = Data(x=x, pos_edge_index=edge_index, edge_attr=edge_attribute, edge_index=edge_index)

        dataset.append(data)

    return dataset


def post(x):

    for i in range(x.shape[1]):
        x[:, i, i] = 1

    x[x<0] = 0
    x[x>1] = 1

    x_tr = torch.transpose(x, 1, 2)
    x = (x + x_tr)/2

    return x




