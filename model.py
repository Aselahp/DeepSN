
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import copy
import time
import networkx as nx
import random
import pickle
from scipy.special import softmax
from scipy.sparse import csr_matrix
import pandas as pd
import argparse
import scipy.sparse as sp
import matplotlib.pyplot as plt

from main.model.diff import DeepSN, LearnableSparseCOO
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
from policy import *
import math
import markov_clustering as mc
import networkx as nx
import community  # python-louvain library
import scipy.sparse as sp
from diffusion_models import *
import statistics

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

print('Is GPU available? {}\n'.format(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CUDA_LAUNCH_BLOCKING=1

parser = argparse.ArgumentParser(description="GenIM")
datasets = ['jazz', 'cora_ml', 'power_grid', 'netscience', 'random5']
parser.add_argument("-d", "--dataset", default="jazz", type=str,
                    help="one of: {}".format(", ".join(sorted(datasets))))
diffusion = ['IC', 'LT', 'SIS']
parser.add_argument("-dm", "--diffusion_model", default="SIS", type=str,
                    help="one of: {}".format(", ".join(sorted(diffusion))))
seed_rate = [1, 5, 10, 20]
parser.add_argument("-sp", "--seed_rate", default=20, type=int,
                    help="one of: {}".format(", ".join(str(sorted(seed_rate)))))
mode = ['Normal', 'Budget Constraint']
parser.add_argument("-m", "--mode", default="normal", type=str,
                    help="one of: {}".format(", ".join(sorted(mode))))
args = parser.parse_args(args=[])

n = 2

def diffusion_evaluation(adj_matrix, seed, diffusion='LT'):
    total_infect = 0
    G = nx.from_scipy_sparse_matrix(adj_matrix).to_directed()
    values = []
    r = 100
    for i in range(r):
        
        if diffusion == 'LT':
            count = linear_threshold_model(G, seed)
            value = count * 100/G.number_of_nodes()
            values.append(value)
            total_infect += value

        elif diffusion == 'IC':
            count = independent_cascade_model(G, seed)
            value = count * 100/G.number_of_nodes()
            values.append(value)
            total_infect += value 
        elif diffusion == 'SIS':
            count = sis_model(G, seed)
            value = count * 100/G.number_of_nodes()
            values.append(value)
            total_infect += value
    return total_infect/r, statistics.stdev(values)


def csr_to_sparse_tensor(csr):
    coo = csr.tocoo()
    indices = torch.tensor([coo.row, coo.col], dtype=torch.long)
    values = torch.tensor(coo.data, dtype=torch.float)
    size = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(indices, values, size)


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def sampling(inverse_pairs):
    diffusion_count = []
    for i, pair in enumerate(inverse_pairs):
        diffusion_count.append(pair[:, 1].sum())
    diffusion_count = torch.Tensor(diffusion_count)
    top_k = diffusion_count.topk(int(0.1*inverse_pairs.shape[0])).indices
    return top_k

with open('data/' + args.dataset + '_mean_' + args.diffusion_model + str(10*args.seed_rate) + '.SG', 'rb') as f:
    graph = pickle.load(f)
    


adj, inverse_pairs = graph['adj'], graph['inverse_pairs']


#generate feature matrix
adjacency_matrix = torch.sparse_coo_tensor(
    torch.tensor([adj.tocoo().row, adj.tocoo().col], dtype=torch.long),
    torch.tensor(adj.tocoo().data, dtype=torch.float32),
    torch.Size(adj.tocoo().shape)
)

num_nodes = adjacency_matrix.size(0)

two_degree = torch.sparse.mm(adjacency_matrix, adjacency_matrix)
three_degree = torch.sparse.mm(two_degree, adjacency_matrix)
degree = (torch.sparse.sum(torch.sparse.mm(adjacency_matrix, adjacency_matrix), dim=1) ).to_dense()
unique_degrees = torch.unique(degree)

one_hot_encoder = {deg.item(): i for i, deg in enumerate(unique_degrees)}
num_unique_degrees = len(unique_degrees)
num_nodes = adjacency_matrix.size(0)
feature_matrix = torch.zeros((num_nodes, num_unique_degrees))

for i, deg in enumerate(degree):
    one_hot_index = one_hot_encoder[deg.item()]
    feature_matrix[i, one_hot_index] = 1.0
    


#adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
#adj = normalize_adj(adj) # + sp.eye(adj.shape[0]))

adj = csr_to_sparse_tensor(adj)

batch_size = 8

train_set, test_set = torch.utils.data.random_split(inverse_pairs, 
                                                    [len(inverse_pairs)-batch_size, 
                                                     batch_size])

train_loader = DataLoader(dataset=inverse_pairs, batch_size=batch_size, shuffle=True, drop_last=False)
test_loader  = DataLoader(dataset=test_set,  batch_size=1, shuffle=False)


forward_model = DeepSN(ninput = feature_matrix.shape[1],
                nfeat=8, 
                nhid=32, 
                nclass=1, 
                dropout=0.5, 
                edge_index=adj)


attention = LearnableSparseCOO(adj.coalesce().indices(), (adj.shape[0], adj.shape[1]))

optimizer = Adam([
        {'params': forward_model.parameters()},
        {'params': attention.parameters()}
    ], 
                 lr=2e-3, weight_decay=1e-4)

adj = adj.to(device)
forward_model = forward_model.to(device)
forward_model.train()

def sparse_mul(tensor1, tensor2):
    assert tensor1._indices().equal(tensor2._indices()), "Sparsity patterns do not match"
    
    result_values = tensor1._values() * tensor2._values()
    result_indices = tensor1._indices()

    return torch.sparse_coo_tensor(result_indices, result_values, tensor1.size())


def estimation_loss(y, y_hat):
    forward_loss = F.mse_loss(y_hat.squeeze(), y, reduction='sum')
    return forward_loss 

def maximization_loss(x_hat, x_i, adj, forward_model):
        
    y_hat = forward_model(x_hat.squeeze(0), adj, x_i, n)
    
    return (1 - (torch.sum(y_hat)).float()/y_hat.shape[0])


for epoch in range(50):
    begin = time.time()
    total_overall = 0

    for batch_idx, data_pair in enumerate(train_loader):
        # input_pair = torch.cat((data_pair[:, :, 0], data_pair[:, :, 1]), 1).to(device)
        
        x = data_pair[:, :, 0].float().to(device)
        y = data_pair[:, :, 1].float().to(device)
        optimizer.zero_grad()
        
        y_true = y.cpu().detach().numpy()
        x_true = x.cpu().detach().numpy()
        
        loss = 0
        for i, x_i in enumerate(x):
            x_i = x[i]
            y_i = y[i]
            
            x_hat = feature_matrix
            y_hat = forward_model(x_hat.squeeze(0), sparse_mul(adj, attention()), x_i, n)
            total = estimation_loss(y_i , y_hat)
                        
            loss += total

        total_overall += loss.item()
        loss = loss/x.size(0)
        #loss.requires_grad = True 
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(forward_model.parameters(), max_norm=1.0)
        #torch.nn.utils.clip_grad_norm_(attention.parameters(), max_norm=1.0)
        optimizer.step()
        
    end = time.time()
    print("Epoch: {}".format(epoch+1), 
          "\tTotal: {:.4f}".format(total_overall / len(train_set)),
          "\tTime: {:.4f}".format(end - begin)
         )




att_adj = sparse_mul(adj, attention()).coalesce() 
#att_adj = adj.coalesce() 

data = att_adj.values().numpy()
row = att_adj.indices()[0].numpy()
col = att_adj.indices()[1].numpy()
shape = att_adj.size()

# Create SciPy sparse COO matrix
scipy_sparse_coo = sp.coo_matrix((data, (row, col)), shape=shape)
# Create NetworkX graph from scipy sparse matrix
G = nx.Graph()
G.add_weighted_edges_from(zip(scipy_sparse_coo.row, scipy_sparse_coo.col, scipy_sparse_coo.data))

# Compute the best partition using the Louvain algorithm
partition = community.best_partition(G, weight='weight', resolution=1)

# Map communities to a torch tensor
community_tensor = torch.zeros(num_nodes, dtype=torch.long)
for node, comm_id in partition.items():
    community_tensor[node] = comm_id


score_model = NodeScoringNN(input_dim = feature_matrix.shape[1], hidden_dim = 128, output_dim = 1)

optimizer = Adam(score_model.parameters(), lr=1e-4)

for epoch in range(200):
    begin = time.time()
    total_overall = 0
    x_hat = feature_matrix
    x_i = torch.squeeze(score_model(x_hat, community_tensor, math.ceil(adj.shape[0] *args.seed_rate / 100)))
    loss = maximization_loss(x_hat, x_i, adj, forward_model,)
    #loss.requires_grad = True 
    loss.backward()
    optimizer.step()
    '''for p in score_model.parameters():
        p.data.clamp_(min=0)'''
    end = time.time()
    print("Epoch: {}".format(epoch+1), 
          "\tTotal: {:.4f}".format(loss),
          "\tTime: {:.4f}".format(end - begin)
         )
         


seed = x_i.cpu().detach().numpy()
seed = [index for index, value in enumerate(seed) if value != 0]
influence, std = diffusion_evaluation(graph['adj'], seed, diffusion = args.diffusion_model)
print(influence)
