import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from scipy.sparse import csr_matrix
import pickle
import argparse

def combine_lists_to_tensor(list1, list2):
    """
    Combine two lists of lists into a single tensor with shape (n, m, 2).
    
    Parameters:
    list1 (list of lists): First list of lists.
    list2 (list of lists): Second list of lists.
    
    Returns:
    torch.Tensor: Combined tensor of shape (n, m, 2).
    """
    # Convert lists to PyTorch tensors
    tensor1 = torch.tensor(list1)
    tensor2 = torch.tensor(list2)
    
    # Stack tensors along the last dimension to create a tensor of shape (n, m, 2)
    combined_tensor = torch.stack((tensor1, tensor2), dim=-1)
    
    return combined_tensor


def select_random_nodes(G, percentage):
    """
    Select a random percentage of nodes from the graph G.

    Parameters:
    G (networkx.Graph): The input graph.
    percentage (float): The percentage of nodes to select (between 0 and 100).

    Returns:
    tuple: A list of randomly selected nodes and a list indicating selected nodes.
    """
    num_nodes = len(G.nodes)
    num_to_select = int(num_nodes * (percentage / 100))
    selected_nodes = random.sample(list(G.nodes), num_to_select)
    
    selection_indicator = [1 if node in selected_nodes else 0 for node in range(num_nodes)]
    
    return selected_nodes, selection_indicator

def linear_threshold_model(sparse_tensor, initial_active, iterations):
    indices = sparse_tensor.coalesce().indices()
    values = sparse_tensor.coalesce().values()
    size = sparse_tensor.coalesce().size()

    n = size[0]
    thresholds = np.random.uniform(0.3, 0.6, n)
    active = set(initial_active)
    
    for _ in range(iterations):
        new_active = set()
        for node in range(n):
            if node not in active:
                # Get neighbors
                neighbors = indices[1][indices[0] == node].tolist()
                active_neighbors = len([nbr for nbr in neighbors if nbr in active])
                neighbor_count = len(neighbors)
                if neighbor_count == 0:
                    neighbor_count = 1
                if active_neighbors / neighbor_count >= thresholds[node]:
                    new_active.add(node)
        if not new_active:
            break
        active.update(new_active)
    
    active_list = [1 if node in active else 0 for node in range(n)]
    return active_list

def independent_cascade_model(sparse_tensor, initial_active, iterations):
    indices = sparse_tensor.coalesce().indices()
    values = sparse_tensor.coalesce().values()
    size = sparse_tensor.size()

    n = size[0]
    active = set(initial_active)
    
    for _ in range(iterations):
        new_active = set()
        for node in active:
            # Get neighbors and their activation probabilities
            neighbors = indices[1][indices[0] == node].tolist()
            probabilities = values[indices[0] == node].tolist()
            for nbr, prob in zip(neighbors, probabilities):
                if nbr not in active and nbr not in new_active and random.random() < prob:
                    new_active.add(nbr)
            print(node)
        if not new_active:
            break
        active.update(new_active)
    
    active_list = [1 if node in active else 0 for node in range(n)]
    return active_list

def sis_model(sparse_tensor, initial_active, iterations, infection_prob, recovery_prob):
    indices = sparse_tensor.coalesce().indices()
    values = sparse_tensor.coalesce().values()
    size = sparse_tensor.size()

    n = size[0]
    active = set(initial_active)
    
    for _ in range(iterations):
        new_active = set(active)
        for node in range(n):
            if node in active:
                if random.random() < recovery_prob:
                    new_active.remove(node)
            else:
                # Get neighbors
                neighbors = indices[1][indices[0] == node].tolist()
                if any(nbr in active and random.random() < infection_prob for nbr in neighbors):
                    new_active.add(node)
        if new_active == active:
            break
        active = new_active
    
    active_list = [1 if node in active else 0 for node in range(n)]
    return active_list

def run_influence_propagation(model, adj_matrix, initial_nodes, iterations, infection_prob=0.1, recovery_prob=0.1):
    if model == 'LT':
        return linear_threshold_model(adj_matrix, initial_nodes, iterations)
    elif model == 'IC':
        return independent_cascade_model(adj_matrix, initial_nodes, iterations)
    elif model == 'SIS':
        return sis_model(adj_matrix, initial_nodes, iterations, infection_prob, recovery_prob)
    else:
        raise ValueError("Invalid model type. Choose from 'LT', 'IC', or 'SIS'.")


def generate_sbm_graph(n, p_in, p_out):
    """
    Generate a Stochastic Block Model (SBM) graph with random community sizes.

    Parameters:
    n (int): Total number of nodes.
    p_in (float): Probability of edges within communities.
    p_out (float): Probability of edges between communities.

    Returns:
    G (networkx.Graph): The generated SBM graph.
    adj_matrix (numpy.matrix): Adjacency matrix of the generated graph.
    """
    num_communities = np.random.randint(5, 15)  # random number of communities between 5 and 15
    sizes = np.random.multinomial(n, np.ones(num_communities) / num_communities)
    p_matrix = np.full((num_communities, num_communities), p_out)
    np.fill_diagonal(p_matrix, p_in)
    G = nx.stochastic_block_model(sizes, p_matrix, seed=42)
    adj_matrix = nx.adjacency_matrix(G).todense()
    return G, adj_matrix
    


'''# Parameters for the SBM
n = 500  # total number of nodes
# Probability of edges within communities
p_in = 0.8
# Probability of edges between communities
p_out = 0.05
per = 20
model = 'LT'
G, adj_matrix = generate_sbm_graph(n, p_in, p_out)'''

parser = argparse.ArgumentParser(description="GenIM")
datasets = ['jazz', 'cora_ml', 'power_grid', 'netscience', 'random5']
parser.add_argument("-d", "--dataset", default="digg", type=str,
                    help="one of: {}".format(", ".join(sorted(datasets))))
diffusion = ['IC', 'LT', 'SIS']
parser.add_argument("-dm", "--diffusion_model", default="LT", type=str,
                    help="one of: {}".format(", ".join(sorted(diffusion))))
seed_rate = [1, 5, 10, 20]
parser.add_argument("-sp", "--seed_rate", default=10, type=int,
                    help="one of: {}".format(", ".join(str(sorted(seed_rate)))))
mode = ['Normal', 'Budget Constraint']
parser.add_argument("-m", "--mode", default="normal", type=str,
                    help="one of: {}".format(", ".join(sorted(mode))))
args = parser.parse_args(args=[])


# Load the sparse tensor
sparse_tensor = torch.load('digg_network.pt')

# Extract indices
indices = sparse_tensor.coalesce().indices()

# Convert indices to a list of edges
edges = [(indices[0, i].item(), indices[1, i].item()) for i in range(indices.shape[1])]

# Create NetworkX graph
G = nx.Graph()  # Use nx.DiGraph() for directed graphs
G.add_edges_from(edges)


x_list = []
y_list = []
for i in range(2):
    activated_nodes, x = select_random_nodes(G, args.seed_rate)
    x_list.append(x)
    y_values = []
    for j in range(1):
        activated_nodes_lt = run_influence_propagation(args.diffusion_model, sparse_tensor, activated_nodes, 10, 0.001, 0.001)
        y_values.append(activated_nodes_lt)
    print(i)
    y_list.append(np.mean(np.array(y_values), axis=0).tolist())
    
indices = sparse_tensor.coalesce().indices().cpu().numpy()
row = indices[0, :]
col = indices[1, :]

sparse_matrix = csr_matrix((np.ones(len(row)), (row, col)), shape=sparse_tensor.size())
combined_tensor = combine_lists_to_tensor(x_list, y_list)
# Example data to save
graph = {
    'adj': sparse_matrix, 
    'inverse_pairs': combined_tensor
}

# File path to save the data
file_path = 'data/' + args.dataset + '_mean_' + args.diffusion_model + str(10*args.seed_rate) + '.SG'

# Save the graph dictionary to file using pickle
with open(file_path, "wb") as file:
    pickle.dump(graph, file)

print(f"Saved graph data to {file_path}")

