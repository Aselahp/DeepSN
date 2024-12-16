import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from scipy.sparse import csr_matrix
import pickle

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


def generate_kronecker_graph(desired_nodes):
    # Define the initial seed graph (adjacency matrix)
    P = np.array([[0, 1], [1, 1]])
    
    # Calculate the number of iterations needed
    k = int(np.ceil(np.log2(desired_nodes)))

    # Generate the Kronecker Graph
    G = P
    for _ in range(k - 1):
        G = np.kron(G, P)

    # Convert the adjacency matrix to a NetworkX graph
    G = nx.from_numpy_array(G)
    
    # Get the adjacency matrix of the graph
    adj_matrix = nx.to_numpy_array(G)
    
    return G, adj_matrix


def generate_barabasi_albert(n, m):
    """
    Generate a Barabási-Albert (BA) model network with `n` nodes
    and `m` edges to attach from a new node to existing nodes.
    
    Parameters:
    - n (int): Number of nodes in the network.
    - m (int): Number of edges to attach from a new node to existing nodes.
    
    Returns:
    - G (networkx.Graph): Generated Barabási-Albert network.
    - adj_matrix (np.ndarray): Adjacency matrix of the generated network.
    """
    # Generate a Barabási-Albert preferential attachment graph
    G = nx.barabasi_albert_graph(n, m)
    
    # Get the adjacency matrix of the graph
    adj_matrix = nx.to_numpy_array(G)
    
    return G, adj_matrix


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

def linear_threshold_model(adj_matrix, initial_active, iterations):
    n = len(adj_matrix)
    thresholds = np.random.uniform(0.3, 0.6, n)
    active = set(initial_active)
    
    for _ in range(iterations):
        new_active = set()
        for node in range(n):
            if node not in active:
                neighbors = np.where(adj_matrix[node] > 0)[0]
                active_neighbors = len([nbr for nbr in neighbors if nbr in active])
                if active_neighbors / len(neighbors) > thresholds[node]:
                    new_active.add(node)
        if not new_active:
            break
        active.update(new_active)
    
    active_list = [1 if node in active else 0 for node in range(n)]
    return active_list

def independent_cascade_model(adj_matrix, initial_active, iterations):
    n = len(adj_matrix)
    active = set(initial_active)
    
    for _ in range(iterations):
        new_active = set()
        for node in active:
            neighbors = np.where(adj_matrix[node] > 0)[0]
            for nbr in neighbors:
                if nbr not in active and nbr not in new_active and random.random() < adj_matrix[node, nbr]:
                    new_active.add(nbr)
        if not new_active:
            break
        active.update(new_active)
    
    active_list = [1 if node in active else 0 for node in range(n)]
    return active_list

def sis_model(adj_matrix, initial_active, iterations, infection_prob, recovery_prob):
    n = len(adj_matrix)
    active = set(initial_active)
    
    for _ in range(iterations):
        new_active = set(active)
        for node in range(n):
            if node in active:
                if random.random() < recovery_prob:
                    new_active.remove(node)
            else:
                neighbors = np.where(adj_matrix[node] > 0)[0]
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
    


# Parameters for the SBM
n = 1000  # total number of nodes
# Probability of edges within communities
p_in = 0.8
# Probability of edges between communities
p_out = 0.05
per = 20
model = 'LT'
G, adj_matrix = generate_sbm_graph(n, p_in, p_out)
#G, adj_matrix = generate_barabasi_albert(1000, 5)

# Desired number of nodes
#desired_nodes = 1000

# Generate the Kronecker Graph
#G, adj_matrix = generate_kronecker_graph(desired_nodes)


x_list = []
y_list = []
for i in range(100):
    activated_nodes, x = select_random_nodes(G, per)
    activated_nodes_lt = run_influence_propagation(model, adj_matrix, activated_nodes, 100, 0.001, 0.001)
    x_list.append(x)
    y_list.append(activated_nodes_lt)
sparse_matrix = csr_matrix(adj_matrix)
combined_tensor = combine_lists_to_tensor(x_list, y_list)
# Example data to save
graph = {
    'adj': sparse_matrix, 
    'inverse_pairs': combined_tensor.float()
}
print("Number of nodes:", G.number_of_nodes())

# File path to save the data
file_path = 'data/' + 'SBM' + '_mean_' + model + str(10*per) + '.SG'

# Save the graph dictionary to file using pickle
with open(file_path, "wb") as file:
    pickle.dump(graph, file)

print(f"Saved graph data to {file_path}")

