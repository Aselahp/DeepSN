import numpy as np
import scipy.sparse as sp
import networkx as nx
import random

import random
import networkx as nx

def linear_threshold_model_v2(G, initial_active, max_iterations=200):
    # Randomly initialize thresholds between 0.3 and 0.6
    thresholds = {node: random.uniform(0.3, 0.6) for node in G.nodes()}
    
    # Normalize edge weights so that the sum of weights for incoming edges to each node is at most 1
    for node in G.nodes:
        in_edges = list(G.in_edges(node, data=True))
        weight_sum = sum(data['weight'] for _, _, data in in_edges)
        if weight_sum > 0:
            for u, v, data in in_edges:
                data['weight'] /= weight_sum
    
    active_nodes = set(initial_active)
    newly_active_nodes = set(initial_active)
    iterations = 0
    
    while newly_active_nodes and iterations < max_iterations:
        next_active_nodes = set()
        for node in G.nodes():
            if node not in active_nodes:
                neighbors = list(G.neighbors(node))
                influence_sum = sum(G[u][node]['weight'] for u in neighbors if u in active_nodes)
                if influence_sum >= thresholds[node]:
                    next_active_nodes.add(node)
        
        newly_active_nodes = next_active_nodes
        active_nodes.update(newly_active_nodes)
        iterations += 1
    
    print(f'Number of active nodes: {len(active_nodes)}')
    return len(active_nodes)


def linear_threshold_model(G, initial_active):
    
    # Randomly initialize thresholds between 0.3 and 0.6
    thresholds = {node: random.uniform(0.3, 0.6) for node in G.nodes()}
    
    # Normalize edge weights so that the sum of weights for incoming edges to each node is at most 1
    for node in G.nodes:
        in_edges = list(G.in_edges(node, data=True))
        weight_sum = sum(data['weight'] for _, _, data in in_edges)
        if weight_sum > 0:
            for u, v, data in in_edges:
                data['weight'] /= weight_sum
    
    active_nodes = set(initial_active)
    newly_active_nodes = set(initial_active)
    
    while newly_active_nodes:
        next_active_nodes = set()
        for node in G.nodes():
            if node not in active_nodes:
                neighbors = list(G.neighbors(node))
                influence_sum = sum(G[u][node]['weight'] for u in neighbors if u in active_nodes)
                if influence_sum >= thresholds[node]:
                    next_active_nodes.add(node)
        
        newly_active_nodes = next_active_nodes
        active_nodes.update(newly_active_nodes)
    
    print(len(active_nodes))
    return len(active_nodes)


def independent_cascade_model_v2(G, initial_active, max_iterations=200):
    # Generate a random seed for this run

    active_nodes = set(initial_active)
    newly_active_nodes = set(initial_active)
    iterations = 0
    
    while newly_active_nodes and iterations < max_iterations:
        next_active_nodes = set()
        for node in newly_active_nodes:
            for neighbor in G.neighbors(node):
                if neighbor not in active_nodes:
                    # Calculate activation probability
                    edge_data = G[node][neighbor]
                    probability = 1.0 / G.in_degree(neighbor)
                    
                    # Activate with probability
                    if random.random() <= probability:
                        next_active_nodes.add(neighbor)
        
        newly_active_nodes = next_active_nodes
        active_nodes.update(newly_active_nodes)
        iterations += 1
    
    print(f'Number of active nodes: {len(active_nodes)}')
    return len(active_nodes)


def independent_cascade_model(G, initial_active):
    # Generate a random seed for this run

    active_nodes = set(initial_active)
    newly_active_nodes = set(initial_active)
    
    while newly_active_nodes:
        next_active_nodes = set()
        for node in newly_active_nodes:
            for neighbor in G.neighbors(node):
                if neighbor not in active_nodes:
                    # Calculate activation probability
                    edge_data = G[node][neighbor]
                    probability = 1.0 / G.in_degree(neighbor)
                    
                    # Activate with probability
                    if random.random() <= probability:
                        next_active_nodes.add(neighbor)
        
        newly_active_nodes = next_active_nodes
        active_nodes.update(newly_active_nodes)
    print(len(active_nodes))
    return len(active_nodes)

def sis_model(G, initial_infected, infection_probability=0.001, recovery_probability=0.001,k=20):
   # Initialize node states
    node_states = {node: 'S' for node in G.nodes()}
    for node in initial_infected:
        node_states[node] = 'I'
    
    stable_iterations = 0
    total_infected = sum(1 for state in node_states.values() if state == 'I')
    
    for iteration in range(200):
        next_node_states = node_states.copy()
        
        # Iterate over nodes and update states
        for node in G.nodes():
            current_status = node_states[node]
            event_prob = np.random.random_sample()
            neighbors = list(G.neighbors(node))
            
            if current_status == 'S':
                infected_neighbors = [neighbor for neighbor in neighbors if node_states[neighbor] == 'I']
                if len(infected_neighbors) > 0 and event_prob < 1 - (1 - infection_probability) ** len(infected_neighbors):
                    next_node_states[node] = 'I'
            
            elif current_status == 'I':
                if event_prob < recovery_probability:
                    next_node_states[node] = 'S'
        
        # Calculate the total number of infected nodes
        new_total_infected = sum(1 for state in next_node_states.values() if state == 'I')
        
        # Check if the total number of infected nodes has changed
        if new_total_infected == total_infected:
            stable_iterations += 1
            if stable_iterations >= k:
                break
        else:
            stable_iterations = 0
        
        # Update node states and total infected count
        node_states = next_node_states
        total_infected = new_total_infected
    
    # Return the count of infected nodes
    infected_nodes_count = sum(1 for state in node_states.values() if state == 'I')
    return infected_nodes_count