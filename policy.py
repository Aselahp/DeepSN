import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class NodeScoringNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NodeScoringNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Define layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)
        init.orthogonal_(self.fc1.weight)
        init.orthogonal_(self.fc2.weight)
        
    def select_nodes_proportional(self, scores, cluster_assignment, budget):
        # Calculate cluster sizes
        cluster_sizes = torch.bincount(cluster_assignment)
        
        # Initialize output tensor
        selected_nodes = torch.zeros_like(scores)
        
        # Initialize budget counter
        remaining_budget = budget
        
        # Iterate over each cluster
        for cluster_idx in range(torch.max(cluster_assignment) + 1):
            # Get indices of nodes in the current cluster
            cluster_indices = torch.nonzero(cluster_assignment == cluster_idx, as_tuple=False).squeeze()
            
            # Calculate the number of nodes to select from this cluster
            cluster_size = cluster_sizes[cluster_idx].item()
            nodes_to_select = min(int(round(budget * (cluster_size / len(scores)))), remaining_budget)
            
            if nodes_to_select == 0:
                continue
            
            # Filter scores for nodes in the current cluster
            cluster_scores = scores[cluster_indices]
            
            # Sort scores in descending order
            sorted_scores, sorted_indices = torch.sort(cluster_scores.squeeze(), descending=True)
            
            # Select nodes from this cluster proportional to its size
            selected_indices = cluster_indices[sorted_indices[:nodes_to_select]]
            
            # Update the selected nodes tensor
            selected_nodes[selected_indices] = 1
            
            # Update remaining budget
            remaining_budget -= nodes_to_select
        
        # If there's remaining budget, select the highest score nodes that are not marked
        if remaining_budget > 0:

            # Get the indices of nodes that are not selected
            unselected_indices = torch.nonzero(selected_nodes == 0, as_tuple=False)[:, 0].unsqueeze(1)

            if unselected_indices.numel() > 0:
                # Filter scores for nodes that are not selected
                unselected_scores = scores[unselected_indices]
            
                # Sort scores in descending order
                sorted_unselected_scores, sorted_unselected_indices = torch.sort(unselected_scores.squeeze(), descending=True)
            
                # Select remaining nodes from the highest score unselected nodes
                remaining_selected_indices = unselected_indices[sorted_unselected_indices[:remaining_budget]]
            
                # Update the selected nodes tensor
                selected_nodes[remaining_selected_indices] = 1

        print(selected_nodes.sum())
        return selected_nodes


    def select_nodes_proportional_differentiable(self, scores, cluster_assignment, budget):
        # Calculate cluster sizes
        cluster_sizes = torch.bincount(cluster_assignment)
    
        # Initialize probabilities tensor
        probabilities = torch.zeros_like(scores)
    
        # Initialize budget counter
        remaining_budget = budget
    
        # Iterate over each cluster
        for cluster_idx in range(torch.max(cluster_assignment) + 1):
            # Get indices of nodes in the current cluster
            cluster_indices = torch.nonzero(cluster_assignment == cluster_idx, as_tuple=False).squeeze()
        
            # Calculate the number of nodes to select from this cluster
            cluster_size = cluster_sizes[cluster_idx].item()
            nodes_to_select = min(int(round(budget * (cluster_size / len(scores)))), remaining_budget)
        
            if nodes_to_select == 0:
                continue
        
            # Filter scores for nodes in the current cluster
            cluster_scores = scores[cluster_indices]
        
            # Use softmax to get probabilities
            softmax_scores = F.softmax(cluster_scores, dim=0)
        
            # Scale probabilities to get the number of nodes to select
            scaled_probabilities = softmax_scores * nodes_to_select
        
            # Update probabilities tensor
            probabilities[cluster_indices] = scaled_probabilities
        
            # Update remaining budget
            remaining_budget -= nodes_to_select

        # If there's remaining budget, select the highest score nodes that are not marked
        if remaining_budget > 0:
            # Get the indices of nodes that are not selected
            unselected_indices = torch.nonzero(probabilities == 0, as_tuple=False)[:, 0].unsqueeze(1)

            if unselected_indices.numel() > 0:
                # Filter scores for nodes that are not selected
                unselected_scores = scores[unselected_indices]
        
                # Use softmax to get probabilities
                softmax_unselected_scores = F.softmax(unselected_scores, dim=0)
        
                # Scale probabilities to get the remaining nodes to select
                remaining_probabilities = softmax_unselected_scores * remaining_budget
        
                # Update probabilities tensor
                probabilities[unselected_indices] = remaining_probabilities

        # Normalize the probabilities so that they sum up to 1
        probabilities = probabilities / (probabilities.sum()++ 1e-10)

        # Handle cases where the budget exceeds the number of nodes
        num_samples = min(int(budget), probabilities.size(0))
    
        if num_samples > 0:
            # Sample nodes based on the probabilities
            if num_samples < probabilities.size(0):
                top_values, top_indices = torch.topk(probabilities.squeeze(), num_samples, largest=True, sorted=False)
                sampled_indices = top_indices
            else:
                # If the number of samples is greater than or equal to the number of available indices, select all nodes
                sampled_indices = torch.arange(probabilities.size(0))
        
            # Create a tensor to mark selected nodes
            selected_nodes = torch.zeros_like(scores)
            sampled_indices = sampled_indices.long()
            selected_nodes[sampled_indices] = 1
        else:
            # No nodes to select
            selected_nodes = torch.zeros_like(scores)
    
        return selected_nodes
    
    
    def forward(self, x, c, k):
        # Forward pass through the network
        x = self.fc1(x)
        x = self.dropout(x)
        #x = self.elu(x)
        x = self.fc2(x)
        #x = self.dropout(x)
        x = self.sigmoid(x)
        x = self.select_nodes_proportional(x, c, k)
        return x






