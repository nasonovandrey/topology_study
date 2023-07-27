import numpy as np
import pandas as pd

# TBD: Estimate the robustness and resilience, estimate the number of feedback loops


def dfs(node, adj_matrix, visited):
    # Mark the current node as visited
    visited[node] = True

    # Perform DFS on neighbors of the current node
    for neighbor in range(len(adj_matrix)):
        if adj_matrix[node, neighbor] == 1 and not visited[neighbor]:
            dfs(neighbor, adj_matrix, visited)


def average_links(adj_matrix):
    # Step 1: Compute the sum of each row in the adjacency matrix
    outdegrees = np.sum(adj_matrix, axis=1)
    # Step 2: Calculate the average number of causal links per node
    average_links_per_node = np.mean(outdegrees)
    return average_links_per_node


def density(adj_matrix):
    num_nodes = adj_matrix.shape[0]
    num_edges = np.sum(adj_matrix)

    # Calculate the density of the network
    density = num_edges / (num_nodes * (num_nodes - 1))

    return density


def clusters_count(adj_matrix):
    num_nodes = len(adj_matrix)
    visited = [False] * num_nodes
    num_clusters = 0

    # Perform DFS from each unvisited node
    for node in range(num_nodes):
        if not visited[node]:
            dfs(node, adj_matrix, visited)
            num_clusters += 1

    return num_clusters


def extract_graph_features(matrices):
    features = {"average_links": list(), "density": list(), "clusters_count": list()}
    for matrix in matrices:
        features["average_links"].append(average_links(matrix))
        features["density"].append(density(matrix))
        features["clusters_count"].append(clusters_count(matrix))

    return pd.DataFrame(features)
