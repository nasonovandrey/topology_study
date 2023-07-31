import numpy as np
import pandas as pd
import networkx as nx


# ---Helpers----------------------------
def dfs(node, adj_matrix, visited):
    visited[node] = True
    for neighbor in range(len(adj_matrix)):
        if adj_matrix[node, neighbor] == 1 and not visited[neighbor]:
            dfs(neighbor, adj_matrix, visited)


# --------------------------------------


def diameter(G):
    diameters = []
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)
        try:
            diameter = nx.diameter(subgraph)
            diameters.append(diameter)
        except nx.NetworkXError:
            pass

    max_diameter = max(diameters)
    return max_diameter


def shortest_average_path(G):
    avg_shortest_paths = []
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)
        try:
            avg_shortest_path = nx.average_shortest_path_length(subgraph)
            avg_shortest_paths.append(avg_shortest_path)
        except nx.NetworkXError:
            pass  # Ignore isolated nodes or components with no paths.

    if avg_shortest_paths:
        avg_path = sum(avg_shortest_paths) / len(avg_shortest_paths)
        return avg_path
    else:
        return None


def clustering_coefficient(G):
    cc = nx.average_clustering(G)
    return cc


def average_links(G):
    total_links = sum([G.degree(node) for node in G.nodes()])
    num_nodes = G.number_of_nodes()
    average_links = total_links / num_nodes
    return average_links


def density(G):
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = 2 * num_edges / (num_nodes * (num_nodes - 1))

    return density


def num_connected_components(G):
    num_components = nx.number_connected_components(G)
    return num_components
