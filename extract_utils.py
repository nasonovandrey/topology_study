import numpy as np
from collections import OrderedDict
from datetime import datetime
from numpy import genfromtxt
import pandas as pd
from read_utils import prepare_sample
from network import build_adjacency_matrix
from network_features import (
    average_links,
    density,
    num_connected_components,
    shortest_average_path,
    clustering_coefficient,
    diameter,
)
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import Amplitude, PersistenceEntropy
import networkx as nx
from pathlib import Path

amplitude_modes = [
    "bottleneck",
    "betti",
    "landscape",
    "silhouette",
    "heat",
    "persistence_image",
    "landscape",
]


def load_topology_structures(sample_size, window_size, dim):
    files = list(
        map(
            lambda p: p.name,
            Path("structures").glob(
                f"topology*_s{sample_size}_w{window_size}_d{dim}.csv"
            ),
        )
    )
    if not files:
        raise Exception("No structures were generated with these parameters")
    ts_index = sorted(map(lambda f: f.split("_")[1][1:], files))
    results = OrderedDict()
    for ts in ts_index:
        results[datetime.strptime(ts, "%Y%m%d%H%M%S")] = genfromtxt(
            f"structures/topology_t{ts}_s{sample_size}_w{window_size}_d{dim}.csv",
            delimiter=",",
        )
    return results


def load_network_structures(sample_size, window_size, lag):
    files = list(
        map(
            lambda p: p.name,
            Path("structures").glob(
                f"network*_s{sample_size}_w{window_size}_l{lag}.csv"
            ),
        )
    )
    if not files:
        raise Exception("No structures were generated with these parameters")
    ts_index = sorted(map(lambda f: f.split("_")[1][1:], files))
    results = OrderedDict()
    for ts in ts_index:
        results[datetime.strptime(ts, "%Y%m%d%H%M%S")] = genfromtxt(
            f"structures/network_t{ts}_s{sample_size}_w{window_size}_l{lag}.csv",
            delimiter=",",
        )
    return results


def build_graph(adj_matrix):
    G = nx.Graph()
    G.add_nodes_from(range(len(adj_matrix)))

    for i in range(len(adj_matrix)):
        for j in range(i + 1, len(adj_matrix[i])):
            if adj_matrix[i][j] != 0:
                G.add_edge(i, j)
    return G


def extract_network_features(matrices, index):
    features = {
        "average_links": list(),
        "shortest_average_path": list(),
        "density": list(),
        "num_connected_components": list(),
        "clustering_coefficient": list(),
        "diameter": list(),
    }
    for matrix in matrices:
        graph = build_graph(matrix)
        features["average_links"].append(average_links(graph))
        features["shortest_average_path"].append(shortest_average_path(graph))
        features["density"].append(density(graph))
        features["num_connected_components"].append(num_connected_components(graph))
        features["clustering_coefficient"].append(clustering_coefficient(graph))
        features["diameter"].append(diameter(graph))

    features = pd.DataFrame(features)
    features.index = index
    return features


def extract_topology_features(diagrams, index):
    PE = PersistenceEntropy(n_jobs=-1)
    persistence_results = PE.fit_transform(diagrams)
    dimensions = list(range(persistence_results.shape[-1]))

    entropy_columns = [f"entropy_hom{dim}" for dim in dimensions]
    persistence_results = pd.DataFrame(
        columns=entropy_columns, data=persistence_results
    )

    amplitude_results = {}
    for mode in amplitude_modes:
        amplitude_columns = [f"{mode}_hom{dim}" for dim in dimensions]
        A = Amplitude(mode, n_jobs=-1)
        amplitude_results[mode] = pd.DataFrame(
            columns=amplitude_columns, data=A.fit_transform(diagrams)
        )

    features = persistence_results
    for df in amplitude_results.values():
        features = features.join(df)

    features.index = index
    return features
