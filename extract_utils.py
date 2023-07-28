import numpy as np
from collections import OrderedDict
from datetime import datetime
from numpy import genfromtxt
import pandas as pd
from read_utils import prepare_sample
from network import build_adjacency_matrix
from network_features import average_links, density, clusters_count
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import Amplitude, PersistenceEntropy
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


def extract_network_features(matrices, index):
    features = {"average_links": list(), "density": list(), "clusters_count": list()}
    for matrix in matrices:
        features["average_links"].append(average_links(matrix))
        features["density"].append(density(matrix))
        features["clusters_count"].append(clusters_count(matrix))

    features = pd.DataFrame(features)
    features.index = index
    return features


def extract_topology_features(diagrams, index):
    dimensions = list(range(diagrams.shape[-1] + 1))
    PE = PersistenceEntropy()
    persistence_results = PE.fit_transform(diagrams)

    entropy_columns = [f"entropy_hom{dim}" for dim in dimensions]
    persistence_results = pd.DataFrame(
        columns=entropy_columns, data=persistence_results
    )

    amplitude_results = {}
    for mode in amplitude_modes:
        amplitude_columns = [f"{mode}_hom{dim}" for dim in dimensions]
        A = Amplitude(mode)
        amplitude_results[mode] = pd.DataFrame(
            columns=amplitude_columns, data=A.fit_transform(diagrams)
        )

    features = persistence_results
    for df in amplitude_results.values():
        features = features.join(df)

    features.index = index
    return features
