import numpy as np
import pandas as pd
from read_utils import prepare_sample
from correlation_network import build_adjacency_matrix
from network_features import extract_network_features
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import Amplitude, PersistenceEntropy

amplitude_modes = [
    "bottleneck",
    "betti",
    "landscape",
    "silhouette",
    "heat",
    "persistence_image",
    "landscape",
]


def build_network_features(sample_size=None, window_size=60):
    sample, index = prepare_sample(sample_size, window_size)
    matrices = [build_adjacency_matrix(df) for df in sample]
    network_features = extract_network_features(matrices)
    network_features.set_index(index, inplace=True)
    return network_features


def build_topology_features(sample_size=None, window_size=60, dimensions=2):
    sample, index = prepare_sample(sample_size, window_size)
    dimensions = range(dimensions)
    VR = VietorisRipsPersistence(homology_dimensions=list(dimensions))
    diagrams = VR.fit_transform(sample)

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

    topology_features = persistence_results
    for df in amplitude_results.values():
        topology_features = topology_features.join(df)

    topology_features.set_index(index, inplace=True)
    return topology_features
