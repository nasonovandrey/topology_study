import numpy as np
import pandas as pd
from read_utils import read, window_generator
from correlation_network import build_adjacency_matrix
from graph_features import extract_graph_features
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


def build_graph_features(sample, index):
    matrices = [build_adjacency_matrix(df) for df in sample]
    graph_features = extract_graph_features(matrices)
    graph_features.set_index(index, inplace=True)
    return graph_features


def build_top_features(sample, index, dimensions=2):
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

    top_features = persistence_results
    for df in amplitude_results.values():
        top_features = top_features.join(df)

    top_features.set_index(index, inplace=True)
    return top_features
