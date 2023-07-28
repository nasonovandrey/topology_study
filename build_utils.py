import numpy as np
import pandas as pd
from datetime import datetime
from read_utils import prepare_sample
from network import build_adjacency_matrix

from gtda.homology import VietorisRipsPersistence


def build_network_structures(sample_size=None, window_size=60, lag=5):
    sample, index = prepare_sample(sample_size, window_size)
    sample_size = len(sample)
    print(f"Saving networks of size {sample_size} between {index[0]} and {index[-1]}")
    for i, (ts, df) in enumerate(zip(index, sample)):
        matrix = build_adjacency_matrix(df, max_lag=lag)
        timestamp = datetime.strftime(ts, "%Y%m%d%H%M%S")
        print(matrix)
        filename = (
            f"structures/network_t{timestamp}_s{sample_size}_w{window_size}_l{lag}.csv"
        )
        np.savetxt(filename, matrix, delimiter=",")
        print(f"File {filename} saved, {i+1} of {sample_size}")


def build_topology_structures(sample_size=None, window_size=60, dim=5):
    sample, index = prepare_sample(sample_size, window_size)
    sample_size = len(sample)
    dimensions = range(dim)
    VR = VietorisRipsPersistence(homology_dimensions=list(dimensions))
    diagrams = VR.fit_transform(sample)
    breakpoint()
    for i, (ts, df) in enumerate(zip(index, diagrams)):
        timestamp = datetime.strftime(ts, "%Y%m%d%H%M%S")
        filename = (
            f"structures/topology_t{timestamp}_s{sample_size}_w{window_size}_d{dim}.csv"
        )
        print(df)
        np.savetxt(filename, df, delimiter=",")
        print(f"File {filename} saved, {i+1} of {sample_size}")
