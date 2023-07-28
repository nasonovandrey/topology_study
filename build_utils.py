import numpy as np
import pandas as pd
from datetime import datetime
from read_utils import prepare_sample
from network import build_adjacency_matrix
import multiprocessing as mp

from gtda.homology import VietorisRipsPersistence


# Define a worker function for each process to execute
def process_network(ts, df, lag, postfix):
    matrix = build_adjacency_matrix(df, max_lag=lag)
    timestamp = datetime.strftime(ts, "%Y%m%d%H%M%S")
    filename = f"structures/network_t{timestamp}_{postfix}.csv"
    np.savetxt(filename, matrix, delimiter=",")


def build_network_structures(sample_size=None, window_size=60, lag=5):
    sample, index = prepare_sample(sample_size, window_size)
    sample_size = len(sample)
    postfix = f"s{sample_size}_w{window_size}_l{lag}"
    print(f"Saving networks of size {sample_size} between {index[0]} and {index[-1]}")

    # Create a pool of worker processes
    num_processes = mp.cpu_count()  # Use the number of available CPU cores
    pool = mp.Pool(processes=num_processes)

    # Use pool.starmap to execute the worker function for each item in the loop in parallel
    pool.starmap(
        process_network, [(ts, df, lag, postfix) for ts, df in zip(index, sample)]
    )

    # Close the pool to release resources
    pool.close()
    pool.join()


def build_topology_structures(sample_size=None, window_size=60, dim=5):
    sample, index = prepare_sample(sample_size, window_size)
    sample_size = len(sample)
    dimensions = range(dim)
    VR = VietorisRipsPersistence(homology_dimensions=list(dimensions), n_jobs=-1)
    diagrams = VR.fit_transform(sample)
    print(f"Saving diagrams of size {sample_size} between {index[0]} and {index[-1]}")
    for i, (ts, df) in enumerate(zip(index, diagrams)):
        timestamp = datetime.strftime(ts, "%Y%m%d%H%M%S")
        filename = (
            f"structures/topology_t{timestamp}_s{sample_size}_w{window_size}_d{dim}.csv"
        )
        np.savetxt(filename, df, delimiter=",")
        print(f"File {filename} saved, {i+1} of {sample_size}")
