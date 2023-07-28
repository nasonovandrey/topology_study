import numpy as np
import pandas as pd
from extract_utils import (
    load_network_structures,
    load_topology_structures,
    extract_network_features,
    extract_topology_features,
)

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["network", "topology"])
    parser.add_argument("--sample_size", default=None, type=int)
    parser.add_argument("--window_size", default=60, type=int)
    parser.add_argument("--dim", default=3, type=int)
    parser.add_argument("--lag", default=5, type=int)
    args = parser.parse_args()

    mode = args.mode
    sample_size = args.sample_size
    window_size = args.window_size
    dim = args.dim
    lag = args.lag

    if mode == "network":
        structures = load_network_structures(sample_size, window_size, lag)
        features = extract_network_features(
            np.array(list(structures.values())), list(structures.keys())
        )
        features.to_csv(
            f"features/network_features_s{sample_size}_w{window_size}_l{lag}.csv"
        )
    elif mode == "topology":
        structures = load_topology_structures(sample_size, window_size, dim)
        features = extract_topology_features(
            np.array(list(structures.values())), list(structures.keys())
        )
        features.to_csv(
            f"features/topology_features_s{sample_size}_w{window_size}_d{dim}.csv"
        )
