import numpy as np
import pandas as pd
from build_utils import build_network_features, build_topology_features

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["network", "topology"])
    parser.add_argument("--sample_size", default=None, type=int)
    parser.add_argument("--window_size", default=60, type=int)
    parser.add_argument("--dimensions", default=3, type=int)
    parser.add_argument("--lag", default=5, type=int)
    args = parser.parse_args()

    mode = args.mode
    sample_size = args.sample_size
    window_size = args.window_size
    dimensions = args.dimensions
    lag = args.lag

    if mode == "network":
        network_features = build_network_features(
            sample_size=sample_size, window_size=window_size, max_lag=lag
        )
        if sample_size is None:
            sample_size = len(network_features)
        network_features.to_csv(
            f"features/network_features_s{sample_size}_w{window_size}_l{lag}.csv"
        )
    elif mode == "topology":
        topology_features = build_topology_features(
            sample_size=sample_size, window_size=window_size, dimensions=dimensions
        )
        if sample_size is None:
            sample_size = len(topology_features)
        topology_features.to_csv(
            f"features/topology_features_s{sample_size}_w{window_size}_d{dimensions}.csv"
        )
