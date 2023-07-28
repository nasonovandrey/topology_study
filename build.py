import numpy as np
import pandas as pd
from build_utils import build_topology_structures, build_network_structures

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["network", "topology"])
    parser.add_argument("--sample_size", default=None, type=int)
    parser.add_argument("--window_size", default=60, type=int)
    parser.add_argument("--dim", default=5, type=int)
    parser.add_argument("--lag", default=5, type=int)
    args = parser.parse_args()

    mode = args.mode
    sample_size = args.sample_size
    window_size = args.window_size
    dim = args.dim
    lag = args.lag

    if mode == "network":
        build_network_structures(
            sample_size=sample_size, window_size=window_size, lag=lag
        )
    elif mode == "topology":
        build_topology_structures(
            sample_size=sample_size, window_size=window_size, dim=dim
        )
