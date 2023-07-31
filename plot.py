import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as n

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=int)
    parser.add_argument("--window_size", type=int)
    parser.add_argument("--dimensions", type=int)
    parser.add_argument("--lag", type=int)
    parser.add_argument("--topology", type=str)
    parser.add_argument("--network", type=str)
    args = parser.parse_args()

    sample_size = args.sample_size
    window_size = args.window_size
    dimensions = args.dimensions
    lag = args.lag

    topology_features = pd.read_csv(
        f"features/topology_features_s{sample_size}_w{window_size}_d{dimensions}.csv",
        index_col=[0],
    )

    network_features = pd.read_csv(
        f"features/network_features_s{sample_size}_w{window_size}_l{lag}.csv",
        index_col=[0],
    )

    scaler = MinMaxScaler()

    features_df = pd.concat([topology_features, network_features], axis=1)
    selected_columns = [args.topology, args.network]
    selected_df = features_df[selected_columns]
    scaled_data = scaler.fit_transform(selected_df.values)
    scaled_df = pd.DataFrame(
        scaled_data,
        columns=["heat_hom0", "num_connected_components"],
        index=features_df.index,
    )

    plt.figure(figsize=(10, 6))
    plt.plot(
        scaled_df.index, scaled_df[args.topology], color="green", label=args.topology
    )
    plt.plot(scaled_df.index, scaled_df[args.network], color="blue", label=args.network)
    plt.show()
