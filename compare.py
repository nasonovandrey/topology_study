import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

import argparse


def present(result):
    filter_cols = []
    for column in result.columns:
        if any(result[column]):
            filter_cols.append(column)
    return result[filter_cols]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=int)
    parser.add_argument("--window_size", type=int)
    parser.add_argument("--dimensions", type=int)
    parser.add_argument("--lag", type=int)
    parser.add_argument("--test_lag", type=int)
    parser.add_argument(
        "--test_name", choices=["ssr_ftest", "ssr_chi2test", "lrtest", "params_ftest"]
    )
    args = parser.parse_args()

    test_lag = args.test_lag
    sample_size = args.sample_size
    window_size = args.window_size
    dimensions = args.dimensions
    lag = args.lag
    test_name = args.test_name

    topology_features = pd.read_csv(
        f"features/topology_features_s{sample_size}_w{window_size}_d{dimensions}.csv",
        index_col="date",
    )
    network_features = pd.read_csv(
        f"features/network_features_s{sample_size}_w{window_size}_l{lag}.csv",
        index_col="date",
    )

    top_cols = topology_features.columns
    net_cols = network_features.columns

    def extract_test_p_value(results, test_lag, test_name):
        return (
            results.applymap(
                lambda cell: cell[test_lag][0][test_name][1] if not pd.isna(cell) else 1
            )
            < 0.05
        )

    network_predicts_topology = [
        pd.DataFrame(columns=top_cols, index=net_cols) for i in range(test_lag)
    ]
    topology_predicts_network = [
        pd.DataFrame(columns=net_cols, index=top_cols) for i in range(test_lag)
    ]
    for top_col in top_cols:
        if topology_features[top_col].var() < 0.03:
            continue
        for net_col in net_cols:
            if network_features[net_col].var() < 0.03:
                continue
            compare = pd.concat(
                [topology_features[top_col], network_features[net_col]], axis=1
            )
            result = grangercausalitytests(compare, test_lag=test_lag, verbose=False)
            for i in range(test_lag):
                print(f"Index {i}")
                network_predicts_topology[i][top_col][net_col] = result[i + 1][0][
                    test_name
                ][1]
            compare = pd.concat(
                [network_features[net_col], topology_features[top_col]], axis=1
            )
            result = grangercausalitytests(compare, test_lag=test_lag, verbose=False)
            for i in range(test_lag):
                topology_predicts_network[i][net_col][top_col] = result[i + 1][0][
                    test_name
                ][1]

    for ind, df in enumerate(network_predicts_topology):
        df.to_csv(
            f"predictions/network_predicts_topology_l{ind+1}_s{sample_size}_w{window_size}_d{dimensions}.csv"
        )
    for ind, df in enumerate(topology_predicts_network):
        df.to_csv(
            f"predictions/topology_predicts_network_l{ind+1}_s{sample_size}_w{window_size}_d{dimensions}.csv"
        )
