import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import networkx as nx
import matplotlib.pyplot as plt


def load_first():
    # Implement the function to load the first dataset and return a pandas DataFrame
    # Replace the following line with your actual data loading process
    df1 = pd.DataFrame(...)  # Replace ... with your data loading code
    return df1


def load_second():
    # Implement the function to load the second dataset and return a pandas DataFrame
    # Replace the following line with your actual data loading process
    df2 = pd.DataFrame(...)  # Replace ... with your data loading code
    return df2


import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

import argparse


def perform_granger_causality_test(
    df1, df2, maxlag, significance_level=0.05, test_name="ssr_ftest"
):
    combined_df = pd.concat([df1, df2], axis=1)
    causality_graph = nx.DiGraph()

    for col1 in df1.columns:
        print("Outer col")
        print(col1)
        print(combined_df[col1])
        if df1[col1].var() < 0.1:
            continue
        for col2 in df2.columns:
            print("Inner col")
            print(col2)
            print(combined_df[col2])
            if df2[col2].var() < 0.3:
                continue
            result = grangercausalitytests(
                combined_df[[col1, col2]], maxlag=maxlag, verbose=False
            )
            p_value = result[maxlag][0][test_name][1]
            if p_value < significance_level:
                print("Adding edge")
                causality_graph.add_edge(col1, col2)

    return causality_graph


def draw_causality_graph(causality_graph):
    pos = nx.spring_layout(causality_graph, seed=42)
    nx.draw(
        causality_graph,
        pos,
        with_labels=True,
        node_size=2000,
        node_color="skyblue",
        font_size=12,
        font_weight="bold",
    )
    plt.show()


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
    parser.add_argument("--threshold", type=float, default=0.05)
    args = parser.parse_args()

    test_lag = args.test_lag
    sample_size = args.sample_size
    window_size = args.window_size
    dimensions = args.dimensions
    lag = args.lag
    threshold = args.threshold
    test_name = args.test_name

    topology_features = pd.read_csv(
        f"features/topology_features_s{sample_size}_w{window_size}_d{dimensions}.csv",
        index_col=[0],
    )

    network_features = pd.read_csv(
        f"features/network_features_s{sample_size}_w{window_size}_l{lag}.csv",
        index_col=[0],
    )

    causality_graph = perform_granger_causality_test(
        topology_features,
        network_features,
        maxlag=lag,
        significance_level=threshold,
        test_name=test_name,
    )
    print(causality_graph)

    # Draw the causality graph
    draw_causality_graph(causality_graph)
