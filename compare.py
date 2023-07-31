import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import networkx as nx
import matplotlib.pyplot as plt
import numpy as n
from statsmodels.tsa.stattools import grangercausalitytests

import argparse


def perform_granger_causality_test(
    df1, df2, maxlag, significance_level=0.05, test_name="ssr_ftest"
):
    combined_df = pd.concat([df1, df2], axis=1)
    causality_graph = nx.DiGraph()

    for col1 in df1.columns:
        if df1[col1].var() < 0.1:
            continue
        for col2 in df2.columns:
            if df2[col2].var() < 0.1:
                continue
            result = grangercausalitytests(
                combined_df[[col1, col2]], maxlag=maxlag, verbose=False
            )
            p_value = result[maxlag][0][test_name][1]
            if p_value < significance_level:
                print("Adding edge")
                causality_graph.add_edge(col1, col2)

    return causality_graph


def draw_causality_graph(causality_graph, df1_columns, df2_columns, output_filename):
    # Set the positions of the nodes explicitly using a circular layout
    pos = nx.circular_layout(causality_graph)

    # Customize node colors and sizes
    node_size = 1000

    # Draw the causality graph with larger arrowheads and different node colors
    nx.draw(
        causality_graph,
        pos,
        with_labels=False,
        node_size=node_size,
        font_size=12,
        font_weight="bold",
        alpha=0.8,
        edge_color="gray",
        arrowsize=20,
        node_color=[
            ("skyblue" if node in df1_columns else "lightcoral")
            for node in causality_graph.nodes()
        ],
    )

    # Set node label positions slightly away from the nodes for better visibility
    labels = nx.draw_networkx_labels(
        causality_graph,
        pos,
        font_size=10,
        font_color="black",
        font_weight="bold",
        verticalalignment="center",
        bbox=dict(boxstyle="round", edgecolor="white", facecolor="white", alpha=0.7),
    )

    # Increase the space between nodes and labels for better readability
    for label in labels:
        labels[label]._y -= 0.05

    plt.savefig(output_filename, format="jpg", dpi=300)

    # Show the plot
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
    parser.add_argument("--forecast", choices=["network", "topology"])
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

    if args.forecast == "network":
        x_df = topology_features
        y_df = network_features
    elif args.forecast == "topology":
        x_df = network_features
        y_df = topology_features
    else:
        raise Exception("Forecast has to be specified!")

    causality_graph = perform_granger_causality_test(
        x_df,
        y_df,
        maxlag=lag,
        significance_level=threshold,
        test_name=test_name,
    )
    print(causality_graph)

    output_filename = f"comparisons/{args.forecast}_t{test_name}_l{test_lag}.jpg"

    # Draw the causality graph
    draw_causality_graph(causality_graph, x_df.columns, y_df.columns, output_filename)
