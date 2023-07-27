import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

maxlag = 10
tests = ["ssr_ftest", "ssr_chi2test", "lrtest", "params_ftest"]

top_features = pd.read_csv("top_features.csv", index_col="date")
graph_features = pd.read_csv("graph_features.csv", index_col="date")

top_columns = top_features.columns
graph_columns = graph_features.columns


graph_affects_top = pd.DataFrame(columns=top_columns, index=graph_columns)
top_affects_graph = pd.DataFrame(columns=graph_columns, index=top_columns)
for top_col in top_columns:
    for graph_col in graph_columns:
        compare = pd.concat([top_features[top_col], graph_features[graph_col]], axis=1)
        graph_affects_top[top_col][graph_col] = grangercausalitytests(
            compare, maxlag=maxlag, verbose=False
        )
        compare = pd.concat([graph_features[graph_col], top_features[top_col]], axis=1)
        top_affects_graph[graph_col][top_col] = grangercausalitytests(
            compare, maxlag=maxlag, verbose=False
        )


def extract_test_p_value(results, maxlag, test_name):
    return results.applymap(lambda cell: cell[maxlag][0][test_name][1]) < 0.05


def present(result):
    filter_cols = []
    for column in result.columns:
        if any(result[column]):
            filter_cols.append(column)
    return result[filter_cols]


for lag in range(1, maxlag + 1):
    for name in tests:
        print(present(extract_test_p_value(graph_affects_top, lag, name)))
