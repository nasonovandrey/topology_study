import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from read_utils import prepare_sample
from statsmodels.tsa.stattools import grangercausalitytests
import networkx as nx
import matplotlib.pyplot as plt


def cointegration_test(ts1, ts2):
    result = adfuller(ts1 - ts2)
    p_value = result[1]
    return p_value


def get_correlated_pairs(df, correlation_threshold):
    correlated_pairs = []
    correlation_matrix = df.corr()
    for i, col1 in enumerate(df.columns):
        for j, col2 in enumerate(df.columns):
            if i < j:  # To avoid duplicate pairs
                correlation_value = correlation_matrix.iloc[i, j]
                if (
                    abs(correlation_value) > correlation_threshold
                ):  # You can adjust the correlation threshold as needed
                    correlated_pairs.append((col1, col2, correlation_value))
    return correlated_pairs


def get_cointegrated_pairs(df, cointegration_threshold, correlated_pairs):
    cointegrated_pairs = []
    for pair in correlated_pairs:
        col1, col2, correlation_value = pair
        p_value = cointegration_test(df[col1], df[col2])
        if p_value < cointegration_threshold:
            cointegrated_pairs.append((col1, col2, correlation_value))
    return cointegrated_pairs


def granger_causality_test(df, cointegrated_pairs, max_lag, test_name):
    results = {}
    for pair in cointegrated_pairs:
        col1, col2, cointegration_value = pair
        data = df[[col1, col2]]
        if df[col1].var() > .1 and df[col2].var() > .1:
            result = grangercausalitytests(data, max_lag, verbose=False)
            results[pair] = [result[i + 1][0][test_name][1] for i in range(max_lag)]
        else:
            results[pair] = [1]
    return results


def build_adjacency_matrix(
    df,
    correlation_threshold=0.95,
    cointegration_threshold=0.95,
    causation_threshold=0.05,
    max_lag=5,
    test_name="ssr_ftest",
):
    correlated_pairs = get_correlated_pairs(df, correlation_threshold)
    cointegrated_pairs = get_cointegrated_pairs(
        df, cointegration_threshold, correlated_pairs
    )
    granger_results = granger_causality_test(df, cointegrated_pairs, max_lag, test_name)
    num_series = len(df.columns)
    adj_matrix = np.zeros((num_series, num_series))
    for pair, p_values in granger_results.items():
        col1, col2, _ = pair
        min_p_value = min(p_values)
        if min_p_value < causation_threshold:
            i = df.columns.get_loc(col1)
            j = df.columns.get_loc(col2)
            adj_matrix[i][j] = 1
    return adj_matrix
