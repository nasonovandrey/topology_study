import numpy as np
import pandas as pd
import networkx as nx
import numpy as np


def build_adjacency_matrix(dataframe, p_threshold=0.95):
    p_value_matrix = dataframe.corr()
    adjacency_matrix = np.zeros_like(p_value_matrix, dtype=int)
    adjacency_matrix[p_value_matrix > p_threshold] = 1
    return adjacency_matrix
