import numpy as np
import pandas as pd
from build_utils import build_top_features
from read_utils import read, window_generator

data = read()
wingen = window_generator(data, size=60)
sample = [window for window in wingen]

top_features = build_top_features(sample, data.index[: len(sample)], 5)

top_features.to_csv("top_features.csv")
