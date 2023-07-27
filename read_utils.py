import pandas as pd


def read(num_columns=None, threshold=0.3):
    df = pd.read_csv("stocks/all_stocks_5yr.csv", index_col="date", parse_dates=True)
    # Pivot each series into column
    pivoted = pd.pivot_table(df, values="close", index="date", columns=["Name"])
    # Remove all series with excessive number of Nones
    filtered = pivoted.loc[:, pivoted.isna().mean() < 0.1]
    # Forward and backward fill the rest of Nones
    filled = filtered.fillna(method="ffill")
    filled = filled.fillna(method="bfill")
    # If any Nones are left remove the corresponding columns
    dropped = filled.dropna()
    # Only non constant columns
    result = dropped.loc[:, dropped.std() > threshold]
    if num_columns:
        result = result.iloc[:, :num_columns]
    return result


def window_generator(dataframe, size, shift=1, time_unit="D"):
    start_index = current_index = 0
    end_index = len(dataframe)
    end_window_index = current_index + size
    while current_index <= end_index and end_window_index != end_index:
        # Calculate the end index of the current window
        end_window_index = current_index + size

        # Yield the window, indexed by the starting index of the window
        yield dataframe.iloc[current_index:end_window_index]

        # Move to the next window with the specified shift
        current_index += shift
