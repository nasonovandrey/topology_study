import yfinance as yf
import pandas as pd

def get_nasdaq_stocks(start_year, end_year):
    # Fetch the NASDAQ stock list from yfinance
    nasdaq_list = yf.Ticker("^IXIC")  # The NASDAQ Composite Index
    print(nasdaq_list)
    nasdaq_symbols = nasdaq_list.tickers
    print(nasdaq_symbols)

    # Create an empty DataFrame to store the stock prices
    df_prices = pd.DataFrame()

    # Loop through each NASDAQ symbol and retrieve the historical stock prices
    for symbol in nasdaq_symbols:
        try:
            stock_data = yf.download(symbol, start=str(start_year) + '-01-01', end=str(end_year) + '-12-31', progress=False)
            stock_data['Symbol'] = symbol
            df_prices = pd.concat([df_prices, stock_data])
        except:
            print(f"Failed to retrieve data for {symbol}")

    return df_prices

# Define the time range
start_year = 2013
end_year = 2018

# Retrieve the stock prices for NASDAQ-listed stocks between 2013 and 2018
stock_prices = get_nasdaq_stocks(start_year, end_year)

# Save the DataFrame to a CSV file
csv_filename = "nasdaq_stock_prices_2013_2018.csv"
stock_prices.to_csv(csv_filename)

# Display the first few rows of the DataFrame containing the stock prices
print(stock_prices.head())
