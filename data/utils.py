import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf  
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import pandas as pd

np.random.seed(42)

def format_time(t):
    """Return a formatted time string 'HH:MM:SS
    based on a numeric time() value"""
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return f'{h:0>2.0f}:{m:0>2.0f}:{s:0>2.0f}'

class MultipleTimeSeriesCV:
    """Generates tuples of train_idx, test_idx pairs
    Assumes the MultiIndex contains levels 'symbol' and 'date'
    purges overlapping outcomes"""

    def __init__(
        self,
        n_splits=3,
        train_period_length=126,
        test_period_length=21,
        lookahead=None,
        date_idx='date',
        shuffle=False
    ):
        self.n_splits = n_splits
        self.lookahead = lookahead
        self.test_length = test_period_length
        self.train_length = train_period_length
        self.shuffle = shuffle
        self.date_idx = date_idx

    def split(self, X, y=None, groups=None):
        unique_dates = X.index.get_level_values(self.date_idx).unique()
        days = sorted(unique_dates, reverse=True)
        split_idx = []
        for i in range(self.n_splits):
            test_end_idx = i * self.test_length
            test_start_idx = test_end_idx + self.test_length
            train_end_idx = test_start_idx + self.lookahead - 1
            train_start_idx = train_end_idx + self.train_length + self.lookahead - 1
            split_idx.append([train_start_idx, train_end_idx, test_start_idx, test_end_idx])

        dates = X.reset_index()[[self.date_idx]]
        for train_start, train_end, test_start, test_end in split_idx:
            train_idx = dates[(dates[self.date_idx] > days[train_start])
                              & (dates[self.date_idx] <= days[train_end])].index
            test_idx = dates[(dates[self.date_idx] > days[test_start])
                             & (dates[self.date_idx] <= days[test_end])].index
            if self.shuffle:
                np.random.shuffle(list(train_idx))
            yield train_idx.to_numpy(), test_idx.to_numpy()

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits

import requests
from bs4 import BeautifulSoup
import pytz

# Function to fetch stock price using yfinance
def get_stock_price(ticker):
    stock = yf.Ticker(ticker)
    try:
        stock_info = stock.info
        price = stock_info.get('currentPrice')
        return price
    except ValueError as e:
        print(f"Error retrieving info for {ticker}: {e}")
        return None

def process_earnings_table(ticker_data_list, table):
    """
    Process a single earnings table to extract ticker symbols, 
    fetch stock prices, and get earnings release dates.
    """
    df = pd.read_html(str(table))[0]
    
    if 'Symbol' in df.columns:
        # Assuming the release date column is named 'Release Date'
        for _, row in df.iterrows():
            ticker = row.get('Symbol')
            if pd.notna(ticker):
                price = get_stock_price(ticker)
                ticker_data_list.append(
                    pd.DataFrame({'Symbol': [ticker], 'Stock Price': [price]})
                )

    return ticker_data_list

def scrape_and_process_yahoo_finance_data(url, ticker_data_list=[]):

    # Scrape data from MarketWatch
    headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:87.0) Gecko/20100101 Firefox/87.0'}
    
    response = requests.get(url, headers=headers)

    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all tables in the webpage
    table = soup.find_all('table')

    ticker_data_list = process_earnings_table(ticker_data_list, table)

    # Concatenate all DataFrame objects into one DataFrame
    ticker_data = pd.concat(ticker_data_list, ignore_index=True)

    # Clean and sort the data
    ticker_data = ticker_data.dropna(subset=['Stock Price'])
    ticker_data_sorted = ticker_data.sort_values(by='Stock Price', ascending=False)

    return ticker_data_sorted, ticker_data_list

def analyze_options(ticker, call_options, put_options):
    """
    Analyze and plot options data for a given ticker.
    
    Parameters:
    ticker (str): The stock ticker symbol.
    call_options (DataFrame): DataFrame containing call options data.
    put_options (DataFrame): DataFrame containing put options data.
    """
    # Setting up the figure and axes for a 1 row, 3 column layout
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Set the main title of the figure
    fig.suptitle(f"Options Analysis for {ticker}", fontsize=16)

    # Plot 1: Volume vs Open Interest for Calls
    axs[0].bar(call_options['strike'], call_options['volume'], color='blue', alpha=0.7, label='Volume')
    axs[0].bar(call_options['strike'], call_options['openInterest'], color='green', alpha=0.5, label='Open Interest')
    axs[0].set_title('Call Options Volume vs Open Interest')
    axs[0].set_xlabel('Strike Price')
    axs[0].legend()

    # Plot 2: Volume vs Open Interest for Puts
    axs[1].bar(put_options['strike'], put_options['volume'], color='red', alpha=0.7, label='Volume')
    axs[1].bar(put_options['strike'], put_options['openInterest'], color='orange', alpha=0.5, label='Open Interest')
    axs[1].set_title('Put Options Volume vs Open Interest')
    axs[1].set_xlabel('Strike Price')
    axs[1].legend()

    # Plot 3: Implied Volatility Skew
    axs[2].plot(call_options['strike'], call_options['impliedVolatility'], label='Call IV', color='blue')
    axs[2].plot(put_options['strike'], put_options['impliedVolatility'], label='Put IV', color='red')
    axs[2].set_title('Implied Volatility Skew')
    axs[2].set_xlabel('Strike Price')
    axs[2].set_ylabel('Implied Volatility')
    axs[2].legend()

    plt.tight_layout()
    plt.show()

    # Moneyness of Options
    itm_calls_count = call_options[call_options['inTheMoney']].shape[0]
    otm_calls_count = call_options[~call_options['inTheMoney']].shape[0]
    itm_puts_count = put_options[put_options['inTheMoney']].shape[0]
    otm_puts_count = put_options[~put_options['inTheMoney']].shape[0]

    print(f"ITM Calls: {itm_calls_count}, OTM Calls: {otm_calls_count}")
    print(f"ITM Puts: {itm_puts_count}, OTM Puts: {otm_puts_count}")

    return

def analyze_stock_options(ticker):
    # Fetch the stock data using the provided ticker symbol
    stock = yf.Ticker(ticker)

    # Initialize variables for aggregating options data
    total_call_volume, total_call_open_interest, total_call_implied_volatility = 0, 0, 0
    total_put_volume, total_put_open_interest, total_put_implied_volatility = 0, 0, 0
    total_itm_calls, total_itm_puts = 0, 0  # Counters for in-the-money options
    exp_dates_count = 0  # Counter for the number of expiration dates

    # Get the list of options expiration dates for the stock
    exp_dates = stock.options

    # Loop through each expiration date to analyze options data
    for date in exp_dates:
        # Retrieve call and put options data for the current expiration date
        options_data = stock.option_chain(date)
        call_options, put_options = options_data.calls, options_data.puts

        # Aggregate call options data: sum volumes and open interests, calculate mean implied volatility
        total_call_volume += call_options['volume'].sum()
        total_call_open_interest += call_options['openInterest'].sum()
        total_call_implied_volatility += call_options['impliedVolatility'].mean()

        # Aggregate put options data: sum volumes and open interests, calculate mean implied volatility
        total_put_volume += put_options['volume'].sum()
        total_put_open_interest += put_options['openInterest'].sum()
        total_put_implied_volatility += put_options['impliedVolatility'].mean()

        # Count in-the-money options: calls with strike price below stock price and puts with strike price above
        total_itm_calls += call_options[call_options['inTheMoney']].shape[0]
        total_itm_puts += put_options[put_options['inTheMoney']].shape[0]

        # Increment the expiration dates counter
        exp_dates_count += 1

    # Calculate average implied volatility if there are expiration dates
    if exp_dates_count > 0:
        avg_call_implied_volatility = total_call_implied_volatility / exp_dates_count
        avg_put_implied_volatility = total_put_implied_volatility / exp_dates_count
    else:
        avg_call_implied_volatility = avg_put_implied_volatility = 0

    total_call_engagement = total_call_volume + total_call_open_interest
    total_put_engagement = total_put_volume + total_put_open_interest

    # Return a dictionary with the aggregated and calculated options metrics
    return {
        "total_call_engagement": total_call_engagement,
        "total_put_engagement": total_put_engagement,
        "avg_call_implied_volatility": avg_call_implied_volatility,
        "avg_put_implied_volatility": avg_put_implied_volatility,
        "total_call_volume": total_call_volume,
        "total_call_open_interest": total_call_open_interest,
        "total_put_volume": total_put_volume,
        "total_put_open_interest": total_put_open_interest,
        "total_itm_calls": total_itm_calls,
        "total_itm_puts": total_itm_puts
    }

def calculate_oscillators(ticker, period='6mo'):
    # Fetch historical stock data
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)

    # Calculate RSI
    def calculate_RSI(series, period=14):
        delta = series.diff().dropna()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        RS = gain / loss
        return 100 - (100 / (1 + RS))

    # Calculate MACD
    def calculate_MACD(series, fast_period=12, slow_period=26, signal_period=9):
        exp1 = series.ewm(span=fast_period, adjust=False).mean()
        exp2 = series.ewm(span=slow_period, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal_period, adjust=False).mean()
        return macd, signal_line

    # RSI
    hist['RSI'] = calculate_RSI(hist['Close'])

    # MACD and Signal Line
    macd, signal = calculate_MACD(hist['Close'])
    hist['MACD'] = macd
    hist['Signal_Line'] = signal

    # Drop any NaN values from the dataframe
    hist.dropna(inplace=True)

    return hist[['Close', 'RSI', 'MACD', 'Signal_Line']]

def get_most_recent_monday():
    today = datetime.now()
    # Calculate the number of days to subtract to get the most recent Monday
    # 0 is Monday, 1 is Tuesday, ... 6 is Sunday
    days_to_subtract = (today.weekday() - 0) % 7
    most_recent_monday = today - timedelta(days=days_to_subtract)
    return most_recent_monday

def print_options_data(ticker, options_metrics):
    
    print("===========================================")
    print(f"Options data for {ticker}:")

    print(f"Average IV for Calls: {options_metrics['avg_call_implied_volatility']}")
    print(f"Average IV for Puts: {options_metrics['avg_put_implied_volatility']}")

    print(f"Total Call Volume: {options_metrics['total_call_volume']}")
    print(f"Total Call open interest: {options_metrics['total_call_open_interest']}")
    print(f"Total Call engagement: {options_metrics['total_call_engagement']}")

    print(f"Total Put Volume: {options_metrics['total_put_volume']}")
    print(f"Total Put open interest: {options_metrics['total_put_open_interest']}")
    print(f"Total Put engagement: {options_metrics['total_put_engagement']}")

    print(f"Number of ITM Call Options: {options_metrics['total_itm_calls']}")
    print(f"Number of ITM Put Options: {options_metrics['total_itm_puts']}")
    
    return

import matplotlib.ticker as mticker  # Importing the correct module

def plot_stock_history(ticker, start_date, end_date, release_date=None):
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, end=end_date)

    # Plotting the closing prices
    plt.figure(figsize=(3, 3))
    plt.plot(hist.index, hist['Close'], '-o', markersize=2)
    plt.title(f"Stock Price History of {ticker}", fontsize=12)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.xticks(rotation=45)
    plt.xlabel('Date', fontsize=10)
    plt.ylabel('Closing Price', fontsize=10)
    plt.grid(True)
    plt.tick_params(axis='x', labelsize=6)

    # Set y-axis label format
    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

    if release_date:
        # Adding a vertical line at the release date
        release_date_datetime = pd.to_datetime(release_date)
        plt.axvline(x=release_date_datetime, color='red', linestyle='--', label='Release Date')

    plt.show()
    
def get_stock_data(ticker_symbol):
    # Create a Ticker object
    stock = yf.Ticker(ticker_symbol)

    # 1. Historical Market Data (for the past 5 days)
    hist_data = stock.history(period="5d")
    print(f"Historical Market Data for {ticker_symbol} (Last 5 Days):")
    print(hist_data)

    # 2. Company Information
    info = stock.info
    print(f"\nCompany Information for {ticker_symbol}:")
    for key, value in info.items():
        print(f"{key}: {value}")

    # 3. Financial Data
    # Income Statement
    income_statement = stock.financials
    print(f"\nIncome Statement for {ticker_symbol}:")
    print(income_statement)

    # Balance Sheet
    balance_sheet = stock.balance_sheet
    print(f"\nBalance Sheet for {ticker_symbol}:")
    print(balance_sheet)

    # Cash Flows
    cash_flow = stock.cashflow
    print(f"\nCash Flow for {ticker_symbol}:")
    print(cash_flow)

    # 4. Stock Dividends and Splits
    dividends = stock.dividends
    splits = stock.splits
    print(f"\nDividends for {ticker_symbol}:")
    print(dividends)
    print(f"\nStock Splits for {ticker_symbol}:")
    print(splits)
    
    return

def get_options_data(ticker_symbol):
    # Create a Ticker object
    stock = yf.Ticker(ticker_symbol)

    # 5. Options Data (if available)
    try:
        options_dates = stock.options
        print(f"\nOptions Expiry Dates for {ticker_symbol}:")
        print(options_dates)
        for date in options_dates:
            options_data = stock.option_chain(date)
            print(f"\nOptions Data for {ticker_symbol} on {date}:")
            print("Calls:")
            print(options_data.calls)
            print("Puts:")
            print(options_data.puts)
    except Exception as e:
        print(f"\nOptions Data for {ticker_symbol} is not available or an error occurred:", e)

    return

# VIX
def get_stock_data(ticker, start_date, end_date):
    """ Fetch historical stock data using yf.Ticker """
    stock = yf.Ticker(ticker)
    stock_data = stock.history(start=start_date, end=end_date)
    return stock_data['Close']

def calculate_beta(stock_data, market_data):
    """ Calculate the beta of the stock """
    stock_returns = stock_data.pct_change().dropna()
    market_returns = market_data.pct_change().dropna()

    covariance = stock_returns.cov(market_returns)
    variance = market_returns.var()

    return covariance / variance

def plot_stock_vs_market(stock_data, market_data, ticker):
    """ Plot stock data against market data """
    normalized_stock = stock_data / stock_data.iloc[0]
    normalized_market = market_data / market_data.iloc[0]

    # Plotting the closing prices
    plt.figure(figsize=(3, 3))
    plt.plot(normalized_stock, label=f'{ticker} Stock Price ')
    plt.plot(normalized_market, label='S&P 500')
    plt.title(f'Stock Price vs S&P 500 (Both normalized)', fontsize=8)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))  # Format as 'Month-Day'
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())  # Set major ticks to days
    plt.xticks(rotation=45)  # Rotate for better readability
    plt.xlabel('Date', fontsize=8)
    plt.ylabel('Normalized Price', fontsize=8)
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tick_params(axis='x', labelsize=8)

    plt.show()
    return

def get_stock_beta(stock_ticker, start_date, end_date):
    """ Calculate and print the beta of a given stock """
    market_index_ticker = '^GSPC'  # S&P 500

    stock_data = get_stock_data(stock_ticker, start_date, end_date)
    market_data = get_stock_data(market_index_ticker, start_date, end_date)

    beta_value = calculate_beta(stock_data, market_data)
    print(f"Beta of {stock_ticker}: {beta_value}")

    # plot_stock_vs_market(stock_data, market_data, stock_ticker)

    return beta_value

def analyze_stock_performance_post_earnings(ticker, release_date, start_date, end_date):
    """
    Analyze the stock performance by comparing the closing price on the day of the earnings release
    with the closing price on the following day.

    :param ticker: Stock ticker symbol.
    :param release_day: Number of days from the most recent Monday to the earnings release.
    :param start_date: Start date for the analysis period.
    :param end_date: End date for the analysis period.
    :return: A message indicating whether the stock price increased or decreased after the earnings release.
    """

    # Fetch the stock data
    stock_data = get_stock_data(ticker, start_date, end_date)

    release_date = datetime(release_date.year, release_date.month, release_date.day)
    release_date = pytz.timezone('America/New_York').localize(release_date)
    
    # Check if release date data is available
    if release_date not in stock_data.index:
        return "Data not available for the release date."

    # Get closing price on the release date
    release_day_close = stock_data.loc[stock_data.index == release_date].iloc[0]
    
    # Calculate the post release date
    post_release_date = release_date + timedelta(days=1)

    # Check if post release date data is available
    if post_release_date not in stock_data.index:
        return "Data not available for the post release date."

    # Get closing price on the post release date
    post_release_close = stock_data.loc[stock_data.index == post_release_date].iloc[0]

    # Check if the price increased or decreased
    return "Up" if post_release_close > release_day_close else "Down"

def plot_values_with_directions(values, directions, betas):
    """
    Plots a scatter plot of the given values with colors representing the directions and annotates each point with its corresponding beta value.
    
    :param values: List of numerical values.
    :param directions: Corresponding list of directions ('Up' or 'Down').
    :param betas: List of beta values corresponding to each point.
    """
    if len(values) != len(directions) or len(values) != len(betas):
        raise ValueError("The length of values, directions, and betas lists must be the same.")

    # Convert directions to colors ('Up' as blue and 'Down' as red)
    colors = ['blue' if d == 'Up' else 'red' for d in directions]

    plt.figure(figsize=(10, 6))
    sc = plt.scatter(range(len(values)), values, c=colors)
    
    # Annotating each point with its corresponding beta value
    for i, beta in enumerate(betas):
        plt.annotate(
            f"{beta:.1f}", (i, values[i]), 
            textcoords="offset points", 
            xytext=(0,10), ha='center', fontsize=8
        )

    plt.title("Scatter Plot of Values with Up/Down Directions and Beta Annotations")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)

    # Adding a legend for direction
    plt.scatter([], [], c='blue', label='Up')
    plt.scatter([], [], c='red', label='Down')
    plt.legend()

    plt.show()

def analyze_ticker_options(filtered_tickers, ticker_data_sorted):
    
    for index, row in ticker_data_sorted.iterrows():
        ticker = row['Symbol']
        # Loop only over filtered tickers
        if ticker not in filtered_tickers:
            continue

        stock = yf.Ticker(ticker)
        exp_dates = stock.options

        for date in exp_dates:
            options_data = stock.option_chain(date)
            call_options, put_options = options_data.calls, options_data.puts

            analyze_options(ticker, call_options, put_options)
