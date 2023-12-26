import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf  
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pytz


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

def get_most_recent_monday():
    today = datetime.now()
    # Calculate the number of days to subtract to get the most recent Monday
    # 0 is Monday, 1 is Tuesday, ... 6 is Sunday
    days_to_subtract = (today.weekday() - 0) % 7
    most_recent_monday = today - timedelta(days=days_to_subtract)
    return most_recent_monday

def calculate_and_plot_oscillators(ticker, start_date, end_date):
    # Fetch historical data from Yahoo Finance
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    
    # Check if 'Adj Close' column exists, if not use 'Close'
    close_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'

    if close_col not in data.columns:
        raise ValueError(f"Expected column '{close_col}' not found in the data.")
    
    # Calculate RSI
    delta = data[close_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate Stochastic Oscillator
    low_min = data['Low'].rolling(window=14).min()
    high_max = data['High'].rolling(window=14).max()
    data['%K'] = (data[close_col] - low_min) * 100 / (high_max - low_min)
    data['%D'] = data['%K'].rolling(window=3).mean()
    
    # Calculate MACD
    exp1 = data[close_col].ewm(span=12, adjust=False).mean()
    exp2 = data[close_col].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # Calculate CCI
    TP = (data['High'] + data['Low'] + data['Close']) / 3
    data['CCI'] = (TP - TP.rolling(window=20).mean()) / (0.015 * TP.rolling(window=20).std())
    
    # Create subplots
    fig, axs = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
    
    # Plot Adjusted Close Price
    axs[0].plot(data.index, data[close_col])
    axs[0].set_title(f'{ticker} Adjusted Close Price')
    
    # Plot RSI
    axs[1].plot(data.index, data['RSI'], color='purple')
    axs[1].axhline(70, color='red', linestyle='--')
    axs[1].axhline(30, color='green', linestyle='--')
    axs[1].set_title('Relative Strength Index (RSI)')
    
    # Plot Stochastic Oscillator
    axs[2].plot(data.index, data['%K'], label='%K', color='blue')
    axs[2].plot(data.index, data['%D'], label='%D', color='orange')
    axs[2].axhline(80, color='red', linestyle='--')
    axs[2].axhline(20, color='green', linestyle='--')
    axs[2].set_title('Stochastic Oscillator')
    axs[2].legend()
    
    # Plot MACD
    axs[3].plot(data.index, data['MACD'], label='MACD', color='blue')
    axs[3].plot(data.index, data['Signal line'], label='Signal line', color='orange')
    axs[3].set_title('Moving Average Convergence Divergence (MACD)')
    axs[3].legend()
    
    # Plot CCI
    axs[4].plot(data.index, data['CCI'], color='black')
    axs[4].axhline(100, color='red', linestyle='--')
    axs[4].axhline(-100, color='green', linestyle='--')
    axs[4].set_title('Commodity Channel Index (CCI)')

    plt.tight_layout()
    plt.show()

    return

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

    return

