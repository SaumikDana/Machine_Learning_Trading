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

    calls_metric = total_call_volume + total_call_open_interest
    puts_metric = total_put_volume + total_put_open_interest
    sentiment = "Bullish" if calls_metric > puts_metric else "Bearish"

    # Return a dictionary with the aggregated and calculated options metrics
    return {
        "sentiment": sentiment,
        "avg_call_implied_volatility": avg_call_implied_volatility,
        "avg_put_implied_volatility": avg_put_implied_volatility,
        "total_call_volume": total_call_volume,
        "total_call_open_interest": total_call_open_interest,
        "total_put_volume": total_put_volume,
        "total_put_open_interest": total_put_open_interest,
        "total_itm_calls": total_itm_calls,
        "total_itm_puts": total_itm_puts
    }

def get_most_recent_monday():
    today = datetime.now()
    # Calculate the number of days to subtract to get the most recent Monday
    # 0 is Monday, 1 is Tuesday, ... 6 is Sunday
    days_to_subtract = (today.weekday() - 0) % 7
    most_recent_monday = today - timedelta(days=days_to_subtract)
    return most_recent_monday

def print_options_data(ticker, options_metrics, release_day):
    # Get the most recent Monday as the base date
    base_date = get_most_recent_monday()

    # Calculate the release date
    release_date = base_date + timedelta(days=release_day)

    print("===========================================")
    print(f"Options data for {ticker}:")
    if release_day != -1:
        print(f"Earnings Released on {release_date.strftime('%b %d, %Y')}")
    print(f"Market Sentiment for {ticker} is leaning {options_metrics['sentiment']}.")
    print(f"Average Implied Volatility for Calls: {options_metrics['avg_call_implied_volatility']}")
    print(f"Average Implied Volatility for Puts: {options_metrics['avg_put_implied_volatility']}")
    print(f"Total Call Volume: {options_metrics['total_call_volume']}")
    print(f"Total Call open interest: {options_metrics['total_call_open_interest']}")
    print(f"Total Put Volume: {options_metrics['total_put_volume']}")
    print(f"Total Put open interest: {options_metrics['total_put_open_interest']}")
    print(f"Number of ITM Call Options: {options_metrics['total_itm_calls']}")
    print(f"Number of ITM Put Options: {options_metrics['total_itm_puts']}")
    
    return

def plot_stock_history(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, end=end_date)

    # Plotting the closing prices
    plt.figure(figsize=(3, 3))
    plt.plot(hist.index, hist['Close'], '-o', markersize=2)
    plt.title(f"Stock Price History of {ticker} Over the Past Week", fontsize=8)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))  # Format as 'Month-Day'
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())  # Set major ticks to days
    plt.xticks(rotation=45)  # Rotate for better readability
    plt.xlabel('Date', fontsize=8)
    plt.ylabel('Closing Price', fontsize=8)
    plt.grid(True)
    plt.tick_params(axis='x', labelsize=8)

    plt.show()

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

def process_earnings_table(ticker_data_list, table, threshold, index):
    """
    Process a single earnings table to extract ticker symbols, 
    determine release status, fetch stock prices, and get earnings release dates.
    """
    df = pd.read_html(str(table))[0]
    release_status = 'Yes' if index < threshold else 'No'

    if 'Symbol' in df.columns:
        # Assuming the release date column is named 'Release Date'
        for _, row in df.iterrows():
            ticker = row.get('Symbol')
            if pd.notna(ticker):
                price = get_stock_price(ticker)
                ticker_data_list.append(pd.DataFrame({
                    'Symbol': [ticker],
                    'Stock Price': [price],
                    'Released': [release_status],
                    'Release Day': [index]
                }))

    return ticker_data_list

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