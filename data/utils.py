
import numpy as np

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

    def __init__(self,
                 n_splits=3,
                 train_period_length=126,
                 test_period_length=21,
                 lookahead=None,
                 date_idx='date',
                 shuffle=False):
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
            split_idx.append([train_start_idx, train_end_idx,
                              test_start_idx, test_end_idx])

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

import matplotlib.pyplot as plt

def analyze_options(ticker, released, call_options, put_options):
    """
    Analyze and plot options data for a given ticker.
    
    Parameters:
    ticker (str): The stock ticker symbol.
    released (str): 'Yes' if earnings have been released, otherwise 'No'.
    call_options (DataFrame): DataFrame containing call options data.
    put_options (DataFrame): DataFrame containing put options data.
    """
    if released == 'Yes':
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

# utils.py

import yfinance as yf

def analyze_stock_options(ticker):
    stock = yf.Ticker(ticker)

    # Initialize variables for options data
    total_call_volume, total_call_open_interest, total_call_implied_volatility = 0, 0, 0
    total_put_volume, total_put_open_interest, total_put_implied_volatility = 0, 0, 0
    total_itm_calls, total_itm_puts = 0, 0
    exp_dates_count = 0

    # Get options expiration dates
    exp_dates = stock.options

    # Retrieve and analyze options data for each expiration date
    for date in exp_dates:
        options_data = stock.option_chain(date)
        call_options, put_options = options_data.calls, options_data.puts

        # Aggregate call and put metrics
        total_call_volume += call_options['volume'].sum()
        total_call_open_interest += call_options['openInterest'].sum()
        total_call_implied_volatility += call_options['impliedVolatility'].mean()

        total_put_volume += put_options['volume'].sum()
        total_put_open_interest += put_options['openInterest'].sum()
        total_put_implied_volatility += put_options['impliedVolatility'].mean()

        # Count ITM options
        total_itm_calls += call_options[call_options['inTheMoney']].shape[0]
        total_itm_puts += put_options[put_options['inTheMoney']].shape[0]

        exp_dates_count += 1

    # Averaging implied volatility over all expiration dates
    if exp_dates_count > 0:
        avg_call_implied_volatility = total_call_implied_volatility / exp_dates_count
        avg_put_implied_volatility = total_put_implied_volatility / exp_dates_count
    else:
        avg_call_implied_volatility = avg_put_implied_volatility = 0

    # Return a dictionary of calculated metrics
    return {
        "avg_call_implied_volatility": avg_call_implied_volatility,
        "avg_put_implied_volatility": avg_put_implied_volatility,
        "total_call_volume": total_call_volume,
        "total_call_open_interest": total_call_open_interest,
        "total_put_volume": total_put_volume,
        "total_put_open_interest": total_put_open_interest,
        "total_itm_calls": total_itm_calls,
        "total_itm_puts": total_itm_puts
    }

# utils.py

# Other imports and methods...

def print_options_data(ticker, options_metrics):
    calls_metric = options_metrics['total_call_volume'] + options_metrics['total_call_open_interest']
    puts_metric = options_metrics['total_put_volume'] + options_metrics['total_put_open_interest']
    sentiment = "Bullish" if calls_metric > puts_metric else "Bearish"
    print(f"===========================================")
    print(f"Options data for {ticker}:")
    print(f"Market Sentiment for {ticker} is leaning {sentiment}.")
    print(f"Average Implied Volatility for Calls: {options_metrics['avg_call_implied_volatility']}")
    print(f"Average Implied Volatility for Puts: {options_metrics['avg_put_implied_volatility']}")
    print(f"Total Call Volume: {options_metrics['total_call_volume']}")
    print(f"Total Call open interest: {options_metrics['total_call_open_interest']}")
    print(f"Total Put Volume: {options_metrics['total_put_volume']}")
    print(f"Total Put open interest: {options_metrics['total_put_open_interest']}")
    print(f"Number of ITM Call Options: {options_metrics['total_itm_calls']}")
    print(f"Number of ITM Put Options: {options_metrics['total_itm_puts']}")

# utils.py

import matplotlib.dates as mdates

# Other imports and methods...

def plot_stock_history(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, end=end_date)

    # Plotting the closing prices
    plt.figure(figsize=(3, 3))
    plt.plot(hist.index, hist['Close'])
    plt.title(f"Stock Price History of {ticker} Over the Past Week", fontsize=8)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))  # Format as 'Month-Day'
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())  # Set major ticks to days
    plt.xticks(rotation=45)  # Rotate for better readability
    plt.xlabel('Date', fontsize=8)
    plt.ylabel('Closing Price', fontsize=8)
    plt.grid(True)
    plt.show()
