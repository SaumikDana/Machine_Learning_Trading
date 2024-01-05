import matplotlib.pyplot as plt
import yfinance as yf  
import matplotlib.dates as mdates
import numpy as np
import matplotlib.ticker as mticker
from datetime import datetime, timedelta
from scipy.stats import gaussian_kde

# Specific Stock Analysis

def print_info_keys(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    try:
        stock_info_keys = stock.info.keys()
        print(f"Keys in stock.info for {ticker_symbol}:")
        for key in stock_info_keys:
            print(key)
    except Exception as e:
        print(f"Error retrieving info for {ticker_symbol}: {e}")

# Function to fetch historical financial ratios
def get_financial_ratios(ticker_symbol, start_date, end_date):
    stock = yf.Ticker(ticker_symbol)
    hist = stock.history(start=start_date, end=end_date)

    # Use trailing EPS to calculate P/E ratio if available
    if 'trailingEps' in stock.info:
        trailing_eps = stock.info['trailingEps']
        hist['P/E'] = hist['Close'] / trailing_eps

    # Other calculations could follow a similar pattern if the data were available
    
    return hist

# Function to plot the P/E ratio time series
def plot_pe_ratio(ticker_symbol, date):
    start_date = date - timedelta(days=365)  # Approximately 1 year back
    end_date = date  # Up to the current date

    hist = get_financial_ratios(ticker_symbol, start_date, end_date)

    if 'P/E' in hist.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(hist.index, hist['P/E'], '-o', markersize=2, label='P/E')
        plt.title(f"P/E Ratio History of {ticker_symbol}")
        plt.xlabel('Date')
        plt.ylabel('P/E Ratio')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.show()
    
def stock_tracker(ticker_symbol, subplot_position):
    def get_todays_prices(ticker_symbol):
        try:
            ticker = yf.Ticker(ticker_symbol)
            todays_data = ticker.history(period='1d', interval='1m')
            print(f"Data fetched for {ticker_symbol}, entries: {len(todays_data)}")
            return todays_data
        except Exception as e:
            print(f"Error fetching historical prices: {e}")
            return None

    todays_prices = get_todays_prices(ticker_symbol)
    times = [ts.strftime('%H:%M') for ts in todays_prices.index]
    
    plt.subplot(1, 2, subplot_position)
    plt.plot(times, todays_prices['Close'])
    plt.title(f"Today's Stock Price of {ticker_symbol}", fontsize='small')
    plt.xticks(times[::20], rotation=45)
    plt.yticks(fontsize='small')
    plt.xlabel('Time', fontsize='small')
    plt.ylabel('Price', fontsize='small')
    plt.grid(True)
    plt.tick_params(axis='x', labelsize=6)
    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

def plot_stock_history(ticker_symbol, start_date, end_date):
    # Adjust the overall plot size
    plt.figure(figsize=(10, 4))
    
    # First plot: Today's prices
    stock_tracker(ticker_symbol, 1)

    stock = yf.Ticker(ticker_symbol)
    hist = stock.history(start=start_date, end=end_date)

    plt.subplot(1, 2, 2)
    plt.plot(hist.index, hist['Close'], '-o', markersize=2)
    plt.title(f"Stock Price History of {ticker_symbol}", fontsize='small')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.xticks(rotation=45)
    plt.yticks(fontsize='small')
    plt.xlabel('Date', fontsize='small')
    plt.ylabel('Closing Price', fontsize='small')
    plt.grid(True)
    plt.tick_params(axis='x', labelsize=6)
    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    
    plt.show()

def get_info(ticker, options_metrics, start_date, end_date):
    
    # Print 
    print_options_data(ticker, options_metrics)
    
    # Call the plot_stock_history method
    plot_stock_history(ticker, start_date, end_date)

    # Strike price distribution
    plot_strike_price_distribution(options_metrics, ticker)

    # IV distribution
    plot_iv_strike_price(options_metrics, ticker)

    return

def plot_iv_strike_price(options_data, ticker):
    # Extract call and put strike prices and their implied volatilities
    call_strike_prices = options_data['call_strike_prices']
    call_ivs = options_data['call_ivs']
    put_strike_prices = options_data['put_strike_prices']
    put_ivs = options_data['put_ivs']

    # Pair each strike price with its IV and then sort by strike price
    paired_call_data = sorted(zip(call_strike_prices, call_ivs))
    paired_put_data = sorted(zip(put_strike_prices, put_ivs))

    # Unzip the paired data into two lists for plotting
    sorted_call_strikes, sorted_call_ivs = zip(*paired_call_data)
    sorted_put_strikes, sorted_put_ivs = zip(*paired_put_data)

    # Create subplots
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot implied volatility against sorted strike prices for calls
    ax[0].plot(sorted_call_strikes, sorted_call_ivs, marker='o', linestyle='-', color='blue', alpha=0.7)
    ax[0].set_title(f'Call Options Implied Volatility for {ticker}')
    ax[0].set_xlabel('Strike Price')
    ax[0].set_ylabel('Implied Volatility')

    # Plot implied volatility against sorted strike prices for puts
    ax[1].plot(sorted_put_strikes, sorted_put_ivs, marker='o', linestyle='-', color='red', alpha=0.7)
    ax[1].set_title(f'Put Options Implied Volatility for {ticker}')
    ax[1].set_xlabel('Strike Price')
    ax[1].set_ylabel('Implied Volatility')

    # Display the plots
    plt.tight_layout()
    plt.show()

def plot_strike_price_distribution(options_data, ticker):
    # Extract call and put strike prices from the options data dictionary
    call_strike_prices = options_data['call_strike_prices']
    put_strike_prices = options_data['put_strike_prices']

    # Count the frequency of each strike price for calls and puts
    call_strike_counts = {price: call_strike_prices.count(price) for price in set(call_strike_prices)}
    put_strike_counts = {price: put_strike_prices.count(price) for price in set(put_strike_prices)}

    # Sort the strike prices and corresponding frequencies
    sorted_call_strikes = sorted(call_strike_counts.keys())
    sorted_put_strikes = sorted(put_strike_counts.keys())
    call_frequencies = [call_strike_counts[strike] for strike in sorted_call_strikes]
    put_frequencies = [put_strike_counts[strike] for strike in sorted_put_strikes]

    # Create subplots
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the distribution of strike prices for calls as dots connected by lines
    ax[0].plot(sorted_call_strikes, call_frequencies, marker='o', linestyle='-', color='blue', alpha=0.7)
    ax[0].set_title(f'Call Options Strike Price Frequency for {ticker}')
    ax[0].set_xlabel('Strike Price')
    ax[0].set_ylabel('Frequency')

    # Plot the distribution of strike prices for puts as dots connected by lines
    ax[1].plot(sorted_put_strikes, put_frequencies, marker='o', linestyle='-', color='red', alpha=0.7)
    ax[1].set_title(f'Put Options Strike Price Frequency for {ticker}')
    ax[1].set_xlabel('Strike Price')
    ax[1].set_ylabel('Frequency')

    # Display the plots
    plt.tight_layout()
    plt.show()

def analyze_stock_options(ticker, volume_factor=0.25):
    # Fetch the stock data using the provided ticker symbol
    stock = yf.Ticker(ticker)

    # Initialize variables for aggregating options data
    total_call_volume, total_call_open_interest, total_call_implied_volatility = 0, 0, []
    total_put_volume, total_put_open_interest, total_put_implied_volatility = 0, 0, []
    total_itm_calls, total_itm_puts = 0, 0  # Counters for in-the-money options
    call_strike_prices, put_strike_prices = [], []  # Lists to store strike prices
    call_ivs, put_ivs = [], []  # Lists to store implied volatilities
    exp_dates_count = 0  # Counter for the number of expiration dates

    # Lists to collect all volumes for calculating averages
    all_call_volumes = []
    all_put_volumes = []

    # Get the list of options expiration dates for the stock
    exp_dates = stock.options

    # Loop through each expiration date to analyze options data
    # Average Volume: Calculate the average volume across all strikes 
    # and use a multiple of this average as your threshold. 
    # Strikes with volumes significantly lower than the average could be excluded.
    for date in exp_dates:
        # Retrieve call and put options data for the current expiration date
        options_data = stock.option_chain(date)

        call_options, put_options = options_data.calls, options_data.puts

        # Collect all volumes to calculate the average later
        all_call_volumes.extend(call_options['volume'].dropna().tolist())
        all_put_volumes.extend(put_options['volume'].dropna().tolist())

    # Calculate the average volumes
    average_call_volume = sum(all_call_volumes) / len(all_call_volumes) if all_call_volumes else 0
    average_put_volume = sum(all_put_volumes) / len(all_put_volumes) if all_put_volumes else 0

    # Determine volume thresholds based on the average and volume factor
    call_vol_threshold = average_call_volume * volume_factor
    put_vol_threshold = average_put_volume * volume_factor
 
    # Loop through each expiration date to analyze options data
    for date in exp_dates:
        # Retrieve call and put options data for the current expiration date
        options_data = stock.option_chain(date)

        call_options, put_options = options_data.calls, options_data.puts

        # Filter out options with low trading volume
        call_options = call_options[call_options['volume'] > call_vol_threshold]
        put_options = put_options[put_options['volume'] > put_vol_threshold]

        # Append strike prices to the respective lists
        call_strike_prices.extend(call_options['strike'].tolist())
        put_strike_prices.extend(put_options['strike'].tolist())

        # Append implied volatilities to the respective lists
        call_ivs.extend(call_options['impliedVolatility'].tolist())
        put_ivs.extend(put_options['impliedVolatility'].tolist())

        # Aggregate call options data: sum volumes and open interests
        total_call_volume += call_options['volume'].sum()
        total_call_open_interest += call_options['openInterest'].sum()
        total_call_implied_volatility.extend(call_options['impliedVolatility'].tolist())

        # Aggregate put options data: sum volumes and open interests
        total_put_volume += put_options['volume'].sum()
        total_put_open_interest += put_options['openInterest'].sum()
        total_put_implied_volatility.extend(put_options['impliedVolatility'].tolist())

        # Count in-the-money options: calls with strike price below stock price and puts with strike price above
        total_itm_calls += call_options[call_options['inTheMoney']].shape[0]
        total_itm_puts += put_options[put_options['inTheMoney']].shape[0]

        # Increment the expiration dates counter
        exp_dates_count += 1

    # Average the implied volatilities if there are any entries in the list
    avg_call_implied_volatility = sum(total_call_implied_volatility) / len(total_call_implied_volatility) if total_call_implied_volatility else 0
    avg_put_implied_volatility = sum(total_put_implied_volatility) / len(total_put_implied_volatility) if total_put_implied_volatility else 0

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
        "total_itm_puts": total_itm_puts,
        "call_strike_prices": call_strike_prices,
        "put_strike_prices": put_strike_prices,
        "call_ivs": call_ivs,
        "put_ivs": put_ivs
    }

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
