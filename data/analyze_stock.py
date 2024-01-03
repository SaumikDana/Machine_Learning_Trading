import matplotlib.pyplot as plt
import yfinance as yf  
import matplotlib.dates as mdates
import pandas as pd
import matplotlib.ticker as mticker
from datetime import datetime, timedelta

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
