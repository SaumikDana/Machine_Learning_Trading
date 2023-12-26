import matplotlib.pyplot as plt
import yfinance as yf  
import matplotlib.dates as mdates
import pandas as pd
import matplotlib.ticker as mticker  

# Specific Stock Analysis

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

def plot_stock_history(ticker, start_date, end_date, release_date=None):
    """ 
    Plot stock History
    """
    
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
    
    return