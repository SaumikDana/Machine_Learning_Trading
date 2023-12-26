import matplotlib.pyplot as plt
import yfinance as yf  
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.ticker as mticker  # Importing the correct module


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

def process_earnings_table(table, ticker_data_list=[]):
    """
    Process a single earnings table to extract ticker symbols, 
    fetch stock prices, and get earnings release dates.
    """
    if table == None:
        return ticker_data_list

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

def extract_table(url):
    
    # Scrape data from Yahoo Finance
    
    headers = {'user-agent': 'Mozilla/5.0'}
    try:
        # Send the GET request
        response = requests.get(url, headers=headers)
        # Parse the page content with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all tables in the webpage
        table = soup.find_all('table')
        
        # Check if there are any tables found
        if not table:
            return None
        
        # Further processing or return the tables
        return table

    except requests.RequestException as e:
        return None

def convert_to_dataframe(ticker_data_list, ticker_data_sorted=pd.DataFrame()):

    # Concatenate all DataFrame objects into one DataFrame
    ticker_data = pd.concat(ticker_data_list, ignore_index=True)

    # Clean and sort the data
    ticker_data = ticker_data.dropna(subset=['Stock Price'])
    ticker_data_sorted = ticker_data.sort_values(by='Stock Price', ascending=False)

    return ticker_data_sorted

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

