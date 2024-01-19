import matplotlib.pyplot as plt
import yfinance as yf  
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import griddata
import pandas as pd

def plot_volatility_surface(options_data, ticker):
    try:
        # Extract data
        stock = yf.Ticker(ticker)
        current_price = stock.info.get('currentPrice', stock.info.get('previousClose', None))
        call_strike_prices = list(np.array(options_data['call_strike_prices'])/current_price)
        put_strike_prices = list(np.array(options_data['put_strike_prices'])/current_price)
        call_ivs = options_data['call_ivs']
        put_ivs = options_data['put_ivs']
        call_expirations = options_data['call_expirations']
        put_expirations = options_data['put_expirations']

        # Convert expiration dates to numerical format
        call_exp_nums = mdates.date2num(pd.to_datetime(call_expirations))
        put_exp_nums = mdates.date2num(pd.to_datetime(put_expirations))

        # Combine call and put data
        strikes = np.array(call_strike_prices + put_strike_prices)
        expirations = np.array(list(call_exp_nums) + list(put_exp_nums))
        ivs = np.array(call_ivs + put_ivs)

        # Check for sufficient variation in data
        if len(set(strikes)) < 2 or len(set(expirations)) < 2:
            print("Insufficient variation in strike prices or expiration dates for interpolation.")
            return

        # Check for NaNs or infinite values
        if np.isnan(ivs).any() or np.isinf(ivs).any():
            print("Data contains NaNs or infinite values.")
            return

        # Create a 2D grid of strikes and expirations
        strike_grid, exp_grid = np.meshgrid(
            np.linspace(strikes.min(), strikes.max(), 100),
            np.linspace(expirations.min(), expirations.max(), 100)
        )

        # Interpolate IV data over the grid
        ivs_grid = griddata((strikes, expirations), ivs, (strike_grid, exp_grid), method='cubic')

        # Plot the surface
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(strike_grid, exp_grid, ivs_grid, cmap='viridis', edgecolor='none')

        # Labels and title
        ax.set_title(f'Volatility Surface for {ticker}')
        ax.set_xlabel('Moneyness (Strike Price/Current Price)')
        ax.set_ylabel('Expiration Date')
        ax.set_zlabel('Implied Volatility')

        # Date formatting
        ax.yaxis.set_major_locator(mdates.AutoDateLocator())
        ax.yaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
        ax.yaxis.set_tick_params(labelsize=9)
        for label in ax.yaxis.get_majorticklabels():
            label.set_rotation(45)

        # View adjustment
        ax.view_init(30, 210)
        colorbar = fig.colorbar(surf, shrink=0.5, aspect=30, pad=-0.025)
        plt.subplots_adjust(left=0.1, right=1.0, top=1.0, bottom=0.05)

        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

def plot_iv_skew_otm_only(options_data, target_date, ticker, days_range=21):
    # Fetch the current stock price
    stock = yf.Ticker(ticker)
    current_price = stock.info.get('currentPrice', stock.info.get('previousClose', None))

    historical_volatility = calculate_historical_volatility(ticker)

    # Extract call and put strike prices and IVs
    call_strike_prices = options_data['call_strike_prices']
    put_strike_prices = options_data['put_strike_prices']
    call_ivs = options_data['call_ivs']
    put_ivs = options_data['put_ivs']
    call_expirations = options_data['call_expirations']
    put_expirations = options_data['put_expirations']

    # Convert expiration dates to datetime objects and filter by target date
    call_expirations_dt = [datetime.strptime(date, "%Y-%m-%d") for date in call_expirations]
    put_expirations_dt = [datetime.strptime(date, "%Y-%m-%d") for date in put_expirations]
    target_date_dt = np.datetime64(target_date)

    # Filter call data for OTM options
    filtered_call_data_otm = [(strike, iv, exp) for strike, iv, exp in zip(call_strike_prices, call_ivs, call_expirations_dt)
                              if exp > target_date_dt and exp <= target_date_dt + np.timedelta64(days_range, 'D') and strike > current_price]
    call_strikes_rel_otm, call_ivs_rel_otm = zip(*[(strike/current_price, iv) for strike, iv, _ in filtered_call_data_otm]) if filtered_call_data_otm else ([], [])

    # Filter put data for OTM options
    filtered_put_data_otm = [(strike, iv, exp) for strike, iv, exp in zip(put_strike_prices, put_ivs, put_expirations_dt)
                             if exp > target_date_dt and exp <= target_date_dt + np.timedelta64(days_range, 'D') and strike < current_price]
    put_strikes_rel_otm, put_ivs_rel_otm = zip(*[(strike/current_price, iv) for strike, iv, _ in filtered_put_data_otm]) if filtered_put_data_otm else ([], [])

    # Create plot for combined OTM skew
    fig, ax = plt.subplots(figsize=(15, 6))

    # Plot OTM call and put data
    ax.scatter(call_strikes_rel_otm, call_ivs_rel_otm, color='blue', marker='o', label='OTM Calls')
    ax.scatter(put_strikes_rel_otm, put_ivs_rel_otm, color='green', marker='o', label='OTM Puts')
    ax.axvline(1, color='grey', linestyle='--', label='Current Price')

    # Overlay historical volatility as a horizontal line
    ax.axhline(y=historical_volatility, color='purple', linestyle='--', label=f'Historical Volatility ({historical_volatility:.2%})')

    ax.set_title(f'OTM Options Implied Volatility Skew - {ticker}')
    ax.set_xlabel('Strike Price / Current Price')
    ax.set_ylabel('Implied Volatility')
    ax.legend(frameon=False)
    ax.grid(True)

    plt.tight_layout()
    plt.show()

def calculate_historical_volatility(ticker_symbol, period="1y"):
    # Fetch historical stock data
    stock = yf.Ticker(ticker_symbol)
    hist = stock.history(period=period)

    # Calculate daily returns
    daily_returns = hist['Close'].pct_change().dropna()

    # Calculate standard deviation of daily returns (historical volatility)
    historical_volatility = np.std(daily_returns)
    
    return historical_volatility

def analyze_stock_options(ticker, price_range_factor=0.25):
    # Fetch the stock data using the provided ticker symbol
    stock = yf.Ticker(ticker)

    # Get current stock price
    current_price = stock.info.get('currentPrice', stock.info.get('previousClose', None))

    # Calculate bounds for strike price filtering based on current price
    lower_bound = current_price * (1 - price_range_factor)
    upper_bound = current_price * (1 + price_range_factor)

    # Initialize variables for aggregating options data
    total_call_volume, total_call_open_interest, total_call_implied_volatility = 0, 0, []
    total_put_volume, total_put_open_interest, total_put_implied_volatility = 0, 0, []
    total_itm_calls, total_itm_puts = 0, 0  # Counters for in-the-money options
    total_otm_calls, total_otm_puts = 0, 0  # Counters for out-of-the-money options
    call_strike_prices, put_strike_prices, call_expirations, put_expirations = [], [], [], []  # Lists to store data
    call_ivs, put_ivs = [], []  # Lists to store implied volatilities
    exp_dates_count = 0  # Counter for the number of expiration dates

    # Get the list of options expiration dates for the stock
    exp_dates = stock.options

    # Loop through each expiration date to analyze options data
    for date in exp_dates:
        # Retrieve call and put options data for the current expiration date
        options_data = stock.option_chain(date)

        call_options, put_options = options_data.calls, options_data.puts

        # Filter options with strike prices within the defined range
        filtered_call_options = call_options[(call_options['strike'] >= lower_bound) & (call_options['strike'] <= upper_bound)]
        filtered_put_options = put_options[(put_options['strike'] >= lower_bound) & (put_options['strike'] <= upper_bound)]

        # Append strike prices, implied volatilities, and expiration dates to the respective lists
        call_strike_prices.extend(filtered_call_options['strike'].tolist())
        put_strike_prices.extend(filtered_put_options['strike'].tolist())
        call_ivs.extend(filtered_call_options['impliedVolatility'].tolist())
        put_ivs.extend(filtered_put_options['impliedVolatility'].tolist())
        call_expirations.extend([date] * len(filtered_call_options))
        put_expirations.extend([date] * len(filtered_put_options))

        # Aggregate call and put options data
        total_call_volume += filtered_call_options['volume'].sum()
        total_call_open_interest += filtered_call_options['openInterest'].sum()
        total_call_implied_volatility.extend(filtered_call_options['impliedVolatility'].tolist())

        total_put_volume += filtered_put_options['volume'].sum()
        total_put_open_interest += filtered_put_options['openInterest'].sum()
        total_put_implied_volatility.extend(filtered_put_options['impliedVolatility'].tolist())

        # Count in-the-money and out-of-the-money options based on the current price
        total_itm_calls += len(filtered_call_options[filtered_call_options['strike'] < current_price])
        total_otm_calls += len(filtered_call_options[filtered_call_options['strike'] > current_price])

        total_itm_puts += len(filtered_put_options[filtered_put_options['strike'] > current_price])
        total_otm_puts += len(filtered_put_options[filtered_put_options['strike'] < current_price])

        # Increment the expiration dates counter
        exp_dates_count += 1

    # Average the implied volatilities if there are any entries in the list
    avg_call_implied_volatility = sum(total_call_implied_volatility) / len(total_call_implied_volatility) if total_call_implied_volatility else 0
    avg_put_implied_volatility = sum(total_put_implied_volatility) / len(total_put_implied_volatility) if total_put_implied_volatility else 0

    # Calculate total engagement for calls and puts
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
        "total_otm_calls": total_otm_calls,
        "total_otm_puts": total_otm_puts,
        "call_strike_prices": call_strike_prices,
        "put_strike_prices": put_strike_prices,
        "call_ivs": call_ivs,
        "put_ivs": put_ivs,
        "call_expirations": call_expirations,
        "put_expirations": put_expirations
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

    print(f"Number of OTM Call Options: {options_metrics['total_otm_calls']}")
    print(f"Number of OTM Put Options: {options_metrics['total_otm_puts']}")

    print("===========================================")

    return

