# Importing necessary libraries for data manipulation, mathematical operations and plotting
import matplotlib.pyplot as plt
import yfinance as yf  
import pandas as pd
import numpy as np
import scipy.stats as si

# Black-Scholes Call Price Calculation
def black_scholes_call(S, K, T, r, sigma):
    # Prevent division by zero or near-zero values in sigma and T
    sigma = max(sigma, 0.0001)
    T = max(T, 1/(365*24*60))  # Ensuring at least 1 minute to expiration

    # Calculating d1 and d2 using the Black-Scholes formula components
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Try-except block to handle exceptions during call price calculation
    try:
        call_price = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    except Exception as e:
        print(f"Error calculating call price: {e}")
        call_price = np.nan

    return call_price

# Calculate Greeks for Call Option
def call_greeks(S, K, T, r, sigma):
    # Prevent division by zero or near-zero values in sigma and T
    sigma = max(sigma, 0.0001)
    T = max(T, 1/(365*24*60))  # Ensuring at least 1 minute to expiration

    # Try-except block to handle exceptions during Greeks calculation
    try:
        # Calculating d1 and d2 for Greeks
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Calculating each of the Greeks for Call options
        delta = si.norm.cdf(d1, 0.0, 1.0)
        gamma = si.norm.pdf(d1, 0.0, 1.0) / (S * sigma * np.sqrt(T))
        theta = -((S * si.norm.pdf(d1, 0.0, 1.0) * sigma) / (2 * np.sqrt(T))) - r * K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
        vega = S * si.norm.pdf(d1, 0.0, 1.0) * np.sqrt(T)
        rho = K * T * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
    except Exception as e:
        print(f"Error calculating Greeks: {e}")
        return {'delta': np.nan, 'gamma': np.nan, 'theta': np.nan, 'vega': np.nan, 'rho': np.nan}

    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'rho': rho}

# Black-Scholes Put Price Calculation
def black_scholes_put(S, K, T, r, sigma):
    # Prevent division by zero or near-zero values in sigma and T
    sigma = max(sigma, 0.0001)
    T = max(T, 1/(365*24*60))

    # Calculating d1 and d2 for the put option price
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Calculating the put price using the Black-Scholes formula
    return (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))

# Calculate Greeks for Put Option
def put_greeks(S, K, T, r, sigma):
    # Calculating d1 and d2 for the Greeks
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Calculating each of the Greeks for Put options
    delta = -si.norm.cdf(-d1, 0.0, 1.0)
    gamma = si.norm.pdf(d1, 0.0, 1.0) / (S * sigma * np.sqrt(T))
    theta = -((S * si.norm.pdf(d1, 0.0, 1.0) * sigma) / (2 * np.sqrt(T))) + r * K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0)
    vega = S * si.norm.pdf(d1, 0.0, 1.0) * np.sqrt(T)
    rho = -K * T * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0)
    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'rho': rho}

# Function to plot Greeks and prices for options
def plot_greeks_and_prices(dates, bs_prices, greeks_data, option_type='Option', subplot=None):
    if subplot is not None:
        plt.subplot(subplot)
    else:
        plt.figure(figsize=(10, 4))
    
    # Plotting the theoretical prices
    plt.plot(dates, bs_prices, label='Theoretical Price', color='green')
    
    # Plotting each of the Greeks over time
    for greek, values in greeks_data.items():
        plt.plot(dates, values, label=greek.capitalize())
    
    # Setting the plot title and labels
    plt.title(f'{option_type} Prices & Greeks v Time')
    plt.legend(loc='best',frameon=False, fontsize='small')
    plt.grid(True)
    plt.xticks(rotation=45, fontsize='small')
    plt.yticks(fontsize='small')

# Main function to analyze and plot Greeks for a given stock ticker
def analyze_and_plot_greeks(ticker_symbol, risk_free_rate=0.01):
    stock = yf.Ticker(ticker_symbol)
    exp_dates = stock.options
    stock_info = stock.info
    S = stock_info.get('currentPrice', stock_info.get('previousClose', None))
    if S is None:
        raise ValueError("No current price data available.")

    call_bs_prices, put_bs_prices, dates = [], [], []
    call_greeks_data = {'delta': [], 'gamma': [], 'theta': [], 'vega': [], 'rho': []}
    put_greeks_data = {'delta': [], 'gamma': [], 'theta': [], 'vega': [], 'rho': []}

    for date in exp_dates:
        options_data = stock.option_chain(date)
        call_options = options_data.calls
        put_options = options_data.puts

        T = (pd.to_datetime(date) - pd.Timestamp.now()).days / 365
        dates.append(pd.to_datetime(date))

        selected_call_option = call_options.iloc[(call_options['strike'] - S).abs().argsort()[:1]]
        for _, call_row in selected_call_option.iterrows():
            K = call_row['strike']
            sigma = call_row['impliedVolatility']
            price = black_scholes_call(S, K, T, risk_free_rate, sigma)
            call_bs_prices.append(price)
            call_greeks_result = call_greeks(S, K, T, risk_free_rate, sigma)
            for key in call_greeks_result:
                call_greeks_data[key].append(call_greeks_result[key])

        selected_put_option = put_options.iloc[(put_options['strike'] - S).abs().argsort()[:1]]
        for _, put_row in selected_put_option.iterrows():
            K = put_row['strike']
            sigma = put_row['impliedVolatility']
            put_price = black_scholes_put(S, K, T, risk_free_rate, sigma)
            put_bs_prices.append(put_price)
            put_greeks_result = put_greeks(S, K, T, risk_free_rate, sigma)
            for key in put_greeks_result:
                put_greeks_data[key].append(put_greeks_result[key])

    aligned_call_dates, aligned_call_bs_prices = [], []
    for date, price in zip(dates, call_bs_prices):
        if not np.isnan(price):
            aligned_call_dates.append(date)
            aligned_call_bs_prices.append(price)
    call_data_is_aligned = all(len(values) == len(aligned_call_dates) for values in call_greeks_data.values())

    aligned_put_dates, aligned_put_bs_prices = [], []
    for date, price in zip(dates, put_bs_prices):
        if not np.isnan(price):
            aligned_put_dates.append(date)
            aligned_put_bs_prices.append(price)
    put_data_is_aligned = all(len(values) == len(aligned_put_dates) for values in put_greeks_data.values())

    plt.figure(figsize=(10, 4))
    
    if call_data_is_aligned and len(aligned_call_dates) == len(aligned_call_bs_prices):
        plot_greeks_and_prices(aligned_call_dates, aligned_call_bs_prices, call_greeks_data, 'Call Option', 121)
    else:
        print("Call data is not aligned. Cannot plot.")

    if put_data_is_aligned and len(aligned_put_dates) == len(aligned_put_bs_prices):
        plot_greeks_and_prices(aligned_put_dates, aligned_put_bs_prices, put_greeks_data, 'Put Option', 122)
    else:
        print("Put data is not aligned. Cannot plot.")

    plt.tight_layout()
    plt.show()