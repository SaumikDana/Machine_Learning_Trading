import matplotlib.pyplot as plt
import yfinance as yf  
import pandas as pd
import numpy as np
import scipy.stats as si

# Black-Scholes Call Price Calculation
def black_scholes_call(S, K, T, r, sigma):
    # Prevent division by zero or near-zero values
    sigma = max(sigma, 0.0001)
    T = max(T, 1/(365*24*60))  # At least 1 minute to expiration

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    try:
        call_price = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    except Exception as e:
        print(f"Error calculating call price: {e}")
        call_price = np.nan

    return call_price

# Calculate Greeks for Call Option
def call_greeks(S, K, T, r, sigma):
    # Prevent division by zero or near-zero values
    sigma = max(sigma, 0.0001)
    T = max(T, 1/(365*24*60))  # At least 1 minute to expiration

    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
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
    sigma = max(sigma, 0.0001)
    T = max(T, 1/(365*24*60))
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))

# Calculate Greeks for Put Option
def put_greeks(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = -si.norm.cdf(-d1, 0.0, 1.0)
    gamma = si.norm.pdf(d1, 0.0, 1.0) / (S * sigma * np.sqrt(T))
    theta = -((S * si.norm.pdf(d1, 0.0, 1.0) * sigma) / (2 * np.sqrt(T))) + r * K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0)
    vega = S * si.norm.pdf(d1, 0.0, 1.0) * np.sqrt(T)
    rho = -K * T * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0)
    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'rho': rho}

def plot_greeks_and_prices(dates, call_bs_prices, greeks_data):
    plt.figure(figsize=(4, 4))
    plt.plot(dates, call_bs_prices, label='Theoretical Price', color='green')
    
    for greek, values in greeks_data.items():
        plt.plot(dates, values, label=greek.capitalize())
    
    plt.title('Theoretical Prices & Greeks v Time', fontsize='small')
    plt.legend(loc='best', fontsize='x-small', frameon=False)

    plt.grid(True)
    
    # Rotate and set font size for the x and y ticks
    plt.xticks(rotation=45, fontsize='x-small')
    plt.yticks(fontsize='x-small')
    
    plt.show()

# Main Analysis Function
def analyze_and_plot_stock_options(ticker_symbol, risk_free_rate=0.01):
    
    stock = yf.Ticker(ticker_symbol)
    exp_dates = stock.options
    stock_info = stock.info
    S = stock_info.get('currentPrice', stock_info.get('previousClose', None))
    if S is None:
        raise ValueError("No current price data available.")

    call_bs_prices, put_bs_prices, dates = [], [], []
    greeks_data = {'delta': [], 'gamma': [], 'theta': [], 'vega': [], 'rho': []}

    for date in exp_dates:
        options_data = stock.option_chain(date)
        call_options = options_data.calls
        T = (pd.to_datetime(date) - pd.Timestamp.now()).days / 365
        dates.append(pd.to_datetime(date))

        # Select a specific call option for each date, e.g., the option with the closest strike price to the current price
        selected_option = call_options.iloc[(call_options['strike'] - S).abs().argsort()[:1]]
        for _, call_row in selected_option.iterrows():
            K = call_row['strike']
            sigma = call_row['impliedVolatility']
            price = black_scholes_call(S, K, T, risk_free_rate, sigma)
            call_bs_prices.append(price)
            greeks = call_greeks(S, K, T, risk_free_rate, sigma)
            for key in greeks:
                greeks_data[key].append(greeks[key])

    aligned_dates = []
    aligned_call_bs_prices = []

    for date, price in zip(dates, call_bs_prices):
        if not np.isnan(price):
            aligned_dates.append(date)
            aligned_call_bs_prices.append(price)

    # Check if data is aligned
    data_is_aligned = all(len(values) == len(aligned_dates) for values in greeks_data.values())

    # If everything is aligned, proceed to plot
    if data_is_aligned and len(aligned_dates) == len(aligned_call_bs_prices):
        plot_greeks_and_prices(aligned_dates, aligned_call_bs_prices, greeks_data)
    else:
        print("Data is not aligned. Cannot plot.")
