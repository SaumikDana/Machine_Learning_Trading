import matplotlib.pyplot as plt
import yfinance as yf  
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from datetime import datetime, timedelta
from analyze_options import *


def print_info_keys(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    try:
        stock_info = stock.info  # Fetch stock information
        print(f"Information for {ticker_symbol}:")
        for key, value in stock_info.items():
            print(f"{key}: {value}")
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

def plot_etf_historical_data(ticker_symbol, start_date, end_date):

    # Fetch stock information
    stock = yf.Ticker(ticker_symbol)

    # Get the industry of the stock
    industry = stock.info.get("industry", None)

    # Extract etf ticker symbol corresponding to industry
    etf_ticker_symbol = get_sector_etf_for_stock().get(industry)
    
    # Check if ticker_symbol is not None
    if etf_ticker_symbol is None:
        return

    # Plot historical data
    plot_historical_data(etf_ticker_symbol, industry, start_date, end_date)

    return

def plot_stock_historical_data(ticker_symbol, start_date, end_date):

    # Fetch stock information
    stock = yf.Ticker(ticker_symbol)

    # Get the industry of the stock
    industry = stock.info.get("industry", None)

    # Plot historical data
    plot_historical_data(ticker_symbol, industry, start_date, end_date)

    return

def plot_historical_data(ticker_symbol, industry, start_date, end_date, long='False'):

    stock = yf.Ticker(ticker_symbol)
    hist = stock.history(start=start_date, end=end_date)

    # Determine which data to plot: Close or regularMarketPreviousClose
    if 'Close' in hist.columns:
        prices = hist['Close']
    elif hasattr(stock.info, 'regularMarketPreviousClose'):
        prices = [stock.info['regularMarketPreviousClose']] * len(hist)
    else:
        raise ValueError("No suitable price data found for this stock.")

    plt.plot(hist.index, prices, '-o', markersize=2)
    plt.title(f"Stock Price History of {ticker_symbol} ({industry})", fontsize='small')
    plt.yticks(fontsize='small')
    plt.xlabel('Date', fontsize='small')
    plt.ylabel('Price', fontsize='small')
    plt.grid(True)

    if not long:
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.xticks(rotation=45)
        plt.tick_params(axis='x', labelsize=6)

    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    plt.show()

def plot_stock_history(ticker_symbol, start_date, end_date):
    plt.figure(figsize=(10, 4))
    
    # Plotting today's prices - Assuming stock_tracker is a defined function
    stock_tracker(ticker_symbol, 1)

    # Plotting stock history
    plt.subplot(1, 2, 2)
    plot_stock_historical_data(ticker_symbol, start_date, end_date)
    
def get_info(ticker, options_metrics, start_date, end_date):
    
    # Print 
    print_options_data(ticker, options_metrics)
    
    # Call the plot_stock_history method
    plot_stock_history(ticker, start_date, end_date)

    # Volatility surface
    plot_volatility_surface(options_metrics, ticker)

    return

def get_sector_etf_for_stock():
    # Dictionary mapping industries to their corresponding ETFs
    industry_etf_dict = {
        "Residential Construction": "XHB",
        "Specialty Chemicals": "XLB",
        "Credit Services": "XLF",
        "Financial Data & Stock Exchanges": "IYG",
        "Electrical Equipment & Parts": "XLI",
        "Computer Hardware": "XLK",
        "Farm & Heavy Construction Machinery": "XLI",
        "Insurance Brokers": "IAK",
        "Software - Infrastructure": "IGV",
        "Steel": "SLX",
        "Semiconductors": "SMH",
        "Semiconductor Equipment & Materials": "SMH",
        "Aerospace & Defense": "ITA",
        "REIT - Office": "IYR",
        "Capital Markets": "IAI",
        "Furnishings, Fixtures & Appliances": "XLY",
        "Banks - Regional": "KRE",
        "Industrial Distribution": "FXR",
        "Specialty Industrial Machinery": "XLI",
        "Medical Instruments & Supplies": "IHI",
        "Railroads": "IYT",
        "Medical Devices": "IHI",
        "REIT - Residential": "REZ",
        "Conglomerates": None,  # No specific ETF
        "Electronic Components": "SOXX",
        "Packaged Foods": "XLP",
        "REIT - Specialty": "XLRE",
        "Insurance - Life": "IAK",
        "Software - Application": "IGV",
        "Asset Management": "XLF",
        "Communication Equipment": "XLK",
        "Internet Content & Information": "XLC",
        "Oil & Gas Drilling": "OIH",
        "Electronics & Computer Distribution": "XLK",
        "Thermal Coal": None,  # No specific ETF
        "Information Technology Services": "XLK",
        "Airlines": "JETS",
        "REIT - Mortgage": "REM",
        "Packaging & Containers": "XLB",
        "Auto Parts": "CARZ",
        "Food Distribution": "XLP",
        "Diagnostics & Research": "IHF",
        "Pharmaceutical Retailers": "XLP",
        "Telecom Services": "XLC",
        "Biotechnology": "IBB",
        "Drug Manufacturers - Specialty & Generic": "XPH",
        "Pollution & Treatment Controls": "XLI",  # Part of a broader category
        "Tobacco": "XLP",
        "Restaurants": "PBJ"
    }

    return industry_etf_dict
