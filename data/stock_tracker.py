import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker  

def stock_tracker(ticker_symbol):
    
    # Define the function to fetch the historical stock prices for the day
    def get_todays_prices(ticker_symbol):
        try:
            ticker = yf.Ticker(ticker_symbol)
            # Use interval='1m' for minute-level data during market hours
            todays_data = ticker.history(period='1d', interval='1m')
            print(f"Data fetched for {ticker_symbol}, entries: {len(todays_data)}")
            return todays_data
        except Exception as e:
            print(f"Error fetching historical prices: {e}")
            return None

    # Fetch historical prices for today
    todays_prices = get_todays_prices(ticker_symbol)

    # Plotting the closing prices
    plt.figure(figsize=(3, 3))
    plt.plot(todays_prices.index, todays_prices['Close'])
    plt.title(f"Todays Stock Price of {ticker_symbol}", fontsize=12)
    plt.xticks(rotation=45)
    plt.xlabel('Time', fontsize=10)
    plt.ylabel('Price', fontsize=10)
    plt.grid(True)
    plt.tick_params(axis='x', labelsize=6)

    # Set y-axis label format
    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
        
    plt.show()
