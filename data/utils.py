
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
