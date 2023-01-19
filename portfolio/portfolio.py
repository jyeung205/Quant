from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from seaborn import displot

from data_utils.stock_data import StockData


class Portfolio:

    def __init__(self):
        self.stock_data = StockData()

    def correlation(self, ticker1, ticker2, start='2020-01-01', end='2021-01-01'):
        """
        Correlation of log returns
        """
        start = datetime.strptime(start, '%Y-%m-%d').date()
        end = datetime.strptime(end, '%Y-%m-%d').date()
        df1 = self.stock_data.get_stock_data(ticker1)['adj close'][start:end]
        df2 = self.stock_data.get_stock_data(ticker2)['adj close'][start:end]
        log_returns1 = np.log(df1) - np.log(df1.shift(1))
        log_returns2 = np.log(df2) - np.log(df2.shift(1))
        log_returns1.dropna(inplace=True)
        log_returns2.dropna(inplace=True)
        return pearsonr(log_returns1, log_returns2)[0]

    def plot_correlation(self, tickers, start, end):
        return

    def var(self, ticker, start='2020-01-01', end='2022-12-31'):
        start = datetime.strptime(start, '%Y-%m-%d').date()
        end = datetime.strptime(end, '%Y-%m-%d').date()
        df = self.stock_data.get_stock_data(ticker)['adj close'][start:end]
        log_returns = np.log(df) - np.log(df.shift(1))
        log_returns.dropna(inplace=True)
        displot(log_returns, bins=50)
        plt.show()

    def cvar(self):
        return

    def markowitz(self):
        return


if __name__ == '__main__':
    p = Portfolio()
    # print(p.correlation('AAPL', 'TSLA'))
    p.var('TSLA')
