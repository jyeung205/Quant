from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, norm
from seaborn import displot
import seaborn as sns
import yfinance as yf
from scipy.optimize import minimize

from data_utils.stock_data import StockData

START = '2010-01-01'
END = '2021-12-01'


class Portfolio:

    def __init__(self):
        self.stock_data = StockData()

    def _calc_log_returns_weighted_portfolio(self, tickers, weights):
        df = yf.download(tickers, start=start, end=end)['Adj Close']
        return
    @staticmethod
    def correlation(tickers, start=START, end=END):
        """
        Correlation of log returns
        """
        df = yf.download(tickers, start=start, end=end)['Adj Close']
        log_returns = np.log(df) - np.log(df.shift())
        corr = log_returns.corr()
        return corr

    def plot_correlation(self, tickers, start=START, end=END):
        corr = self.correlation(tickers, start, end)
        plt.figure(figsize=(10, 10))
        sns.heatmap(corr, annot=True)
        plt.title('Log Returns Correlation Heatmap')
        plt.show()
        return corr

    def plot_returns_dist(self, tickers, start=START, end=END):
        start = datetime.strptime(start, '%Y-%m-%d').date()
        end = datetime.strptime(end, '%Y-%m-%d').date()
        df = self.stock_data.get_stock_data(tickers)['adj close'][start:end]  # todo should i just use yfinance?
        log_returns = np.log(df) - np.log(df.shift(1))
        log_returns.dropna(inplace=True)
        displot(log_returns, bins=50)
        plt.show()

    def sharpe(self, tickers, start=START, end=END):
        return

    def sortino(self, tickers, start=START, end=END):
        return

    @staticmethod
    def var(tickers, perc=5, start=END, end=END):
        data = yf.download(tickers, start=start, end=end)['Adj Close']
        log_returns = np.log(data) - np.log(data.shift(1))
        mean = log_returns.mean()
        vol = log_returns.std()
        var = norm.ppf(perc, mean, vol)
        plt.plot(log_returns)
        plt.axhline(y=var, color='r', linestyle='-')
        plt.title(f'{perc}% VaR')
        plt.xlabel('Returns')
        plt.ylabel('Dist')
        plt.show()
        return var

    def cvar(self, tickers):
        return

    def markowitz(self, tickers):
        # Get historical data for the stocks
        data = yf.download(tickers, start='2010-01-01', end='2022-12-31')['Adj Close']

        # Calculate daily returns
        returns = data.pct_change()

        # Define the objective function
        def portfolio_variance(w, r):
            cov = r.cov()
            return w.T @ cov @ w

        # Define the optimization constraints
        def constraint_sum(w):
            return w.sum() - 1

        # Set the initial guess for the weights
        w0 = np.ones(len(tickers)) / len(tickers)

        # Define the optimization bounds
        bounds = [(0, 1) for i in range(len(tickers))]

        # Perform the optimization
        result = minimize(portfolio_variance, w0, args=(returns,), bounds=bounds,
                          constraints={"type": "eq", "fun": constraint_sum})

        # Get the optimal weights
        optimal_weights = result.x

        # Return the optimized portfolio
        return optimal_weights


if __name__ == '__main__':
    p = Portfolio()
    W = p.markowitz(['TSLA', 'AAPL', 'AMZN'])
    print(W)
