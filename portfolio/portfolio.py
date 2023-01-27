from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from seaborn import displot
import yfinance as yf

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

    def portfolio_expected_shortfall(self, tickers):
        # Get historical data for the stocks
        data = yf.download(tickers, start='2010-01-01', end='2022-12-31')['Adj Close']

        # Calculate daily returns
        returns = data.pct_change()

        # Calculate the portfolio returns
        portfolio_returns = returns.mean(axis=1)

        # Sort the portfolio returns in descending order
        portfolio_returns = -np.sort(-portfolio_returns)

        # Calculate the value at risk (VaR) at the 1% significance level
        VaR = norm.ppf(0.01, portfolio_returns.mean(), portfolio_returns.std())

        # Calculate the expected shortfall (ES)
        ES = np.mean(portfolio_returns[portfolio_returns < VaR])

        # Plot the portfolio returns
        plt.plot(portfolio_returns)
        plt.axhline(y=VaR, color='r', linestyle='-')
        plt.title('Expected Shortfall')
        plt.xlabel('Sorted Returns')
        plt.ylabel('Value')
        plt.show()

        # Return the expected shortfall
        return ES

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
    # print(p.correlation('AAPL', 'TSLA'))
    p.var('TSLA')
