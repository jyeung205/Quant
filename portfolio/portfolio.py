import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, norm
from seaborn import displot
import seaborn as sns
import yfinance as yf
from scipy.optimize import minimize
import pandas as pd

START = '2011-01-01'
END = '2021-12-01'
RFR = 0.04


class Portfolio:

    def __init__(self, tickers, weights=None, start=START, end=END):
        self.cvar = None
        self.var = None
        self.sharpe = None
        self.tickers = tickers
        if len(tickers) != 1:
            data = yf.download(tickers, start=start, end=end)['Adj Close'][tickers]
        else:
            data = yf.download(tickers, start=start, end=end)['Adj Close']
        self.data = data
        self.log_returns = np.log(data) - np.log(data.shift(1))
        if not weights:
            if len(tickers) == 1:
                self.weights = [1.]
            else:
                self.weights = self.markowitz()
        else:
            self.weights = weights
        if len(tickers) != 1:
            self.portfolio_returns = self.log_returns.dot(self.weights)
        else:
            self.portfolio_returns = self.log_returns

    def markowitz(self):
        log_returns = np.log(self.data) - np.log(self.data.shift(1))

        def portfolio_variance(w, r):
            cov = r.cov()
            return w.T @ cov @ w

        def constraint_sum(w):
            return w.sum() - 1

        w0 = np.ones(len(self.tickers)) / len(self.tickers)
        bounds = [(0, 1) for i in range(len(self.tickers))]
        result = minimize(portfolio_variance, w0, args=(log_returns,), bounds=bounds,
                          constraints={"type": "eq", "fun": constraint_sum})
        optimal_weights = result.x
        return optimal_weights

    def correlation(self):
        corr = self.log_returns.corr()
        return corr

    def plot_correlation(self):
        corr = self.correlation()
        plt.figure(figsize=(10, 10))
        sns.heatmap(corr, annot=True, cmap="YlGnBu")
        plt.title('Log Returns Correlation Heatmap')
        plt.show()

    def plot_returns_dist(self):
        self.portfolio_returns.dropna(inplace=True)
        displot(self.portfolio_returns, bins=50)
        plt.show()

    def calc_sharpe(self):
        mean = self.portfolio_returns.mean()
        vol = self.portfolio_returns.std()
        self.sharpe = np.sqrt(252) * (mean - (RFR/360)) / vol
        return self.sharpe

    def calc_sortino(self):
        return

    def calc_var(self, perc=0.05):
        mean = self.portfolio_returns.mean()
        vol = self.portfolio_returns.std()
        self.var = norm.ppf(perc, mean, vol)
        fig, ax = plt.subplots()
        ax.hist(self.portfolio_returns, bins=50)
        ax.axvline(x=self.var, color='r', linestyle='-')
        plt.title(f'{100*perc}% VaR')
        plt.xlabel('Returns')
        plt.ylabel('Dist')
        plt.show()
        return self.var

    def calc_cvar(self, confidence_level=0.05):
        sorted_returns = self.portfolio_returns.sort_values()
        index = int(confidence_level * len(sorted_returns))
        worst_returns = sorted_returns[:index]
        self.cvar = worst_returns.mean()
        return self.cvar

    def plot_efficient_frontier(self):
        returns = self.log_returns.dropna()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()

        num_portfolios = 5000
        results = np.zeros((3, num_portfolios))
        weights_record = []

        for i in range(num_portfolios):
            weights = np.random.random(len(self.tickers))
            weights /= np.sum(weights)
            returns = np.dot(weights, mean_returns)
            volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            results[0, i] = returns
            results[1, i] = volatility
            results[2, i] = results[0, i] / results[1, i]
            weights_record.append(weights)

        results_frame = pd.DataFrame(results.T, columns=['returns', 'volatility', 'sharpe'])

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(results_frame.volatility, results_frame.returns, c=results_frame.sharpe, cmap='YlGnBu',
                   edgecolors='black', s=10, alpha=0.3)
        ax.set_xlabel('Volatility')
        ax.set_ylabel('Returns')
        ax.set_title('Efficient Frontier')
        # plt.colorbar(label='Sharpe Ratio')
        plt.show()


if __name__ == '__main__':
    p = Portfolio(['TSLA', 'AAPL', 'AMZN', 'NFLX', 'MSFT', 'BBY', 'A'])
    p.plot_correlation()
