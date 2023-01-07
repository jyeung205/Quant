from datetime import date, datetime
import datetime as dt
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf
from scipy.optimize import fsolve

from data.options_data import options_chain_by_expiry, options_chain_by_strike, get_options_grid


class BlackScholes:

    def __init__(self, s=None, k=None, t=None, r=None,
                 sigma=None):
        self.s = s
        self.k = k
        self.t = t
        self.r = r
        self.sigma = sigma
        if type(t) == str:
            delta = datetime.strptime(t, '%Y-%m-%d').date() - date.today()
            self.t = int(delta.days + 1) / 365

        if type(s) == str:
            self.s = yf.download(tickers=s,
                                 period='1d',
                                 interval='1m')['Close'][-1]

    def _config_inputs(self, s, k, t, r, sigma):
        if s is None:
            s = self.s
        if k is None:
            k = self.k
        if t is None:
            t = self.t
        if r is None:
            r = self.r
        if sigma is None:
            sigma = self.sigma

        if type(t) == str:
            delta = datetime.strptime(t, '%Y-%m-%d').date() - date.today()
            t = int(delta.days + 1) / 365

        if type(s) == str:
            s = yf.download(tickers=s,
                            period='1d',
                            interval='1m')['Close'][-1]

        return s, k, t, r, sigma

    def _calc_d1(self, s=None, k=None, t=None, r=None, sigma=None):
        s, k, t, r, sigma = self._config_inputs(s, k, t, r, sigma)
        return (np.log(s / k) + (r + sigma ** 2 / 2) * t) / (sigma * np.sqrt(t))

    def _calc_d2(self, s=None, k=None, t=None, r=None, sigma=None):
        s, k, t, r, sigma = self._config_inputs(s, k, t, r, sigma)
        return self._calc_d1(s, k, t, r, sigma) - sigma * np.sqrt(t)

    def call(self, s=None, k=None, t=None, r=None, sigma=None):
        s, k, t, r, sigma = self._config_inputs(s, k, t, r, sigma)
        d1 = self._calc_d1(s, k, t, r, sigma)
        d2 = self._calc_d2(s, k, t, r, sigma)
        return s * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)

    def put(self, s=None, k=None, t=None, r=None, sigma=None):
        s, k, t, r, sigma = self._config_inputs(s, k, t, r, sigma)
        d1 = self._calc_d1(s, k, t, r, sigma)
        d2 = self._calc_d2(s, k, t, r, sigma)
        return k * np.exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1)

    def iv_call(self, s=None, k=None, t=None, r=None, sigma0=0.30, option_price=10):
        s, k, t, r, _ = self._config_inputs(s, k, t, r, None)

        def f(sigma):
            return self.call(s, k, t, r, sigma) - option_price

        sigma = fsolve(f, sigma0)[0]
        return sigma

    def iv_put(self, s=None, k=None, t=None, r=None, sigma0=0.30, option_price=10):
        s, k, t, r, _ = self._config_inputs(s, k, t, r, None)

        def f(sigma):
            return self.put(s, k, t, r, sigma) - option_price

        sigma = fsolve(f, sigma0)[0]
        return sigma

    def forward(self):
        return self.s - self.k * np.exp(-self.r * self.t)

    def delta_call(self, s=None, k=None, t=None, r=None, sigma=None):
        s, k, t, r, sigma = self._config_inputs(s, k, t, r, sigma)
        d1 = self._calc_d1(s, k, t, r, sigma)
        return norm.cdf(d1)

    def delta_put(self, s=None, k=None, t=None, r=None, sigma=None):
        s, k, t, r, sigma = self._config_inputs(s, k, t, r, sigma)
        d2 = self._calc_d2(s, k, t, r, sigma)
        return norm.cdf(d2)

    def gamma(self, s=None, k=None, t=None, r=None, sigma=None):
        s, k, t, r, sigma = self._config_inputs(s, k, t, r, sigma)
        d1 = self._calc_d1(s, k, t, r, sigma)
        return (norm.pdf(d1)) / (s * sigma * np.sqrt(t))

    def vega(self, s=None, k=None, t=None, r=None, sigma=None):
        s, k, t, r, sigma = self._config_inputs(s, k, t, r, sigma)
        d1 = self._calc_d1(s, k, t, r, sigma)
        vega = s * np.sqrt(t) * norm.pdf(d1)
        return vega

    def theta_call(self, s=None, k=None, t=None, r=None, sigma=None):
        s, k, t, r, sigma = self._config_inputs(s, k, t, r, sigma)
        d1 = self._calc_d1(s, k, t, r, sigma)
        d2 = self._calc_d2(s, k, t, r, sigma)
        p1 = - s * (norm.pdf(d1)) * sigma / (2 * np.sqrt(t))
        p2 = r * k * np.exp(-r * t) * norm.cdf(d2)
        return p1 - p2

    def theta_put(self, s=None, k=None, t=None, r=None, sigma=None):
        s, k, t, r, sigma = self._config_inputs(s, k, t, r, sigma)
        d1 = self._calc_d1(s, k, t, r, sigma)
        d2 = self._calc_d2(s, k, t, r, sigma)
        p1 = - s * (norm.pdf(d1)) * sigma / (2 * np.sqrt(t))
        p2 = r * k * np.exp(-r * t) * norm.cdf(-d2)
        return p1 + p2

    def rho_call(self, s=None, k=None, t=None, r=None, sigma=None):
        s, k, t, r, sigma = self._config_inputs(s, k, t, r, sigma)
        d2 = self._calc_d2(s, k, t, r, sigma)
        return k * t * np.exp(-r * t) * norm.cdf(d2)

    def rho_put(self, s=None, k=None, t=None, r=None, sigma=None):
        s, k, t, r, sigma = self._config_inputs(s, k, t, r, sigma)
        d2 = self._calc_d2(s, k, t, r, sigma)
        return -k * t * np.exp(-r * t) * norm.cdf(-d2)

    def plot_delta_call(self, s=None, k=None, t=None, r=None, sigma=None):
        s, k, t, r, sigma = self._config_inputs(s, k, t, r, sigma)
        spots = [i for i in range(int(s * 0.01), int(s * 2), 1)]
        deltas = [self.delta_call(s, k, t, r, sigma) for s in spots]
        plt.figure()
        plt.title('Delta - Call')
        plt.plot(spots, deltas)
        plt.show()

    def plot_delta_put(self, s=None, k=None, t=None, r=None, sigma=None):
        s, k, t, r, sigma = self._config_inputs(s, k, t, r, sigma)
        spots = [i for i in range(int(s * 0.01), int(s * 2), 1)]
        deltas = [self.delta_put(s, k, t, r, sigma) for s in spots]
        plt.figure()
        plt.title('Delta - Put')
        plt.plot(spots, deltas)
        plt.show()

    def plot_gamma(self, s=None, k=None, t=None, r=None, sigma=None):
        s, k, t, r, sigma = self._config_inputs(s, k, t, r, sigma)
        spots = [i for i in range(int(s * 0.01), int(s * 2), 1)]
        deltas = [self.gamma(s, k, t, r, sigma) for s in spots]
        plt.figure()
        plt.title('Gamma')
        plt.plot(spots, deltas)
        plt.show()

    def plot_vega(self, s=None, k=None, t=None, r=None, sigma=None):
        s, k, t, r, sigma = self._config_inputs(s, k, t, r, sigma)
        spots = [i for i in range(int(s * 0.01), int(s * 2), 1)]
        deltas = [self.vega(s, k, t, r, sigma) for s in spots]
        plt.figure()
        plt.title('Vega')
        plt.plot(spots, deltas)
        plt.show()

    def plot_theta_call(self, s=None, k=None, t=None, r=None, sigma=None):
        s, k, t, r, sigma = self._config_inputs(s, k, t, r, sigma)
        spots = [i for i in range(int(s * 0.01), int(s * 2), 1)]
        deltas = [self.theta_call(s, k, t, r, sigma) for s in spots]
        plt.figure()
        plt.title('Theta - Call')
        plt.plot(spots, deltas)
        plt.show()

    def plot_theta_put(self, s=None, k=None, t=None, r=None, sigma=None):
        s, k, t, r, sigma = self._config_inputs(s, k, t, r, sigma)
        spots = [i for i in range(int(s * 0.01), int(s * 2), 1)]
        deltas = [self.theta_put(s, k, t, r, sigma) for s in spots]
        plt.figure()
        plt.title('Theta - Put')
        plt.plot(spots, deltas)
        plt.show()

    def plot_rho_call(self, s=None, k=None, t=None, r=None, sigma=None):
        s, k, t, r, sigma = self._config_inputs(s, k, t, r, sigma)
        spots = [i for i in range(int(s * 0.01), int(s * 2), 1)]
        deltas = [self.rho_call(s, k, t, r, sigma) for s in spots]
        plt.figure()
        plt.title('Rho - Call')
        plt.plot(spots, deltas)
        plt.show()

    def plot_rho_put(self, s=None, k=None, t=None, r=None, sigma=None):
        s, k, t, r, sigma = self._config_inputs(s, k, t, r, sigma)
        spots = [i for i in range(int(s * 0.01), int(s * 2), 1)]
        deltas = [self.rho_put(s, k, t, r, sigma) for s in spots]
        plt.figure()
        plt.title('Rho - Put')
        plt.plot(spots, deltas)
        plt.show()

    def plot_iv_call(self, ticker, expiry):
        chain = options_chain_by_expiry(ticker, expiry, "c")
        strikes = []
        ivs = []
        s = yf.download(tickers=ticker,
                        period='1d',
                        interval='1m')['Close'][-1]
        for index, row in chain.iterrows():
            strikes.append(row['strike'])
            iv = self.iv_call(s, row['strike'], row['daysToExpiration'] / 365, 0.04, 0.5,
                              0.5 * (row['bid'] + row['ask']))
            ivs.append(iv)

        plt.title(f'{ticker} IV')
        plt.plot(strikes, ivs)
        plt.show()

    def plot_iv_put(self):
        return

    def plot_iv_term_structure_call(self, ticker, strike):
        chain = options_chain_by_strike(ticker, strike, "c")
        expirations = []
        ivs = []
        s = yf.download(tickers=ticker,
                        period='1d',
                        interval='1m')['Close'][-1]
        for index, row in chain.iterrows():
            expirations.append(row['expiration'])
            iv = self.iv_call(s, strike, row['daysToExpiration'] / 365, 0.04, 0.5,
                              0.5 * (row['bid'] + row['ask']))
            ivs.append(iv)

        plt.title(f'{ticker} IV Term Structure, Strike={strike}')
        plt.plot(expirations, ivs)
        plt.xticks(rotation=45)
        plt.show()

    def plot_iv_surface(self, ticker):

        chain = get_options_grid(ticker, 'c')
        s = yf.download(tickers=ticker,
                        period='1d',
                        interval='1m')['Close'][-1]

        for expiry in chain.index:
            for strike in chain.columns:
                price = chain[strike][expiry]
                iv = self.iv_call(s, strike, expiry, 0.04, 0.5, price)
                chain[strike][expiry] = iv

        fig = plt.figure(figsize=(10, 8))

        ax = fig.add_subplot(111, projection='3d')

        x, y, z = chain.columns.values, chain.index.values, chain.values
        today = datetime.today()
        y = [(datetime.strptime(expiry, '%Y-%m-%d') - today).days for expiry in chain.index.values]

        X, Y = np.meshgrid(x, y)

        ax.set_ylabel('Days to expiration')
        ax.set_xlabel('Strike price')
        ax.set_zlabel('Implied volatility')
        ax.set_title(f'{ticker} Implied Volatility Surface - Call')

        ax.plot_surface(X, Y, z)
        plt.show()


if __name__ == "__main__":
    bs = BlackScholes(110, 120, 30 / 365, 0.04, 0.60)
    print(bs.plot_gamma())

    bs = BlackScholes()
    print(bs.plot_gamma(110, 120, 30 / 365, 0.04, 0.6))

    bs = BlackScholes('TSLA', 120, '2023-02-10', 0.04, 0.60)
    print(bs.plot_gamma())

    bs = BlackScholes()
    print(bs.plot_gamma('TSLA', 120, '2023-02-10', 0.04, 0.6))
