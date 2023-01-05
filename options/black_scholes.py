from datetime import date, datetime
import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf

from data.options_data import options_chain_by_expiry, options_chain_by_strike, get_options_grid


class BlackScholes:

    def __init__(self, s=None, k=None, t=None, r=None,
                 sigma=None):
        if type(s) == str:
            self.s = yf.download(tickers=s,
                                 period='1d',
                                 interval='1m')['Close'][-1]
        else:
            self.s = s
        self.k = k
        if type(t) == float:
            self.t = t
        elif type(t) == str:
            delta = datetime.strptime(t, '%d-%m-%Y').date() - date.today()
            self.t = int(delta.days + 1) / 365
        self.r = r
        self.sigma = sigma

    def _calc_d1(self, s=None, k=None, t=None, r=None, sigma=None):
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

        return (np.log(s / k) + (r + sigma ** 2 / 2) * t) / (sigma * np.sqrt(t))

    def _calc_d2(self, s=None, k=None, t=None, r=None, sigma=None):
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

        return self._calc_d1(s, k, t, r, sigma) - sigma * np.sqrt(t)

    def call(self, s=None, k=None, t=None, r=None, sigma=None):
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

        d1 = self._calc_d1(s, k, t, r, sigma)
        d2 = self._calc_d2(s, k, t, r, sigma)
        return s * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)

    def put(self, s=None, k=None, t=None, r=None, sigma=None):
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

        d1 = self._calc_d1(s, k, t, r, sigma)
        d2 = self._calc_d2(s, k, t, r, sigma)
        return k * np.exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1)

    def implied_vol_call(self, s=None, k=None, t=None, r=None, sigma0=0.30, option_price=10):
        if s is None:
            s = self.s
        if k is None:
            k = self.k
        if t is None:
            t = self.t
        if r is None:
            r = self.r
        sigma = sigma0
        max_iterations = 10000
        precision = 0.01

        for i in range(max_iterations):
            fx = self.call(s, k, t, r, sigma) - option_price
            if abs(fx) < precision:
                return sigma
            vega = self.vega(s, k, t, r, sigma)
            sigma = sigma - fx / vega
        return sigma

    def implied_vol_put(self, s=None, k=None, t=None, r=None, sigma0=0.30, option_price=10):
        if s is None:
            s = self.s
        if k is None:
            k = self.k
        if t is None:
            t = self.t
        if r is None:
            r = self.r
        sigma = sigma0
        max_iterations = 100000
        precision = 0.01

        for i in range(max_iterations):
            fx = self.put(s, k, t, r, sigma) - option_price
            if abs(fx) < precision:
                return sigma
            vega = self.vega(s, k, t, r, sigma)
            sigma = sigma - fx / vega
        return sigma

    def forward(self):
        return self.s - self.k * np.exp(-self.r * self.t)

    def delta_call(self, s=None, k=None, t=None, r=None, sigma=None):
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
        d1 = self._calc_d1(s, k, t, r, sigma)
        return norm.cdf(d1)

    def delta_put(self, s=None, k=None, t=None, r=None, sigma=None):
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
        d2 = self._calc_d2(s, k, t, r, sigma)
        return norm.cdf(d2)

    def gamma(self, s=None, k=None, t=None, r=None, sigma=None):
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
        d1 = self._calc_d1(s, k, t, r, sigma)
        return (norm.pdf(d1)) / (s * sigma * np.sqrt(t))

    def vega(self, s=None, k=None, t=None, r=None, sigma=None):
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
        d1 = self._calc_d1(s, k, t, r, sigma)
        vega = s * np.sqrt(t) * norm.pdf(d1)
        return vega

    def theta_call(self, s=None, k=None, t=None, r=None, sigma=None):
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
        d1 = self._calc_d1(s, k, t, r, sigma)
        d2 = self._calc_d2(s, k, t, r, sigma)
        p1 = - s * (norm.pdf(d1)) * sigma / (2 * np.sqrt(t))
        p2 = r * k * np.exp(-r * t) * norm.cdf(d2)
        return p1 - p2

    def theta_put(self, s=None, k=None, t=None, r=None, sigma=None):
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
        d1 = self._calc_d1(s, k, t, r, sigma)
        d2 = self._calc_d2(s, k, t, r, sigma)
        p1 = - s * (norm.pdf(d1)) * sigma / (2 * np.sqrt(t))
        p2 = r * k * np.exp(-r * t) * norm.cdf(-d2)
        return p1 + p2

    def rho_call(self, s=None, k=None, t=None, r=None, sigma=None):
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
        d2 = self._calc_d2(s, k, t, r, sigma)
        return k * t * np.exp(-r * t) * norm.cdf(d2)

    def rho_put(self, s=None, k=None, t=None, r=None, sigma=None):
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
        d2 = self._calc_d2(s, k, t, r, sigma)
        return -k * t * np.exp(-r * t) * norm.cdf(-d2)

    def plot_delta_call(self, s=None, k=None, t=None, r=None, sigma=None):
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
        spots = [i for i in range(int(s * 0.01), int(s * 2), 1)]
        deltas = [self.delta_call(s, k, t, r, sigma) for s in spots]
        plt.figure()
        plt.title('Delta - Call')
        plt.plot(spots, deltas)
        plt.show()

    def plot_delta_put(self, s=None, k=None, t=None, r=None, sigma=None):
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
        spots = [i for i in range(int(s * 0.01), int(s * 2), 1)]
        deltas = [self.delta_put(s, k, t, r, sigma) for s in spots]
        plt.figure()
        plt.title('Delta - Put')
        plt.plot(spots, deltas)
        plt.show()

    def plot_gamma(self, s=None, k=None, t=None, r=None, sigma=None):
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
        spots = [i for i in range(int(s * 0.01), int(s * 2), 1)]
        deltas = [self.gamma(s, k, t, r, sigma) for s in spots]
        plt.figure()
        plt.title('Gamma')
        plt.plot(spots, deltas)
        plt.show()

    def plot_vega(self, s=None, k=None, t=None, r=None, sigma=None):
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
        spots = [i for i in range(int(s * 0.01), int(s * 2), 1)]
        deltas = [self.vega(s, k, t, r, sigma) for s in spots]
        plt.figure()
        plt.title('Vega')
        plt.plot(spots, deltas)
        plt.show()

    def plot_theta_call(self, s=None, k=None, t=None, r=None, sigma=None):
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
        spots = [i for i in range(int(s * 0.01), int(s * 2), 1)]
        deltas = [self.theta_call(s, k, t, r, sigma) for s in spots]
        plt.figure()
        plt.title('Theta - Call')
        plt.plot(spots, deltas)
        plt.show()

    def plot_theta_put(self, s=None, k=None, t=None, r=None, sigma=None):
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
        spots = [i for i in range(int(s * 0.01), int(s * 2), 1)]
        deltas = [self.theta_put(s, k, t, r, sigma) for s in spots]
        plt.figure()
        plt.title('Theta - Put')
        plt.plot(spots, deltas)
        plt.show()

    def plot_rho_call(self, s=None, k=None, t=None, r=None, sigma=None):
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
        spots = [i for i in range(int(s * 0.01), int(s * 2), 1)]
        deltas = [self.rho_call(s, k, t, r, sigma) for s in spots]
        plt.figure()
        plt.title('Rho - Call')
        plt.plot(spots, deltas)
        plt.show()

    def plot_rho_put(self, s=None, k=None, t=None, r=None, sigma=None):
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
            iv = self.implied_vol_call(s, row['strike'], row['daysToExpiration'] / 365, 0.04, 0.5,
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
            iv = self.implied_vol_call(s, strike, row['daysToExpiration'] / 365, 0.04, 0.5,
                                       0.5 * (row['bid'] + row['ask']))
            ivs.append(iv)

        plt.title(f'{ticker} IV Term Structure, Strike={strike}')
        plt.plot(expirations, ivs)
        plt.xticks(rotation=45)
        plt.show()

    def plot_iv_surface(self, ticker):

        chain = get_options_grid(ticker, 'c')
        # chain.interpolate(inplace=True, axis=0)
        chain.fillna(method='ffill', inplace=True)

        fig = plt.figure(figsize=(10, 8))

        ax = fig.add_subplot(111, projection='3d')

        x, y, z = chain.columns.values, chain.index.values, chain.values

        X, Y = np.meshgrid(x, y)

        # set labels
        ax.set_xlabel('Days to expiration')
        ax.set_ylabel('Strike price')
        ax.set_zlabel('Implied volatility')
        ax.set_title('Call implied volatility surface')

        # plot
        ax.plot_surface(X, Y, z)  # todo cannot plot with dates as x-axis
        plt.show()


if __name__ == "__main__":
    bs = BlackScholes()

    # print(bs.implied_vol_call(108, 50, 30 / 365, 0.04, 0.7, 55))
    # print(bs.implied_vol_call(108, 55, 30 / 365, 0.04, 0.7, 68))
    # print(bs.implied_vol_call(108, 60, 30 / 365, 0.04, 0.7, 49))
    # print(bs.implied_vol_call(108, 70, 30 / 365, 0.04, 0.7, 39))
    # print(bs.implied_vol_call(108, 80, 30 / 365, 0.04, 0.7, 24))
    # print(bs.implied_vol_call(108, 85, 30 / 365, 0.04, 0.7, 24))
    # print(bs.implied_vol_call(108, 90, 30 / 365, 0.04, 0.7, 22))
    # print(bs.implied_vol_call(108, 95, 30 / 365, 0.04, 0.7, 19))
    # print(bs.implied_vol_call(108, 100, 30 / 365, 0.04, 0.7, 15))
    # print(bs.implied_vol_call(108, 105, 30 / 365, 0.04, 0.7, 12))
    # print(bs.implied_vol_call(108, 110, 30 / 365, 0.04, 0.7, 10))
    # print(bs.implied_vol_call(108, 200, 30 / 365, 0.04, 0.7, 0.21))
    # bs.plot_iv_call('TSLA', "2023-02-03")
    bs.plot_iv_surface('TSLA')
    # bs.plot_iv_term_structure_call('TSLA', 110)
