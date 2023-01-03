from datetime import date, datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import yfinance as yf


class BlackScholes:

    def __init__(self, s=None, k=None, t=None, r=None,
                 sigma=None):
        if type(s) == str:
            data = yf.download(tickers=s, period='1m', interval='1m')  # todo check if this is live price
            self.s = data['Close'][-1]
        else:
            self.s = s
        self.k = k
        if type(t) == float:
            self.t = t
        else:
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

        return (np.log(s / k) + (r + sigma ** 2 / 2.) * t) / (sigma * np.sqrt(t))

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

    def implied_vol(self, option_price, s=None, k=None, t=None, r=None, sigma0=0.30):
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
            fx = self.call(s, k, t, r, sigma) - option_price
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

    def plot_iv(self, ticker):
        return


if __name__ == "__main__":
    bs = BlackScholes('TSLA', 125, "10-02-2023", 0.04, 0.8535)

    print(bs.call())
