import numpy as np
from scipy.stats import norm


class BlackScholes:

    def __init__(self, s=None, k=None, t=None, r=None, sigma=None):
        self.s = s
        self.k = k
        self.t = t
        self.r = r
        self.sigma = sigma
        self.d1 = self._calc_d1(s, k, t, r, sigma)
        self.d2 = self._calc_d2(s, k, t, r, sigma)

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

        return (np.log(s/k) + (r + sigma**2/2.)*t) / (sigma * np.sqrt(t))

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

    def delta_call(self):
        return norm.cdf(self.d1)

    def delta_put(self):
        return norm.cdf(-self.d2)

    def gamma(self):
        return (norm.pdf(self.d1)) / (self.s * self.sigma * np.sqrt(self.t))

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

    def theta_call(self):
        p1 = - self.s * (norm.pdf(self.d1)) * self.sigma / (2 * np.sqrt(self.t))
        p2 = self.r * self.k * np.exp(-self.r * self.t) * norm.cdf(self.d2)
        return p1 - p2

    def rho_call(self):
        return self.k * self.t * np.exp(-self.r * self.t) * norm.cdf(self.d2)

    def rho_put(self):
        return

    def plot_delta(self):
        return

    def plot_gamma(self):
        return

    def plot_rho(self):
        return

    def plot_vega(self):
        return

    def plot_iv(self):
        return


if __name__ == "__main__":
    bs = BlackScholes(123.18, 125, 39/365, 0.04, 0.8535)
    print(bs.call())
    print(bs.vega())
    print(bs.implied_vol(11.90, sigma0=0.7))
    print(bs.implied_vol(14.20, sigma0=0.7))
    # print(bs.implied_vol(14.20))
