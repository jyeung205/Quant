import numpy as np
from scipy.stats import norm


class BlackScholes:

    def __init__(self):
        return

    def _calculate_d1(s, k, t, sigma, r):
        d1 = (np.log(s / k) + (r + sigma ** 2 / 2) * t) / (sigma * np.sqrt(t))
        return d1

    def black_scholes_call(self, s, k, t, sigma, r):
        d1 = (np.log(s / k) + (r + sigma ** 2 / 2.) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        c = s * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
        return c

    def black_scholes_put(self, s, k, t, sigma, r):
        d1 = (np.log(s/k) + (r + sigma**2/2)*t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        p = k * np.exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1)
        return p

    def calc_delta(self):
        return

    def calc_gamma(self):
        return

    def calc_vega(self):
        return

    def calc_rho(self):
        return

    def calc_implied_vol(self):
        return

    def plot_delta(self):
        return

    def plot_gamma(self):
        return

    def plot_rho(self):
        return

    def plot_vega(self):
        return
