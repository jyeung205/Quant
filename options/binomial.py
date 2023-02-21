import numpy as np
import matplotlib.pyplot as plt
import math

N = 4
S0 = 100
T = 0.5
sigma = 0.4
K = 105
r = 0.05


class Binomial:

    def __init__(self):
        return

    def _combos(self, n, i):
        return math.factorial(n) / (math.factorial(n-i)*math.factorial(i))

    def _binomial(self, S0, K, T, r, sigma, N, type_='call'):
        dt = T / N
        u = np.exp(sigma * np.sqrt(dt))
        d = np.exp(-sigma * np.sqrt(dt))
        p = (np.exp(r * dt) - d) / (u - d)
        value = 0
        for i in range(N + 1):
            node_prob = self._combos(N, i) * p ** i * (1 - p) ** (N - i)
            ST = S0 * (u) ** i * (d) ** (N - i)
            if type_ == 'call':
                value += max(ST - K, 0) * node_prob
            elif type_ == 'put':
                value += max(K - ST, 0) * node_prob
            else:
                raise ValueError("type_ must be 'call' or 'put'")

        return value * np.exp(-r * T)

    def call(self, s0, k, t, r, sigma, N):
        return self._binomial(s0, k, t, r, sigma, N, 'call')

    def put(self, s0, k, t, r, sigma, N):
        return self._binomial(s0, k, t, r, sigma, N, 'put')

    def plot_simulation(self):
        dt = T / N
        Heads = np.exp(sigma * np.sqrt(dt))
        Tails = np.exp(-sigma * np.sqrt(dt))
        p = (np.exp(r*dt) - Tails)/(Heads - Tails)
        paths = np.random.choice([Heads, Tails], p=[p, 1-p], size=(N, 1))
        plt.plot(paths.cumprod(axis=0)*100, color='black')
        plt.xlabel('Steps')
        plt.ylabel('Stock Price')
        plt.show()


if __name__ == "__main__":
    b = Binomial()
    print(b.call())
    # b.plot_simulation()
