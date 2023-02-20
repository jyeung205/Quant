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

    def combos(self, n, i):
        return math.factorial(n) / (math.factorial(n-i)*math.factorial(i))

    def binomial(self, S0, K, T, r, sigma, N, type_='call'):
        dt = T / N
        u = np.exp(sigma * np.sqrt(dt))
        d = np.exp(-sigma * np.sqrt(dt))
        p = (np.exp(r * dt) - d) / (u - d)
        value = 0
        for i in range(N + 1):
            node_prob = self.combos(N, i) * p ** i * (1 - p) ** (N - i)
            ST = S0 * (u) ** i * (d) ** (N - i)
            if type_ == 'call':
                value += max(ST - K, 0) * node_prob
            elif type_ == 'put':
                value += max(K - ST, 0) * node_prob
            else:
                raise ValueError("type_ must be 'call' or 'put'")

        return value * np.exp(-r * T)
    def plot_binomial(self):
        N=100000
        sigma = 0.4
        T = 0.5
        K = 105
        r= 0.05
        dt = T / N
        Heads = np.exp(sigma * np.sqrt(dt))
        Tails = np.exp(-sigma * np.sqrt(dt))
        S0 = 100
        p = (np.exp(r*dt) - Tails )/( Heads - Tails )
        paths = np.random.choice([Heads,Tails],p=[p,1-p],size=(N,1))
        plt.plot(paths.cumprod(axis=0)*100, color='black');
        plt.xlabel('Steps')
        plt.ylabel('Stock Price')


if __name__ == "__main__":
    b = Binomial()
    b.binom_EU1(S0, K, T, r, sigma, N)
