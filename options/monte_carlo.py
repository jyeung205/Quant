import numpy as np
import matplotlib.pyplot as plt


class MonteCarlo:
    def __init__(self):
        return

    def simulate_gbm(self, S, T, r, sigma, steps, N):
        dt = T / steps
        ST = np.log(S) + np.cumsum((r - sigma ** 2 / 2) * dt + sigma * np.sqrt(dt) * np.random.normal(size=(steps, N)),
                                   axis=0)
        return np.exp(ST)

    def plot_gbm(self, s, t, r, sigma, steps, N):
        paths = self.simulate_gbm(s, t, r, sigma, steps, N)

        plt.plot(paths)
        plt.xlabel("Time Increments")
        plt.ylabel("Stock Price")
        plt.title("Geometric Brownian Motion")
        plt.show()

    def options_price(self, s, t, k, r, sigma, steps, N):
        paths = self.simulate_gbm(s, t, r, sigma, steps, N)
        payoffs = np.maximum(paths[-1] - k, 0)
        option_price = np.mean(payoffs) * np.exp(-r * t)
        return option_price


if __name__ == "__main__":
    mc = MonteCarlo()
    print(mc.simulate_gbm(100, 1, 0.04, 0.05, 100, 100))
