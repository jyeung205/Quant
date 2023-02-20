import numpy as np
import matplotlib.pyplot as plt


class Numerical:
    def __init__(self):
        return

    def simulate_gbm(self, s, t, r, sigma, steps, N):
        dt = t / steps
        st = np.log(s) + np.cumsum((r - sigma ** 2 / 2) * dt + sigma * np.sqrt(dt) * np.random.normal(size=(steps, N)),
                                   axis=0)
        return np.exp(st)

    def plot_gbm(self, s, t, r, sigma, steps, N):
        paths = self.simulate_gbm(s, t, r, sigma, steps, N)
        plt.plot(paths)
        plt.xlabel("Time Increments")
        plt.ylabel("Stock Price")
        plt.title("Geometric Brownian Motion")
        plt.show()

    def call_european(self, s, k, t, r, sigma, steps, N):
        paths = self.simulate_gbm(s, t, r, sigma, steps, N)
        payoffs = np.maximum(paths[-1] - k, 0)
        return np.mean(payoffs) * np.exp(-r * t)

    def put_european(self, s, k, t, r, sigma, steps, N):
        paths = self.simulate_gbm(s, t, r, sigma, steps, N)
        payoffs = np.maximum(k - paths[-1], 0)
        return np.mean(payoffs) * np.exp(-r * t)

    def delta_fdm_call(self, method):  # TODO greeks using finite difference
        if method == 'central':
            pass
        if method == 'forward':
            pass
        if method == 'backward':
            pass
        return

    def delta_fdm_put(self, method):
        if method == 'central':
            pass
        if method == 'forward':
            pass
        if method == 'backward':
            pass
        return


if __name__ == "__main__":
    mc = Numerical()
    mc.plot_gbm(100, 1, 0.04, 0.05, 100, 100)
    print(mc.options_price(100, 103, 1, 0.04, 0.05, 100, 100))
