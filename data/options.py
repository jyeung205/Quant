import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt


def options_chain(ticker):
    asset = yf.Ticker(ticker)
    expirations = asset.options

    chains = pd.DataFrame()

    for expiration in expirations:
        # tuple of two dataframes
        opt = asset.option_chain(expiration)

        calls = opt.calls
        calls['optionType'] = "call"

        puts = opt.puts
        puts['optionType'] = "put"

        chain = pd.concat([calls, puts])
        chain['expiration'] = pd.to_datetime(expiration) + pd.DateOffset(hours=23, minutes=59, seconds=59)

        chains = pd.concat([chains, chain])

    chains["daysToExpiration"] = (chains.expiration - dt.datetime.today()).dt.days + 1
    chains = chains[chains.impliedVolatility >= 0.001]
    return chains


if __name__ == '__main__':
    chain = options_chain('TSLA')
    print(chain.head().to_string())
