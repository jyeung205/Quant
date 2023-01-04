import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt


def options_chain(ticker, expiration, type):
    asset = yf.Ticker(ticker)
    # expirations = asset.options

    opt = asset.option_chain(expiration)

    if type == 'c' or type == 'call':
        chain = opt.calls
        chain['optionType'] = "call"
    elif type == 'p' or type == 'put':
        chain = opt.puts
        chain['optionType'] = "put"
    else:
        raise Exception('Input Correct Option type')

    chain['expiration'] = pd.to_datetime(expiration) + pd.DateOffset(hours=23, minutes=59, seconds=59)
    chain["daysToExpiration"] = (chain.expiration - dt.datetime.today()).dt.days + 1
    chain.sort_values('strike', inplace=True)
    return chain


if __name__ == '__main__':
    chain = options_chain('TSLA', "2023-02-03", "p")
    print(chain.head(100).to_string())
