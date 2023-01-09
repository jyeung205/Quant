import pandas as pd
import yfinance as yf
from datetime import datetime


def get_options_chain_by_expiry(ticker, expiry, type):
    asset = yf.Ticker(ticker)
    opt = asset.option_chain(expiry)

    if type == 'c' or type == 'call':
        chain = opt.calls
        chain['optionType'] = "call"
    elif type == 'p' or type == 'put':
        chain = opt.puts
        chain['optionType'] = "put"
    else:
        raise Exception('Input Correct Option type')

    chain['expiration'] = pd.to_datetime(expiry) + pd.DateOffset(hours=23, minutes=59, seconds=59)
    chain["daysToExpiration"] = (chain.expiration - datetime.today()).dt.days + 1
    chain.sort_values('strike', inplace=True)
    return chain


def get_options_chain_by_strike(ticker, strike, type):
    asset = yf.Ticker(ticker)
    expirations = asset.options
    chains = pd.DataFrame()
    for expiration in expirations:
        opt = asset.option_chain(expiration)

        if type == 'c' or type == 'call':
            chain = opt.calls
            chain['optionType'] = "call"
        elif type == 'p' or type == 'put':
            chain = opt.puts
            chain['optionType'] = "put"
        else:
            raise Exception('Input Correct Option type')

        chain = chain[chain["strike"] == strike]
        chain['expiration'] = pd.to_datetime(expiration) + pd.DateOffset(hours=23, minutes=59, seconds=59)
        chains = pd.concat([chains, chain])

    chains["daysToExpiration"] = (chains.expiration - datetime.today()).dt.days + 1
    chains.sort_values('expiration', inplace=True)
    return chains


def get_options_grid(ticker, type):
    asset = yf.Ticker(ticker)
    expirations = asset.options
    strikes = list(asset.option_chain(expirations[4]).calls.strike)
    chains = pd.DataFrame(index=expirations, columns=strikes)
    for expiration in expirations:
        opt = asset.option_chain(expiration)
        if type == 'c' or type == 'call':
            chain = opt.calls
        elif type == 'p' or type == 'put':
            chain = opt.puts
        else:
            raise Exception('Input Correct Option type')
        for index, row in chain.iterrows():
            strike = row['strike']
            if strike in strikes:
                chains[strike][expiration] = 0.5*(row['bid']+row['ask'])
    chains.fillna(method='ffill', inplace=True)
    # chain.interpolate(inplace=True, method='ffill')
    # chains.dropna(thresh=4, inplace=True)
    return chains


if __name__ == '__main__':
    # chain = options_chain_by_strike('TSLA', 120, "c")
    # print(chain.head(100).to_string())
    # o = option_chains_pyquantnews('TSLA')
    print(get_options_grid('TSLA', 'c').to_string())
