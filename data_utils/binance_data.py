from binance.client import Client
import os
import pandas as pd
from datetime import datetime
import sqlite3
import pickle


class BinanceData:

    def __init__(self):
        api_key = os.environ.get('BINANCE_API_KEY')
        secret_key = os.environ.get('BINANCE_SECRET_KEY')
        self.client = Client(api_key, secret_key)
        self.conn = sqlite3.connect("../data/crypto_data.db")
        self.c = self.conn.cursor()

    def save_tickers(self):
        prices = self.client.get_all_tickers()

        tickers = []
        for price in prices:
            tickers.append(price['symbol'])
        tickers.sort()

        usd_tickers = []
        for ticker in tickers:
            if 'USD' in ticker:
                usd_tickers.append(ticker)

        btc_tickers = []
        for ticker in tickers:
            if 'BTC' in ticker:
                btc_tickers.append(ticker)

        with open("../data/tickers/crypto_tickers.pickle", "wb") as f:
            pickle.dump(tickers, f)
        with open("../data/tickers/crypto_usd_tickers.pickle", "wb") as f:
            pickle.dump(usd_tickers, f)
        with open("../data/tickers/crypto_btc_tickers.pickle", "wb") as f:
            pickle.dump(btc_tickers, f)

    def get_binance_data(self, symbol, freq, start_date, end_date):
        data = self.client.get_historical_klines(symbol=symbol,
                                                 interval=freq,
                                                 start_str=start_date,
                                                 end_str=end_date,
                                                 )
        data_df = pd.DataFrame(data,
                               columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av',
                                        'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
        data_df = data_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        data_df['timestamp'] = pd.to_datetime(data_df['timestamp'], unit='ms')
        data_df.set_index('timestamp', inplace=True)
        return data_df

    def get_binance_futures_data(self, symbol, freq, start_date, end_date):
        start_timestamp = int(datetime.timestamp(
            datetime.strptime(start_date, '%Y-%m-%d')) * 1000)  # Convert start_date and end_date to timestamp
        end_timestamp = int(datetime.timestamp(datetime.strptime(end_date, '%Y-%m-%d')) * 1000)
        data = self.client.futures_klines(symbol=symbol,
                                          interval=freq,
                                          startTime=start_timestamp,
                                          endTime=end_timestamp,
                                          )
        data_df = pd.DataFrame(data,
                               columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av',
                                        'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
        data_df = data_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        data_df['timestamp'] = pd.to_datetime(data_df['timestamp'], unit='ms')
        data_df.set_index('timestamp', inplace=True)
        return data_df

    def download_binance_data(self, symbol, freq, start_date, end_date):  # todo check last date then append
        data_df = self.get_binance_data(symbol, freq, start_date, end_date)
        data_df.to_sql(f'{symbol}-{freq}', self.conn, schema=None, if_exists="replace")

    def download_binance_futures_data(self, symbol, freq, start_date, end_date):
        data_df = self.get_binance_data(symbol, freq, start_date, end_date)
        data_df.to_sql(f'{symbol}-future-{freq}', self.conn, schema=None, if_exists="replace")


if __name__ == '__main__':
    # save_tickers()

    # with open('crypto_tickers.pickle', 'rb') as f:
    #     tickers = pickle.load(f)

    # with open('crypto_usd_tickers.pickle', 'rb') as f:
    #     usd_tickers = pickle.load(f)
    #
    # for ticker in usd_tickers:
    #      download_all_time_binance_data(ticker, "1h", save=True)
    #

    # with open('tickers/crypto_btc_tickers.pickle', 'rb') as f:
    #     btc_tickers = pickle.load(f)
    # for ticker in btc_tickers:
    #     download_all_time_binance_data(ticker, "1d", save=True, futures=False)

    # df = download_all_time_binance_data('ARNBTC', '1m', save=True, futures=False)
    # print(df)

    bd = BinanceData()
    bd.download_binance_data('ALPHABNB', '1d', '2022-01-01', '2022-01-30')
