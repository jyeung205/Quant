from binance.client import Client
import os
import pandas as pd
from datetime import timedelta, datetime
import math
from dateutil import parser
import sqlite3
import pickle

binsizes = {"1m": 1, "5m": 5, "1h": 60, "1d": 1440}
batch_size = 750


class BinanceData:

    def __init__(self):
        api_key = os.environ.get('BINANCE_API_KEY')
        secret_key = os.environ.get('BINANCE_SECRET_KEY')
        conn = sqlite3.connect("../data/crypto_data.db")
        c = conn.cursor()
        self.client = Client(api_key, secret_key)

    def save_tickers(self):
        prices = self.client.get_all_tickers()

        # Get crypto symbols
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

        # Write into pickle file
        with open("../data/tickers/crypto_tickers.pickle", "wb") as f:
            pickle.dump(tickers, f)

        with open("../data/tickers/crypto_usd_tickers.pickle", "wb") as f:
            pickle.dump(usd_tickers, f)

        with open("../data/tickers/crypto_btc_tickers.pickle", "wb") as f:
            pickle.dump(btc_tickers, f)
        print(btc_tickers)
        return

    def minutes_of_new_data(self, symbol, kline_size, data, source):
        try:
            if len(data) > 0:
                # old = parser.parse(data["timestamp"].iloc[-1])
                old = parser.parse(data.index[-1])
            elif source == "binance":
                old = datetime.strptime('1 Jan 2017', '%d %b %Y')
            if source == "binance":
                new = pd.to_datetime(self.client.get_klines(symbol=symbol, interval=kline_size)[-1][0], unit='ms')
            return old, new
        except:
            print(f'Could not get last date for {symbol}')
            return

    def download_binance_data(self, symbol, kline_size, start_date, end_date, futures, save):
        start_timestamp = int(datetime.timestamp(datetime.strptime(start_date, '%Y-%m-%d')) * 1000)  # Convert start_date and end_date to timestamp
        end_timestamp = int(datetime.timestamp(datetime.strptime(end_date, '%Y-%m-%d')) * 1000)

        if futures:
            klines = self.client.futures_klines(symbol=symbol,
                                                interval=kline_size,
                                                startTime=start_timestamp,
                                                endTime=end_timestamp,
                                                limit=1000)
        else:
            klines = self.client.get_historical_klines(symbol=symbol,
                                                       interval=kline_size,
                                                       start_str=start_date,
                                                       end_str=end_date,
                                                       limit=1000)

        data_df = pd.DataFrame(klines,
                               columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av',
                                        'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])

        data_df['timestamp'] = pd.to_datetime(data_df['timestamp'], unit='ms')
        data_df['close_time'] = pd.to_datetime(data_df['close_time'], unit='ms')
        data_df.set_index('timestamp', inplace=True)

        if save:  # todo separate download/save data from just getting data
            if futures:
                data_df.to_sql(f'{symbol}-future-{kline_size}', self.conn, schema=None, if_exists="replace")
            else:
                data_df.to_sql(f'{symbol}-{kline_size}', self.conn, schema=None, if_exists="replace")
        return data_df

    def download_all_time_binance_data(self, symbol, kline_size, save=False, futures=False):
        # if not os.path.exists("csv_data/%s" % (symbol)):
        #     os.makedirs("csv_data/%s" % (symbol))
        #
        # filename = '%s-%s-data.csv' % (symbol, kline_size)
        #
        # if os.path.isfile(filename):
        #     data_df = pd.read_csv(filename)
        # else:
        #     data_df = pd.DataFrame()
        try:
            self.c.execute(f"SELECT count(name) FROM sqlite_master WHERE type='table' AND name='{symbol}-{kline_size}' ")

            if self.c.fetchone()[0] == 1:  # If table exists
                data_df = pd.read_sql(
                    f"SELECT * FROM '{symbol}-{kline_size}'",
                    self.conn,
                    index_col="timestamp",
                    parse_dates=True
                )
            else:
                data_df = pd.DataFrame()

            oldest_point, newest_point = self.minutes_of_new_data(symbol, kline_size, data_df, source="binance")

            delta_min = (newest_point - oldest_point).total_seconds() / 60
            available_data = math.ceil(delta_min / binsizes[kline_size])

            if oldest_point == datetime.strptime('1 Jan 2013', '%d %b %Y'):
                print('Downloading all available %s data for %s. Be patient..!' % (kline_size, symbol))
            else:
                print('Downloading %d minutes of new data available for %s, i.e. %d instances of %s data.' % (
                    delta_min, symbol, available_data, kline_size))

            if futures:
                klines = self.client.futures_klines(symbol=symbol,
                                                    interval=kline_size,
                                                    start_str=oldest_point.strftime("%d %b %Y %H:%M:%S"),
                                                    end_str=newest_point.strftime("%d %b %Y %H:%M:%S"),
                                                    limit=1000)
            else:
                klines = self.client.get_historical_klines(symbol=symbol,
                                                           interval=kline_size,
                                                           start_str=oldest_point.strftime("%d %b %Y %H:%M:%S"),
                                                           end_str=newest_point.strftime("%d %b %Y %H:%M:%S"),
                                                           limit=1000)
            data = pd.DataFrame(klines,
                                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                         'quote_av',
                                         'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])

            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            data['close_time'] = pd.to_datetime(data['close_time'], unit='ms')

            if len(data_df) > 0:
                temp_df = pd.DataFrame(data)
                data_df = data_df.append(temp_df)
            else:
                data_df = data

            data_df.set_index('timestamp', inplace=True)

            if save:
                # if not os.path.exists("csv_data/%s/%s" % (symbol, filename)):
                # data_df.to_csv("csv_data/%s/%s" % (symbol, filename))
                if futures:
                    data_df.to_sql(f'{symbol}-future-{kline_size}', self.conn, schema=None, if_exists="replace")
                else:
                    data_df.to_sql(f'{symbol}-{kline_size}', self.conn, schema=None, if_exists="replace")

            print('All caught up..!')

            return data_df

        except:
            print(f'Could not get data for {symbol}')
            return


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

    df = bd.download_binance_data(symbol='BCHUSDT',
                               kline_size='15m',
                               start_date='2021-03-01',
                               end_date='2021-03-07',
                               futures=False,
                               save=False)

    print(df)
    df = bd.download_binance_data(symbol='BCHUSDT',
                               kline_size='15m',
                               start_date='2021-03-01',
                               end_date='2021-03-07',
                               futures=True,
                               save=False)
    print(df)

    # klines = client.get_historical_klines(symbol='BCHUSDT',
    #                                       interval='1m',
    #                                       start_str='2021-02-04',
    #                                       end_str='2021-02-05',
    #                                       limit=1000)
    #
    # data_df = pd.DataFrame(klines,
    #                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av',
    #                                 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
    #
    # data_df['timestamp'] = pd.to_datetime(data_df['timestamp'], unit='ms')
    # data_df['close_time'] = pd.to_datetime(data_df['close_time'], unit='ms')
    #
    # # if len(data_df) > 0:
    # #     temp_df = pd.DataFrame(data)
    # #     data_df = data_df.append(temp_df)
    # # else:
    # #     data_df = data
    #
    # data_df.set_index('timestamp', inplace=True)
    # print(data_df)
