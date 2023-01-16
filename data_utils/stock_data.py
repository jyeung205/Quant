import requests
import bs4 as bs
import pickle
import datetime
from datetime import datetime, timedelta
import pandas as pd
import sqlite3

from alpha_vantage.timeseries import TimeSeries


class StockData:
    yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')

    def __init__(self):
        self.conn = sqlite3.connect("../data/stock_data.db")
        self.c = self.conn.cursor()
        self.app = TimeSeries("LHE90GZROAECNZ96",
                              output_format='pandas')  # TODO add alpha vantagae api key to env variables

    def save_index_tickers(self, index='S&P 100'):
        """
        This function saves the list of S&P 500 tickers in a .pickle file
        """
        if index == "S&P 500":
            resp = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        elif index == "S&P 100":
            resp = requests.get("https://en.wikipedia.org/wiki/S%26P_100")
        elif index == "NASDAQ 100":
            resp = requests.get("https://en.wikipedia.org/wiki/NASDAQ-100")
        else:
            print('Input valid index!!!')
            return

        soup = bs.BeautifulSoup(resp.text, "lxml")
        table = soup.find("table", {"class": "wikitable sortable", "id": "constituents"})
        tickers = []
        for row in table.findAll("tr")[1:]:
            if index == "S&P 100" or index == "S&P 500":
                ticker = row.findAll("td")[0].text
            elif index == "NASDAQ 100":
                ticker = row.findAll("td")[1].text
            tickers.append(ticker)

        tickers = [ticker[:-1] for ticker in tickers]  # Remove \n from each ticker

        if index == "S&P500":
            with open("../data/tickers/sp500tickers.pickle", "wb") as f:
                pickle.dump(tickers, f)
        if index == "S&P100":
            with open("../data/tickers/sp100tickers.pickle", "wb") as f:
                pickle.dump(tickers, f)
        if index == "NASDAQ 100":
            with open("../data/tickers/nasdaq100tickers.pickle", "wb") as f:
                pickle.dump(tickers, f)

        print(tickers)
        return tickers

    def get_stock_data(self, ticker, start, end):

        df = self.app.get_daily_adjusted(ticker, 'full')[0]
        df.sort_index(inplace=True)
        new_df = pd.concat([df['2. high'][start:end],
                            df['3. low'][start:end],
                            df['1. open'][start:end],
                            df['4. close'][start:end],
                            df['6. volume'][start:end],
                            df['5. adjusted close'][start:end]], axis=1)
        new_df.columns = ['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']
        new_df.index = new_df.index.date
        new_df.index.name = "Date"
        return new_df

    def download_stock_data(self, ticker, start, end):
        self.c.execute(f"SELECT count(name) FROM sqlite_master WHERE type='table' AND name='{ticker}'")
        if self.c.fetchone()[0] == 1:  # If table already exists in db
            for date in self.conn.execute(f"SELECT DATE FROM '{ticker}' ORDER BY DATE DESC LIMIT 1"):  # get last date
                if date[0] == end:
                    print(f"{ticker} is up to date")
                else:
                    next_date_in_db = datetime.strptime(date[0], "%Y-%m-%d") + timedelta(days=1)
                    next_date_in_db = next_date_in_db.strftime("%Y-%m-%d")
                    print(f"Downloading Data for {ticker} from {next_date_in_db} to {end}")
                    df = self.get_stock_data(ticker, next_date_in_db, end)
                    df.to_sql(ticker, self.conn, schema=None, if_exists="append")
        else:  # If table does not exist in db
            print(f"Downloading Data for {ticker} from {start} to {end}")
            df = self.get_stock_data(ticker, start, end)
            df.to_sql(ticker, self.conn, schema=None, if_exists="replace")

    def download_many_stock_data(self, tickers, start="2000-01-01", end=yesterday):
        """
        Only run on trading days. Function will add a repeated date if executed on a non-trading day.
        """
        no_data_tickers = []

        for ticker in tickers:
            self.download_stock_data(ticker, start, end)

    def delete_latest_date(self):
        """
        Delete last row in table in cases of duplicated dates
        :return:
        """
        conn = sqlite3.connect("../data/stock_data.db")
        for ticker in conn.execute("select name from sqlite_master where type='table' "):
            print("Deleting latest date for", ticker[0])
            conn.execute(f"DELETE FROM '{ticker[0]}' WHERE Date = (SELECT MAX(Date) FROM '{ticker[0]}')")
        conn.commit()
        
    def read_stock_data(self, ticker):
        return


if __name__ == "__main__":
    data = StockData()
    data.download_many_stock_data(['AAL'])
