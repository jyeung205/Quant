
import requests
import bs4 as bs
import pickle
import os
import pandas_datareader.data as web
import datetime
import pandas as pd
import sqlite3
import yfinance as yf


class Data:
    yesterday = datetime.datetime.today() - datetime.timedelta(days=1)
    yesterday = yesterday.strftime('%Y-%m-%d')

    def save_index_tickers(self, index):
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

        # Remove \n from each ticker
        tickers = [ticker[:-1] for ticker in tickers]

        if index == "S&P500":
            with open("tickers/sp500tickers.pickle", "wb") as f:
                pickle.dump(tickers, f)
        if index == "S&P100":
            with open("tickers/sp100tickers.pickle", "wb") as f:
                pickle.dump(tickers, f)
        if index == "NASDAQ 100":
            with open("tickers/nasdaq100tickers.pickle", "wb") as f:
                pickle.dump(tickers, f)

        print(tickers)
        return tickers

    def download_data(self, tickers, start="2000-01-01", end=yesterday):
        """
        Will add a repeated date if this function is executed on a non-trading day.
        """

        conn = sqlite3.connect("stock_data.db")
        c = conn.cursor()
        no_data_tickers = []

        if type(tickers) == str:
            tickers = [tickers]

        for ticker in tickers:
            c.execute(f"SELECT count(name) FROM sqlite_master WHERE type='table' AND name='{ticker}'")
            if c.fetchone()[0] == 1:  # If table exists
                for date in conn.execute(f"SELECT DATE FROM '{ticker}' ORDER BY DATE DESC LIMIT 1"):  # get last date
                    # print(date[0])
                    # last_date = datetime.datetime.strptime(date[0], "%Y-%m-%d")
                    if date[0] == self.yesterday:
                        print(f"{ticker} is up to date")
                    else:
                        last_date_next = datetime.datetime.strptime(date[0], "%Y-%m-%d") + datetime.timedelta(days=1)
                        last_date_next = last_date_next.strftime("%Y-%m-%d")
                        try:
                            print(f"Downloading Data for {ticker} from {last_date_next} to {self.yesterday}")
                            # df = web.DataReader(ticker, "yahoo", start=last_date_next, end=self.yesterday)
                            df = yf.Ticker(ticker, start=last_date_next, end=self.yesterday)[["High", "Low", "Open", "Close", "Volume", "Adj Close"]]
                            df.index = df.index.date
                            df.index.name = "Date"
                            df.to_sql(ticker, conn, schema=None, if_exists="append")
                        except:
                            print("No Data for " + ticker + f"from {last_date_next} to {self.yesterday}")
            else:  # If table does not exist
                # try:
                df = web.DataReader(ticker, "yahoo", start, end)
                print(f"Downloading Data for {ticker} from {start} to {end}")
                # except :
                # print("No Data for " + ticker)
                no_data_tickers.append(ticker)

                df.index = pd.to_datetime(df.index)
                df.index = df.index.date
                df.index.name = "Date"
                df.to_sql(ticker, conn, schema=None, if_exists="replace")

        if len(no_data_tickers) != 0:
            print("Finished!. No Data for", no_data_tickers)

    def delete_latest_date(self):
        """
        Delete last row in table in cases of duplicated dates
        :return:
        """
        conn = sqlite3.connect("stock_data.db")
        for ticker in conn.execute("select name from sqlite_master where type='table' "):
            print("Deleting latest date for", ticker[0])
            conn.execute(f"DELETE FROM '{ticker[0]}' WHERE Date = (SELECT MAX(Date) FROM '{ticker[0]}')")
        conn.commit()


if __name__ == "__main__":
    # data = Data()
    # tickers = data.save_index_tickers("S&P 500")
    # data.download_data(["AAPL"])
    # df = web.DataReader("AAPL", "yahoo")
    aapl = yf.Ticker('AAPL')
    print(aapl.history()[["High", "Low", "Open", "Close", "Volume"]])
