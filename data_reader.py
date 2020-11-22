from alpha_vantage.timeseries import TimeSeries
from pandas import DataFrame, read_pickle

from api_config import API

import pickle
import os.path

PATH = "data/"


def read_daily_data(ticker: str) -> DataFrame:
    ts = TimeSeries(key=API, output_format='pandas', indexing_type='date')
    pp = PATH + ticker + ".daily"
    if (os.path.isfile(pp)):
        data = read_pickle(pp)
    else:
        data = ts.get_daily_adjusted(ticker, outputsize='full')[0]
        data.to_pickle(pp)

    df = DataFrame()
    df["Date"] = data.index
    df["Open"] = data["1. open"].values
    df["High"] = data["2. high"].values
    df["Low"] = data["3. low"].values
    df["Close"] = data["4. close"].values
    df["Adj Close"] = data["5. adjusted close"].values
    df["Volume"] = data["6. volume"].values
    return df
