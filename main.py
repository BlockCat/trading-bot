from agent import Agent
from model import Model
import numpy as np
import pandas as pd

import os.path

from alpha_vantage.timeseries import TimeSeries
from pandas import DataFrame, read_pickle
import mplfinance as mpf


import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import random

import pickle

from api_config import API
sns.set()


PATH = "data/"


def read_daily_data(ticker: str) -> DataFrame:
    ts = TimeSeries(key=API, output_format='pandas', indexing_type='date')
    pp = PATH + ticker + ".daily"
    if (os.path.isfile(pp)):
        data = read_pickle(pp)
    else:
        data, meta_data = ts.get_daily_adjusted(ticker, outputsize='full')
        data.to_pickle(pp)
    return data

def read_weights(ticker: str, model: Model) -> bool:
    pp = PATH + ticker + ".model"
    if (os.path.isfile(pp)):        
        weights = pickle.load(open(pp, 'rb'))
        model.set_weights(weights)
        return True

    return False


def write_weights(ticker: str, model: Model):
    pp = PATH + ticker + ".model"    
    pickle.dump(model.get_weights(), open(pp, 'wb'))
    

def analyze(ticker):
    read_data = read_daily_data(ticker)

    df = DataFrame()
    df["Date"] = read_data.index
    df["Open"] = read_data["1. open"].values
    df["High"] = read_data["2. high"].values
    df["Low"] = read_data["3. low"].values
    df["Close"] = read_data["4. close"].values
    df["Adj Close"] = read_data["5. adjusted close"].values
    df["Volume"] = read_data["6. volume"].values

    df_train = df[-400:-200]
    df_test = df[-200:]

    print(df_train.head(5))
    close_train = df_train.Close.values.tolist()
    close_test = df_test.Close.values.tolist()
    window_size = 30
    skip = 1
    initial_money = 10000

    model = Model(input_size=window_size, layer_size=500, output_size=3)
    agent = Agent(model=model,
                window_size=window_size,
                trend=close_train,
                skip=skip,
                initial_money=initial_money)


    print("Starting training")
    if (not read_weights(ticker, model)):
        agent.fit(iterations=500, checkpoint=10)
        write_weights(ticker, model)
    print("Finished training")

    states_buy, states_sell, total_gains, invest = agent.buy(close_test)
    print(states_buy, states_sell, total_gains, invest)

    fig = plt.figure(figsize=(15, 5))
    plt.plot(close_test, color='r', lw=2.)
    plt.plot(close_test, '^', markersize=10, color='m',
            label='buying signal', markevery=states_buy)
    plt.plot(close_test, 'v', markersize=10, color='k',
            label='selling signal', markevery=states_sell)
    plt.title('total gains %f, total investment %f%%' % (total_gains, invest))
    plt.legend()
    plt.show()

tickers = [
    "GOOG",
    "EA",
    "SNE",
    "NTDOY",
    "UBI.PA",
    "NIO",
    "MSFT",
    "AAPL",
    "AMZN",
    "INTC",
    "CSCO",    
]

for ticker in tickers[6:]:
    analyze(ticker)