import seaborn as sns

from random_nn_agent_mini.agent import Agent
import wandb

from api_config import WANDB

sns.set()

WINDOW_LENGTH = 4


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

# for ticker in tickers[6:]:
#     analyze(ticker)

wandb.login(key=WANDB)

agent1 = Agent("EA")
agent1.train()