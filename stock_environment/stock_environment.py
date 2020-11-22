import numpy as np

import gym
from gym import spaces
from enum import Enum
from stock_environment.inventory import Inventory, StockEntry
from dataclasses import dataclass

INITIAL_MONEY = 10000
SIZE_WINDOW = 20


class MarketActions(Enum):
    BUY = 1
    SELL = 2
    NOTHING = 3


@dataclass
class MarketAction:
    action: MarketActions
    stock: str
    buy_price: float
    amount: int


class StockTradingEnvironment(gym.Env):
    """A stock trading environment for OpenAI gym"""

    metadata = {'render.modes': ['human']}
    reward_range = (0, float('inf'))

    def __init__(self, df):
        super(StockTradingEnvironment, self).__init__()
        self.df = df

        self.reset()

        # buy, sell, nothing, amount        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(SIZE_WINDOW, 6),
            dtype=np.float16
        )

    def step(self, action: MarketAction):
        self.steps += 1

        # Execute one time step within the environment
        if (action.action == MarketActions.BUY):
            stocks = self.inventory.stocks
            for _ in range(action.amount):
                self.inventory.stocks.append(StockEntry(
                    action.stock,
                    action.buy_price
                ))
                self.money -= action.buy_price
        elif (action.action == MarketActions.SELL):
            for _ in range(action.amount):
                stock = self.inventory.stocks.pop(0)
                self.money += stock

        obs = self._next_observation()
        reward = self._reward()
        done = self.steps >= len(self.df.index)
        return obs, reward, done, {}

    def reset(self):
        self.steps = SIZE_WINDOW        
        self.inventory = Inventory()
        self.money = INITIAL_MONEY

        return self._next_observation()

    def _next_observation(self):
        return self.df[(self.steps - SIZE_WINDOW): self.steps]
        

    def _reward(self):
        stock_worth = self.df["Close"][self.steps] * len(self.inventory.stocks)
        return ((self.money + stock_worth - INITIAL_MONEY) / INITIAL_MONEY) * 100

    def render(self, mode='human'):
        stock_worth = self.df["Close"][self.steps] * len(self.inventory.stocks)

        print(f'Step: {self.steps}')
        print(f'Balance: {self.money}')

        print(f'Worth: {self.money + stock_worth}')
        print(f'Projected profit {self.money + stock_worth - INITIAL_MONEY}')
