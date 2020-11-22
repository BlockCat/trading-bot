from typing import Dict, List
from dataclasses import dataclass

class Inventory:
    def __init__(self):
        self.stocks: List[StockEntry] = []


@dataclass
class StockEntry:
    name: str
    buy_price: float
