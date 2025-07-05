from ai_trader.strategy.base.base_strategy import BaseStrategy
from typing import Optional, Protocol, Literal
from datetime import datetime, timedelta

class HistoricalStrategy(BaseStrategy, Protocol):
    '''
        Interface (protocol) for historical strategies
        
        input for version on historical data (year 2021):
            * live: bool = False 
            * t0: start time of simulation
            * t1: end time of simulation
            * delta: time delta between using tactic
            * symbol: ticker
            * exchange: stock name
            * stop-loss: flaot if you want to set it, none if not
    '''
    def __init__(   self, 
                    *,
                    live: Literal[False],
                    t0: datetime,
                    t1: datetime, 
                    delta: timedelta, 
                    symbol: str, 
                    exchange: str,
                    stop_loss: Optional[float] = None
                ) -> None: ... 
    
    def get_t0(self) -> timedelta: ... 
    def get_t1(self) -> timedelta: ...
    def get_delta(self) -> timedelta: ... 