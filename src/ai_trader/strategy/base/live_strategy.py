from ai_trader.strategy.base.base_strategy import BaseStrategy
from typing import Optional, Protocol, Literal
from tvDatafeed import Interval

class LiveStrategy(BaseStrategy, Protocol): 
    '''
        Interface (protocol) for live strategies
        input for version live:
            * live: bool = True
            * n_bars: bars number
            * interval: type Interval from tvDatafeed
            * tactic: function for making decisions
            * symbol: ticker 
            * exchange: stock name
            * stop-loss: flaot if you want to set it, none if not
    '''
    def __init__(   self,
                    *,
                    live: Literal[True],
                    n_bars: int,
                    interval: Interval,
                    symbol: str,
                    exchange: str,
                    stop_loss: Optional[float] = None,
                ) -> None: ...
        
    def get_n_bars(self) -> int: ...
    def get_interval(self) -> Interval: ...