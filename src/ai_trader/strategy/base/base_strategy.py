from ai_trader.simulate.req import Req 
from ai_trader.simulate.res import Res
from typing import Optional, Protocol, Literal
from datetime import datetime, timedelta

class BaseStrategy(Protocol):
    '''
        Interface (protocol) for all strategies
        important info: 
            * we assume that we are in the et timezone. so proper example of datatime is (2021-01-04 04:00:00-05:00) or (2021-03-24 08:00:00-04:00)
    '''
    
    def run(self, req: Req) -> Res: ...
    def get_live(self) -> bool: ...
    def get_symbol(self) -> str: ... 
    def get_exchange(self) -> str: ... 
    def get_stop_loss(self) -> Optional[float]: ...