from ..simulate.req import Req 
from ..simulate.res import Res
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    '''
        Base class for all strategies, you must implement tactic method
    '''
    
    def __init__(self):
        pass
    
    def tactic(self, req: Req) -> Res:
        pass

    
    