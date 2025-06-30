from datetime import datetime

class Req:
    '''
        req from simulate to tactic
    '''
    
    def __init__(self, time: datetime, symbol: str, exchange: str, account_balance: float, shares: float, stop_loss: float|None) -> None:
        '''
            required:
                time: datatime - current datatime
                symbol: str - ticker symbol
                exchange: str - stock name
                account_balance: float
                shares: float
                
            optional:
                stop_loss: float - current stop_loss
        '''
        
        self.time = time 
        self.symbol = symbol 
        self.exchange = exchange 
        self.account_balance = account_balance 
        self.shares = shares 
        self.stop_loss = stop_loss