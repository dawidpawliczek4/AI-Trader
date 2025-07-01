from datetime import datetime

class Res:
    '''
        res to simulare from tactic:
    '''
    
    def __init__(self, action: str, quantity: float|None, set_stop_loss: float|None) -> None:
        '''
            required (always present):
                action: str - "buy", "sell" or "hold"
            optional:
                quantity: float - if action is different than "hold" this key is required
                set_stop_loss : float | None - if passed simulation will set stop-loss, if you will pass None stop-loss will be removed
        '''
        
        self.action = action
        self.quantity = quantity
        self.set_stop_loss = set_stop_loss
        
        self.validate
        
    def validate(self):
        if self.action not in ['buy', 'sell', 'hold']:
            raise ValueError('action must be equal to buy, sell or hold, now action is: {self.action}, make sure you did not make a typo or left space in the action name') 
        
        if self.action != 'hold' and self.quantity is None:
            raise ValueError('quantity must be set if action is not hold, now action is: {self.action}, quantity is: {self.quantity}')
        
        
        
        