from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import kagglehub
import os
from tvDatafeed import TvDatafeed, Interval
from pytz import timezone
import exchange_calendars as ecals
from dotenv import load_dotenv

class simulate:
    '''
        class for evaluating tactic: 
        
            input for version live:
                * live: bool = True
                * n_bars: bars number
                * interval: type Interval from tvDatafeed
                * tactic: function for making decisions
                * symbol: ticker 
                * exchange: stock name
            
            input for version on historical data (2021 year):
                * live: bool = False 
                * t0: start time of simulation
                * t1: end time of simulation
                * delta: time delta between using tactic
                * tactic: function for making decisions
                * symbol: ticker
                * exchange: stock name
                
            tactic: 
                * receive (time, symbol, exchange, accout_ballance, shares)
                * have to return one of 3 options:
                    * ("buy", quantity)
                    * ("sell", quantity)
                    * ("hold", 0)
                    
            important info: 
                * we assume that we are in the et timezone. so proper example of datatime is (2021-01-04 04:00:00-05:00) or (2021-03-24 08:00:00-04:00)
    '''
    
    accout_ballance = 100000
    shares = 0
    
    def __init__(self, live: bool, symbol: str, exchange: str, tactic, n_bars: int = None, interval: Interval = None, t0: datetime = None, t1: datetime = None, delta: timedelta = None):
        load_dotenv()
        
        # set data
        if live:
            self.live = live 
            self.n_bars = n_bars
            self.interval = interval 
            self.tactic = tactic 
            self.symbol = symbol
            self.exchange = exchange
            self.delta = self.timedeltaOfInterval(interval)
            
        else: 
            self.live = live 
            self.t0 = t0 
            self.t1 = t1 
            self.delta = delta 
            self.tactic = tactic 
            self.symbol = symbol 
            self.exchange = exchange
        
        # cook dataset 
        self.cook_stock_dataset()
        
    def timedeltaOfInterval(interval):
        if interval == Interval.in_1_minute:
            return timedelta(minutes=1)
        elif interval == Interval.in_3_minute:
            return timedelta(minutes=3)
        elif interval == Interval.in_5_minute:
            return timedelta(minutes=5)
        elif interval == Interval.in_15_minute:
            return timedelta(minutes=15)
        elif interval == Interval.in_30_minute:
            return timedelta(minutes=30)
        elif interval == Interval.in_45_minute:
            return timedelta(minutes=45)
        elif interval == Interval.in_1_hour:
            return timedelta(hours=1)
        elif interval == Interval.in_2_hour:
            return timedelta(hours=2)
        elif interval == Interval.in_3_hour:
            return timedelta(hours=3)
        elif interval == Interval.in_4_hour:
            return timedelta(hours=4)
        elif interval == Interval.in_daily:
            return timedelta(days=1)
        elif interval == Interval.in_weekly:
            return timedelta(weeks=1)
        elif interval == Interval.in_monthly:
            return timedelta(days=30)
        else:
            raise ValueError(f"Unsupported interval: {interval}")
        
    def is_nasdaq_open(date: datetime) -> bool:
        '''
            method for checking if nasdaq is open in the given datetime
        '''
        calendar = ecals.get_calendar("XNYS")
        if not isinstance(date, pd.Timestamp):
            date = pd.Timestamp(date)
            
        if not calendar.is_session(date.date()):
            return False

        open_time, close_time = calendar.session_open_close(date.date())
        return open_time.time() <= date.time() <= close_time.time()
    
    def simulate_offline(self):
        '''
            method for evaluation our tactic, return accout_ballance and shares at time t1
        '''
        while self.t0 < self.t1: 
            if not (4 <= self.t0.hour <= 18): # market is closed
                self.t0 += self.delta
                continue
                
            if self.price_df[self.price_df['date'] >= self.t0].empty: # we cant buy anything
                self.t0 += self.delta
                continue
            
            action, quantity = self.tactic(self.t0, self.symbol, self.exchange, self.accout_ballance, self.shares)
            if action == 'buy':
               if quantity <= 0:
                   continue 
                
               item = self.price_df[self.price_df['date'] >= self.t0].iloc[0]
               price = (item.high + item.low) / 2
               buy = min(price * quantity, self.accout_ballance)
               
               self.shares += buy 
               self.accout_ballance -= price * quantity
               
            elif action == 'sell':
                if quantity <= 0 or self.shares <= 0:
                    continue
                
                item = self.price_df[self.price_df['date'] >= self.t0].iloc[0]
                price = (item.high + item.low) / 2
                sell = min(quantity, self.shares)
                
                self.shares -= sell
                self.accout_ballance += sell * price
            
            self.t0 += self.delta
            
        return (self.accout_ballance, self.shares)
    
    def simulate_live(self):
        pass
            
    def cook_stock_dataset(self):
        if self.live == False:
            path = kagglehub.dataset_download("leukipp/hourly-stock-prices")
            csv = os.path.join(path, '2021', self.symbol + '.csv')
            self.price_df = pd.read_csv(csv)
            self.price_df['date'] = pd.to_datetime(self.price_df['date'])
            self.price_df.to_csv('../data/stock.csv', index=False)
            
        else:
            username = os.getenv("USERNAME")
            password = os.getenv("PASSWORD")
            tv = TvDatafeed(username, password)
            
            self.price_df = tv.get_hist(symbol=self.symbol, exchange=self.exchange, interval=self.interval, n_bars=self.n_bars)
            self.price_df = self.price_df.drop(columns=['symbol'])
            self.price_df.reset_index(inplace=True)
            self.price_df.rename(columns={'datetime': 'date'}, inplace=True)
            self.price_df['date'] = pd.to_datetime(self.price_df['date'])
            self.price_df['date'] = self.price_df['date'] - timedelta(hours=1)
            self.price_df['date'] = self.price_df['date'].apply(lambda dt: dt.replace(tzinfo=timezone("US/Eastern")))
            self.price_df.to_csv('../data/stock.csv', index=False)
            
            self.t0 = self.price_df['date'].iloc[0]
            self.t1 = self.price_df['date'].iloc[-1]