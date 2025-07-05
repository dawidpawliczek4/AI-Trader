from datetime import datetime, timedelta
import pandas as pd
import kagglehub
import os
from tvDatafeed import TvDatafeed, Interval
from pytz import timezone
import exchange_calendars as ecals
from dotenv import load_dotenv
from dateutil import parser
from ai_trader.strategy.base.historical_strategy import HistoricalStrategy
from ai_trader.strategy.base.live_strategy import LiveStrategy
from typing import TypeGuard

def is_live_strategy(s: LiveStrategy | HistoricalStrategy) -> TypeGuard[LiveStrategy]:
    return s.get_live()

def is_historical_strategy(s: LiveStrategy | HistoricalStrategy) -> TypeGuard[HistoricalStrategy]:
    return not s.get_live()

class simulate:
    '''
    '''
    
    accout_ballance = 100000
    shares = 0
    
    def __init__(self, strategy: HistoricalStrategy | LiveStrategy):
        load_dotenv()
        self.live = strategy.get_live()
        self.symbol = strategy.get_symbol()
        self.exchange = strategy.get_exchange() 
        self.stop_loss = strategy.get_stop_loss()
        
        # set data
        if is_live_strategy(strategy):
            self.n_bars = strategy.get_n_bars()
            self.interval = strategy.get_interval()
            
        elif is_historical_strategy(strategy):
            self.t0 = strategy.get_t0()
            self.t1 = strategy.get_t1()
            self.delta = strategy.get_delta()
            
        else:
            raise ValueError("Unsupported strategy type. Use LiveStrategy or HistoricalStrategy.")
        
        # cook dataset 
        self.cook_stock_dataset()
        
    def timedeltaOfInterval(self, interval: Interval) -> timedelta:
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
        
    def is_nasdaq_open(self, date: datetime) -> bool:
        '''
            method for checking if nasdaq is open in the given datetime
        '''
        calendar = ecals.get_calendar("XNYS")
        if not isinstance(date, pd.Timestamp):
            date = pd.Timestamp(date)
            
        if not calendar.is_session(date.date()):
            return False

        open_time, close_time = calendar.session_open_close(date.date())
        open_time, close_time = open_time.astimezone("US/Eastern"), close_time.astimezone("US/Eastern")
        return open_time.time() <= date.time() <= close_time.time()
    
    def simulate(self):
        '''
            method for evaluation our tactic, return accout_ballance and shares at time t1
        '''
        
        while self.t0 < self.t1: 
            if not self.is_nasdaq_open(self.t0): # market is closed
                self.t0 += self.delta
                continue
                
            if self.price_df[self.price_df['date'] >= self.t0].empty: # we cant buy anything
                self.t0 += self.delta
                continue
            
            if self.stop_loss and self.price_df[self.price_df['date'] >= self.to].iloc[0] <= self.stop_loss:
                self.accout_ballance += self.shares * self.stop_loss
                self.shares = 0
            
            req = {
                "time": self.t0, 
                "symbol": self.symbol,
                "exchange": self.exchange, 
                "account_balance": self.accout_ballance,
                "shares": self.shares
            }
            if self.stop_loss != None:
                req["stop_loss"] = self.stop_loss
                
            res = self.tactic(req)
            if res['action'] == 'buy':
               if res['quantity'] <= 0:
                   continue 
                
               item = self.price_df[self.price_df['date'] >= self.t0].iloc[0]
               price = (item.high + item.low) / 2
               buy = min(price * res['quantity'], self.accout_ballance)
               
               self.shares += buy 
               self.accout_ballance -= price * res['quantity']
               
            elif res['quantity'] == 'sell':
                if res['quantity'] <= 0 or self.shares <= 0:
                    continue
                
                item = self.price_df[self.price_df['date'] >= self.t0].iloc[0]
                price = (item.high + item.low) / 2
                sell = min(res['quantity'], self.shares)
                
                self.shares -= sell
                self.accout_ballance += sell * price
            
            self.t0 += self.delta
            
        return (self.accout_ballance, self.shares)
    
    def convert_time_format(time_str):
        cleaned_time = " ".join(time_str.split())
        dt = parser.parse(cleaned_time)
        et = timezone("US/Eastern")
        return dt.replace(tzinfo=et) 
    
    def cook_news_dataset(self):
        if self.live == False:
            pass                        #TODO
        else:
            path = kagglehub.dataset_download("notlucasp/financial-news-headlines")
            
            csv1 = os.path.join(path, 'cnbc_headlines.csv')
            csv2 = os.path.join(path, 'guardian_headlines.csv')
            csv3 = os.path.join(path, 'reuters_headlines.csv')

            df1 = pd.read_csv(csv1)
            df2 = pd.read_csv(csv2)
            df3 = pd.read_csv(csv3)
            
            # cook df1
            df1 = df1.drop(columns=['Description']).dropna()
            df1['Time'] = df1['Time'].apply(self.convert_time_format)
            
            # cook df2
            df2 = df2.dropna()
            df2['Time'] = df2['Time'].apply(self.convert_time_format)
            
            # cook df3
            df3 = df3.drop(columns=['Description']).dropna()
            df3['Time'] = df3['Time'].apply(self.convert_time_format)
            
            # merge
            df = pd.concat([df1, df2, df3])
            df = df.sort_values(by="Time").reset_index(drop=True)
            self.news_df['Headlines'] = df['Headlines'].apply(lambda s : s.replace('\n\n\n', ''))
            df.to_csv('../data/news.csv', index=False)
            
    def cook_stock_dataset(self):
        if self.live == False:
            path = kagglehub.dataset_download("leukipp/hourly-stock-prices")
            csv = os.path.join(path, '2021', self.symbol + '.csv')
            self.price_df = pd.read_csv(csv)
            self.price_df['date'] = pd.to_datetime(self.price_df['date'])
            self.price_df = self.price_df[self.price_df['date'].apply(self.is_nasdaq_open)].reset_index(drop = True)
            self.price_df = self.price_df.drop(columns=['ticker', 'name'])
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
            self.price_df['date'] = self.price_df['date'].apply(lambda dt: dt.replace(tzinfo=timezone("UTC")))
            self.price_df['date'] = self.price_df['date'].apply(lambda dt: dt.astimezone("US/Eastern"))
            self.price_df.to_csv('../data/stock.csv', index=False)
            
            self.t0 = self.price_df['date'].iloc[0]
            self.t1 = self.price_df['date'].iloc[-1]