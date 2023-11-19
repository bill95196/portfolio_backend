#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 22:03:43 2023

@author: huangqini
"""

import os 
import pandas as pd
os.chdir('/Users/huangqini/Desktop/TradeUp')
transaction_df = pd.read_csv('sample.csv')
# data processing
transaction_df = transaction_df.iloc[3:]
date_time_split = transaction_df['Portfolio ID'].str.split('T', expand=True)
transaction_df['date'] = date_time_split[0]
transaction_df['time'] = date_time_split[1].str[:8]
transaction_df['time'] = pd.to_datetime(transaction_df['time'],format= '%H:%M:%S' ).dt.time

transaction_df = transaction_df.rename(columns={'User Name': 'tic', 'Cash Left':'Qty','Unnamed: 7':'Amount'})

transaction_df = transaction_df[['date','time','tic','Qty','Amount']]
transaction_df['Qty'] = pd.to_numeric(transaction_df['Qty'], errors='coerce', downcast='integer')
transaction_df['Amount'] = pd.to_numeric(transaction_df['Amount'], errors='coerce', downcast='integer')

transaction_df['Price'] = abs(transaction_df['Amount']/transaction_df['Qty'])
transaction_df.loc[transaction_df['Amount'] < 0, 'Qty'] *= -1

transaction_df['time'] = pd.to_datetime(transaction_df['time'],format= '%H:%M:%S' ).dt.time
tran = transaction_df.groupby(['date','tic'])['Qty'].sum()
cha = transaction_df.groupby(['date','tic'])['Amount'].sum()
my_dataframe = pd.DataFrame({'Qty_chg': tran})
my_dataframe['Amount_chg'] = cha
my_dataframe['Qty']  = my_dataframe.groupby(['tic'])['Qty_chg'].cumsum()
my_dataframe['Amount']  = my_dataframe['Amount_chg'].cumsum()

from yahoodownloader import YahooDownloader 

tickers = set(transaction_df['tic'])
Start_Date = '2023-05-01'
End_Date = '2023-11-03'

daily_return_df = YahooDownloader(start_date = Start_Date,
                     end_date = End_Date,
                     ticker_list = tickers).fetch_data()


daily_return_df = daily_return_df.drop(['open','high','low','volume'], axis=1)
daily_return_df = daily_return_df.set_index(['date','tic'])

initial_balance = 1000000
merged_data = pd.merge(daily_return_df, my_dataframe, left_on=['date','tic'], right_on=['date','tic'], how='outer', indicator=True)
merged_data['Amount'] = merged_data['Amount'].ffill()
merged_data['Qty']  = merged_data.groupby(['tic'])['Qty'].ffill()

merged_data = merged_data.drop(['day','Qty_chg','Amount_chg','_merge'],axis=1)
merged_data =merged_data.fillna(0)
merged_data['Investing_value'] = merged_data['close'] * merged_data['Qty']
merged_data['Cash'] = initial_balance - merged_data['Amount']
merged_data = merged_data.reset_index()

investing_value = merged_data.groupby('date')['Investing_value'].sum()
cash = merged_data.groupby('date')['Cash'].last()
portfolio_value = investing_value + cash
portfolio_value= portfolio_value.reset_index()
result_df = pd.merge(merged_data, portfolio_value, on = 'date')
result_df = result_df.rename(columns = {0:'portfolio_value'})

result_df.to_csv('data1.csv')












