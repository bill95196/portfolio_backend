#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 22:48:56 2023

@author: huangqini
"""

import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
os.chdir('/Users/huangqini/Desktop/TradeUp')
df = pd.read_csv('data1.csv', index_col=[0])
# avoid multiple transactions for one stock within one day
df1 = df.groupby(['date','tic']).agg({'close':'mean','Qty':'sum','Amount':'sum','Investing_value':'sum','Cash':'sum','portfolio_value':'sum'}).reset_index()

# Calculate total investing value per day and the weight of each stock
daily_total_investing = pd.DataFrame(df1.groupby('date')['Investing_value'].sum()).reset_index()
daily_total_investing = daily_total_investing.rename(columns = {'Investing_value':'daily_total_investing'})
df1 = pd.merge(df1, daily_total_investing, on = 'date')
df1['weight'] = df1['Investing_value'] / df1['daily_total_investing']

# Calculate daily log return
df1 = df1.fillna(0)
df1['log_price_1'] = np.log(df1.groupby('tic').close.shift(1)) 
df1['log_price'] = np.log(df1.close)
df1['log_return'] = df1['log_price_1'] - df1['log_price']
df1['log_return'] = df1['log_return'].shift(1)

# Calculate Volatility 
port_value = pd.DataFrame(df1.groupby('date')['portfolio_value'].mean())
port_value['port_return'] = port_value['portfolio_value'].pct_change()
port_value = port_value.reset_index()
port_value['port_vol'] = port_value['port_return'].expanding().std()
df1 = pd.merge(df1, port_value[['date','port_return','port_vol']], on = 'date')


# Create an empty column for each stock's daily rolling return volatility
rolling_window = 21  # Set the desired rolling window size, use 21 days for a month

for ticker in df['tic'].unique():
    mask = (df['tic'] == ticker) & (df1['Investing_value'] != 0)
    df1.loc[mask, f'{ticker}_volatility'] = df1.loc[mask, 'log_return'].rolling(window=rolling_window).std()

# Create a new column for each stock's volatility in the original DataFrame
df1['stock_vol'] = df1[['date'] + [f'{ticker}_volatility' for ticker in df['tic'].unique()]].sum(axis=1)
df1.drop(columns=[f'{ticker}_volatility' for ticker in df['tic'].unique()], inplace=True)

# Calculate risk contribution
df1['rho'] = df1['stock_vol'] / df1['port_vol']
df1['MC'] = df1['stock_vol'] * df1['rho']
df1['PC'] = (df1['weight'] * df1['MC'])/ df1['port_vol']

# Plot, Final Dataset is df1, and column PC is the final contribution
df1['date'] = pd.to_datetime(df1['date'])
# Plot PC changes over time grouped by stock ticker
plt.figure(figsize=(12, 6))
sns.lineplot(x='date', y='PC', hue='tic', data=df1)
plt.title('PC Changes Over Time Grouped by Stock Ticker')
plt.xlabel('Date')
plt.ylabel('PC')



