import datetime

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


class DataProcessing:
    """Provides methods for retrieving daily stock data from Yahoo Finance API.

    Attributes
    ----------
        start_date : str
            Start date of the data.
        end_date : str
            End date of the data.
        csv_file : str
            CSV file containing transaction data.

    Methods
    -------
    fetch_data(proxy=None) -> pd.DataFrame
        Fetches data from the Yahoo API.

    transaction_data() -> pd.DataFrame
        Processes transaction data from a CSV file.

    merge_data(initial_balance=1000000) -> pd.Series
        Merges and processes data from Yahoo API and portfolio transactions.
    """
    
    

    def __init__(self, start_date: str, end_date: str, csv_file: str):
        self.start_date = start_date
        self.end_date = end_date
       
        self.csv_file = csv_file
        self.transaction_df = self.transaction_data()
        
        self.daily_return_df = self.daily_return_data()
        self.portfolio_value = self.merge_data() 
        self.sp500_portfolio = self.calculate_sp500_portfolio_value()
    
        if len(self.portfolio_value) <= 10:
            start_date_time = datetime.strptime(self.start_date, "%Y-%m-%d")
            self.start_date = start_date_time - datetime.timedelta(months = 1)
            self.extended_daily_returns_df = self.daily_return_data()
            self.extended_portfolio_value = self.merge_data()
            self.extended_sp500_portfolio = self.calculate_sp500_portfolio_value()
        

    def daily_return_data(self, proxy=None):
        """Fetches data from Yahoo API.

        Returns
        -------
        `pd.DataFrame`
            A DataFrame with daily stock return data.
        """
        # Download and save the data in a pandas DataFrame:
        data_df = pd.DataFrame()
        num_failures = 0
        for tic in self.ticker_list:
            temp_df = yf.download(
                tic, start=self.start_date, end=self.end_date, proxy=proxy, progress = False
            )
            temp_df["tic"] = tic
            if len(temp_df) > 0:
                # data_df = data_df.append(temp_df)
                data_df = pd.concat([data_df, temp_df], axis=0)
            else:
                num_failures += 1
        if num_failures == len(self.ticker_list):
            raise ValueError("no data is fetched.")
        # reset the index, we want to use numbers as index instead of dates
        data_df = data_df.reset_index()
        try:
            # convert the column names to standardized names
            data_df.columns = [
                "date",
                "open",
                "high",
                "low",
                "close",
                "adjcp",
                "volume",
                "tic",
            ]
            # use adjusted close price instead of close price
            data_df["close"] = data_df["adjcp"]
            # drop the adjusted close price column
            data_df = data_df.drop(labels="adjcp", axis=1)
        except NotImplementedError:
            print("the features are not supported currently")
        # create day of the week column (monday = 0)
        data_df["day"] = data_df["date"].dt.dayofweek
        # convert date to standard string format, easy to filter
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        data_df = data_df.set_index(['date', 'tic'])   

        #print("Shape of DataFrame: ", data_df.shape)
        # print("Display DataFrame: ", data_df.head())

        data_df = data_df.sort_values(by=["date", "tic"])
   

        return data_df
    
    def transaction_data(self):
        """Processes transaction data from a CSV file.

        Returns
        -------
        `pd.DataFrame`
            A DataFrame with transaction data.
        """
        transaction_df = pd.read_csv(self.csv_file)
        transaction_df = transaction_df.iloc[3:]
        date_time_split = transaction_df['Portfolio ID'].str.split('T', expand=True)
        transaction_df['date'] = date_time_split[0]
        transaction_df['time'] = date_time_split[1].str[:8]
        transaction_df['time'] = pd.to_datetime(transaction_df['time'], format='%H:%M:%S').dt.time

        transaction_df = transaction_df.rename(columns={'User Name': 'tic', 'Cash Left': 'Qty', 'Unnamed: 7': 'Amount'})

        transaction_df = transaction_df[['date', 'time', 'tic', 'Qty', 'Amount']]
        transaction_df['Qty'] = pd.to_numeric(transaction_df['Qty'], errors='coerce', downcast='integer')
        transaction_df['Amount'] = pd.to_numeric(transaction_df['Amount'], errors='coerce', downcast='integer')

        transaction_df['Price'] = abs(transaction_df['Amount'] / transaction_df['Qty'])
        transaction_df.loc[transaction_df['Amount'] < 0, 'Qty'] *= -1
        self.ticker_list = transaction_df['tic'].unique().tolist()

        QTY = transaction_df.groupby(['date', 'tic'])['Qty'].sum()
        Amount = transaction_df.groupby(['date', 'tic'])['Amount'].sum()

        transaction_df1 = pd.DataFrame({'Qty_chg': QTY})
        transaction_df1['Amount_chg'] = Amount
        transaction_df1['Qty'] = transaction_df1.groupby(['tic'])['Qty_chg'].cumsum()
        transaction_df1['Amount'] = transaction_df1['Amount_chg'].cumsum()
        return transaction_df1
        
        
    def calculate_sp500_portfolio_value(self, initial_balance = 1000000):
        """Calculates the SP500 portfolio value.

        Returns
        -------
        pd.Series
            SP500 portfolio value over time.
        """
        sp500_data = yf.download("^SPX", self.start_date, self.end_date, progress = False)
        sp500_data = sp500_data.drop(['Open', 'High', 'Low', 'Volume','Adj Close'], axis=1)
        sp500_data['return'] = sp500_data['Close']/sp500_data['Close'].shift(1)-1
        sp500_data['cum'] = (sp500_data['return']+1).cumprod()

        sp500_data['portfolio'] = sp500_data['cum']*initial_balance
        sp500_data['portfolio'].iloc[0] = initial_balance
        return sp500_data['portfolio']


    def merge_data(self, initial_balance = 1000000):
        """Merges and processes data from Yahoo API and portfolio transactions.

        Returns
        -------
        pd.Series
            Portfolio value over time after merging and processing data.
        """
        merged_data = pd.merge(self.daily_return_df, self.transaction_df, left_on=['date', 'tic'], 
                               right_on=['date', 'tic'], how='outer',indicator=True)
        merged_data['Amount'] = merged_data['Amount'].ffill()
        merged_data['Qty'] = merged_data.groupby(['tic'])['Qty'].ffill()
        merged_data = merged_data.drop(['day', 'Qty_chg', 'Amount_chg', '_merge'], axis=1)
        merged_data = merged_data.fillna(0)
        merged_data['Investing_value'] = merged_data['close'] * merged_data['Qty']
        merged_data['Cash'] = initial_balance - merged_data['Amount']
        investing_value = merged_data.groupby('date')['Investing_value'].sum()
        cash = merged_data.groupby('date')['Cash'].last()
        portfolio_value = investing_value + cash
        
        return portfolio_value
    
    def plot(self):

        # Create a line plot to compare our porfolio and sp500 portfolio
        plt.figure(figsize=(12, 5))
        plt.plot(pd.to_datetime(self.portfolio_value.index), self.portfolio_value, label='Portfolio value', color='blue')
        
        # Plot the value of s&p 500 portfolio 
        plt.plot(pd.to_datetime(self.portfolio_value.index), self.sp500_portfolio, label='S&P 500 Value', color='red')
        plt.title("Portfolio Value Comparison")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value/$")
        plt.legend()

        plt.xticks(rotation=45)
        plt.show()
        
        
    


# Example usage:
if __name__ == "__main__":
    
    Start_Date = '2023-05-22'
    End_Date = '2023-11-03'
    csv_file = 'sample.csv'
    processor = DataProcessing(Start_Date, End_Date, csv_file)
   


