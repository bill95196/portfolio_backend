
import pandas as pd
import numpy as np
import math

class FinancialMetrics:
    """Provides methods for calculating financial metrics based on portfolio and benchmark data.

    Attributes
    ----------
       portfolio_value (pd.Series): 
           Portfolio value over time.
       sp500_portfolio_value (pd.Series): 
           S&P 500 portfolio value over time.
       risk_free_rate (float, optional): 
           Risk-free interest rate, e.g., Treasury bill rate.
    """

    def __init__(self, portfolio_value, sp500_portfolio_value, risk_free_rate=0.02):
        self.sp500_portfolio_value = sp500_portfolio_value
        self.portfolio_value = portfolio_value
        self.risk_free_rate = risk_free_rate
        self.annualized_return_ = self.annualized_return()
        self.beta_ = self.beta()
        self.max_drawdown_ = self.max_drawdown()
        
    def annualized_return(self):
        """
        Calculate annualized return of the portfolio.

        Returns:
            float: annualized return
        """
        
        return (self.portfolio_value.iloc[-1] / self.portfolio_value.iloc[0])**(252 / (len(self.portfolio_value)-1))-1 
    
    def beta(self):
        """
        Calculate beta of the portfolio.

        Returns:
            float: beta
        """
        daily_returns = (self.portfolio_value / self.portfolio_value.shift(1) - 1)[1:]
        market_daily_returns = (self.sp500_portfolio_value / self.sp500_portfolio_value.shift(1) - 1)[1:]
   
        beta = np.cov(daily_returns, market_daily_returns)[0][1] / np.var(market_daily_returns)
        return beta
        

    def max_drawdown(self):
        """
        Calculate the maximum drawdown of the portfolio.

        Returns:
            float: Maximum drawdown as a decimal.
        """
        
        cumulative_returns = (self.portfolio_value / self.portfolio_value.iloc[0])
        drawdown = (cumulative_returns - cumulative_returns.cummax())
        max_drawdown = -drawdown.min()
        return max_drawdown

    def sortino_ratio(self):
        """
        Calculate the Sortino Ratio, a measure of risk-adjusted return.

        Returns:
            float: Sortino Ratio.
        """
        daily_return = (self.portfolio_value / self.portfolio_value.shift(1) - 1)
        
        excess_returns = daily_return - (self.risk_free_rate)**(1/252)
        downside_returns = excess_returns[excess_returns < 0]
        
        downside_std = downside_returns.std() * math.sqrt(252)

       
        if downside_std != 0:
            sortino_ratio = (self.annualized_return_ - self.risk_free_rate)/ (downside_std)
        else:
            sortino_ratio = np.nan

        return sortino_ratio

    def treynor_ratio(self):
        """
        Calculate the Treynor Ratio, a measure of risk-adjusted return.

        Returns:
            float: Treynor Ratio.
        """

        
        treynor_ratio = (self.annualized_return_ - self.risk_free_rate) / self.beta_
        return treynor_ratio

    def calmar_ratio(self):
        
        """
        Calculate the Calmar Ratio, a measure of risk-adjusted return.

        Returns:
            float: Calmar Ratio.
        """
    
        return self.annualized_return_ / self.max_drawdown_
   

    def tracking_error(self):
        """
        Calculate the tracking error, a measure of portfolio performance relative to a benchmark.

        Returns:
            float: Tracking Error.
        """
        daily_returns = (self.portfolio_value / self.portfolio_value.shift(1) - 1)[1:]
        market_daily_returns = (self.sp500_portfolio_value / self.sp500_portfolio_value.shift(1) - 1)[1:]

        tracking_error = np.std(daily_returns.values - market_daily_returns.values)
        return tracking_error

    def VaR(self, alpha=0.05):
        """
        Calculate the Value at Risk (VaR) at the specified confidence level.

        Parameters:
            alpha (float, optional): Confidence level.

        Returns:
            float: VaR value.
        """

        return None

    def best_return(self):
        """
        Calculate the best return in the dataset.

        Returns:
            float: Best return.
        """
        cumulative_returns = self.portfolio_value / self.portfolio_value.iloc[0] -1
        best_return = cumulative_returns.max()
        return best_return

    def worst_return(self):
        """
        Calculate the worst return in the dataset.

        Returns:
            float: Worst return.
        """
        cumulative_returns = self.portfolio_value / self.portfolio_value.iloc[0] -1 
        worst_return = cumulative_returns.min()
        return worst_return

    def cash_rate(self):
        """
        Get the specified risk-free rate.

        Returns:
            float: Risk-free interest rate.
        """
        return None


