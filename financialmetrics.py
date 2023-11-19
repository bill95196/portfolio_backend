
import pandas as pd
import numpy as np
import math

import statsmodels.api as sm

from dataprocessing import DataProcessing


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

    def __init__(self, portfolio_value, benchmark_portfolio_value, risk_free_rate=0.02,
                 extended_portfolio_value = None, extended_benchmark_portoflio_value = None):
        self.benchmark_portfolio_value = benchmark_portfolio_value
        self.portfolio_value = portfolio_value
        self.risk_free_rate = risk_free_rate
        self.annualized_return_ = self.annualized_return()
        self.beta_ = self.beta()
        self.max_drawdown_ = self.max_drawdown()
        self.std_ = self.standard_deviation()
        self.extended_portfolio_value = extended_portfolio_value
        self.extended_benchmark_portoflio_value = extended_benchmark_portoflio_value
        #extened data frame and optionally imported it into the financial metrics class, only
        #use these extened portfolios when len(portoflio) <= 10
        
    def annualized_return(self):
        """
        Calculate annualized return of the portfolio.

        Returns:
            float: annualized return
        """
        daily_returns = (self.portfolio_value / self.portfolio_value.shift(1) - 1)[1:]
        if len(daily_returns) <= 10:
            return (self.extended_portfolio_value / self.extended_portfolio_value.shift(1) - 1)[1:].mean()*252
        else:
            return daily_returns.mean()*252
    
    def standard_deviation(self):
        """
        Calculate the standard deviation of portfolio returns.

        Returns:
            float: Standard Deviation.
        """
        daily_returns = (self.portfolio_value / self.portfolio_value.shift(1) - 1)[1:]
        return np.std(daily_returns)*math.sqrt(252)
    
    
    
    def beta(self):
        """
        Calculate beta of the portfolio.

        Returns:
            float: beta
        """
        daily_returns = (self.portfolio_value / self.portfolio_value.shift(1) - 1)[1:]
        market_daily_returns = (self.benchmark_portfolio_value / self.benchmark_portfolio_value.shift(1) - 1)[1:]
   
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
    
    def holding_period_return(self):
        """
        Calculate holding period return

        Returns:
            float: holding period return
        """
        return self.portfolio_value.iloc[-1] / self.portfolio_value.iloc[0] - 1
        
        

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
        market_daily_returns = (self.benchmark_portfolio_value / self.benchmark_portfolio_value.shift(1) - 1)[1:]

        tracking_error = np.std(daily_returns.values - market_daily_returns.values) * math.sqrt(252)
        return tracking_error

    def VaR(self, alpha=0.05, days = 10):
        """
        Calculate the Value at Risk (VaR) at the specified confidence level.

        Parameters:
            alpha (float, optional): Confidence level.

        Returns:
            float: VaR value.
        """
        import scipy.stats as st
        z_score = st.norm.ppf(1-alpha)
        
        var_ = self.portfolio_value.iloc[-1] * self.std_* math.sqrt(days/252) * z_score

        return var_

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
    
    def CAGR(self):
        """
        Calculate the Compound Annual Growth Rate (CAGR).

        Returns:
            float: CAGR value.
        """
        if len(self.portfolio_value) <= 10:
            return (self.extended_portfolio_value.iloc[-1] / self.extended_portfolio_value.iloc[0])**(252 / (len(self.extended_portfolio_value)-1))-1
        else:
            return (self.portfolio_value.iloc[-1] / self.portfolio_value.iloc[0])**(252 / (len(self.portfolio_value)-1))-1




    def alpha(self):
        """
        Calculate the alpha of the portfolio.

        Returns:
            float: Alpha.
        """
 
        benchmark_daily_returns = (self.benchmark_portfolio_value / self.benchmark_portfolio_value.shift(1) - 1)[1:]
        benchmark_ann_return = benchmark_daily_returns.mean()*252
        
        return self.annualized_return_ - benchmark_ann_return

    def jensens_alpha(self):
        """
        Calculate Jensen’s Alpha of the portfolio.

        Returns:
            float: Jensen’s Alpha.
        """
        benchmark_daily_returns = (self.benchmark_portfolio_value / self.benchmark_portfolio_value.shift(1) - 1)[1:]
        benchmark_ann_return = benchmark_daily_returns.mean()*252
        
        alpha = self.annualized_return_ - (self.risk_free_rate + self.beta_ * (benchmark_ann_return - self.risk_free_rate))
        return alpha

    def correlation_with_sp500(self):
        """
        Calculate the correlation of portfolio returns with S&P 500 returns.

        Returns:
            float: Correlation with S&P 500.
        """
        daily_returns = (self.portfolio_value / self.portfolio_value.shift(1) - 1)[1:]
        benchmark_daily_returns = (self.benchmark_portfolio_value / self.benchmark_portfolio_value.shift(1) - 1)[1:]
        
        return np.corrcoef(daily_returns, benchmark_daily_returns)[0][1]


        
    def sharpe_ratio(self):
        """
        Calculate the Sharpe Ratio.

        Returns:
            float: Sharpe Ratio.
        """

        return (self.annualized_return_ - self.risk_free_rate) / self.std_

    def r_squared(self):
        """
        Calculate R-Squared of the portfolio.

        Returns:
            float: R-Squared.
        """
        
     
        daily_returns = (self.portfolio_value / self.portfolio_value.shift(1) - 1)[1:]
        benchmark_daily_returns = (self.benchmark_portfolio_value / self.benchmark_portfolio_value.shift(1) - 1)[1:]
        res = sm.OLS(list(daily_returns), list(benchmark_daily_returns)).fit()
        return res.rsquared



