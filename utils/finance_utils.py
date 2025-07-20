"""
Finance Utilities
재무 지표 계산 및 분석 유틸리티
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats


class FinanceUtils:
    """
    재무 지표 계산 및 분석 유틸리티 클래스
    """
    
    @staticmethod
    def calculate_pe_ratio(price: float, earnings_per_share: float) -> float:
        """
        P/E 비율 계산
        
        Args:
            price: 주가
            earnings_per_share: 주당순이익
            
        Returns:
            P/E 비율
        """
        if earnings_per_share <= 0:
            return float('inf')
        return price / earnings_per_share
    
    @staticmethod
    def calculate_pb_ratio(price: float, book_value_per_share: float) -> float:
        """
        P/B 비율 계산
        
        Args:
            price: 주가
            book_value_per_share: 주당순자산
            
        Returns:
            P/B 비율
        """
        if book_value_per_share <= 0:
            return float('inf')
        return price / book_value_per_share
    
    @staticmethod
    def calculate_roe(net_income: float, total_equity: float) -> float:
        """
        ROE (Return on Equity) 계산
        
        Args:
            net_income: 순이익
            total_equity: 자기자본
            
        Returns:
            ROE
        """
        if total_equity <= 0:
            return 0.0
        return net_income / total_equity
    
    @staticmethod
    def calculate_roa(net_income: float, total_assets: float) -> float:
        """
        ROA (Return on Assets) 계산
        
        Args:
            net_income: 순이익
            total_assets: 총자산
            
        Returns:
            ROA
        """
        if total_assets <= 0:
            return 0.0
        return net_income / total_assets
    
    @staticmethod
    def calculate_roic(nopat: float, invested_capital: float) -> float:
        """
        ROIC (Return on Invested Capital) 계산
        
        Args:
            nopat: 세후영업이익
            invested_capital: 투하자본
            
        Returns:
            ROIC
        """
        if invested_capital <= 0:
            return 0.0
        return nopat / invested_capital
    
    @staticmethod
    def calculate_debt_ratio(total_debt: float, total_assets: float) -> float:
        """
        부채비율 계산
        
        Args:
            total_debt: 총부채
            total_assets: 총자산
            
        Returns:
            부채비율
        """
        if total_assets <= 0:
            return float('inf')
        return total_debt / total_assets
    
    @staticmethod
    def calculate_current_ratio(current_assets: float, current_liabilities: float) -> float:
        """
        유동비율 계산
        
        Args:
            current_assets: 유동자산
            current_liabilities: 유동부채
            
        Returns:
            유동비율
        """
        if current_liabilities <= 0:
            return float('inf')
        return current_assets / current_liabilities
    
    @staticmethod
    def calculate_quick_ratio(quick_assets: float, current_liabilities: float) -> float:
        """
        당좌비율 계산
        
        Args:
            quick_assets: 당좌자산 (유동자산 - 재고자산)
            current_liabilities: 유동부채
            
        Returns:
            당좌비율
        """
        if current_liabilities <= 0:
            return float('inf')
        return quick_assets / current_liabilities
    
    @staticmethod
    def calculate_asset_turnover(revenue: float, total_assets: float) -> float:
        """
        총자산회전율 계산
        
        Args:
            revenue: 매출액
            total_assets: 총자산
            
        Returns:
            총자산회전율
        """
        if total_assets <= 0:
            return 0.0
        return revenue / total_assets
    
    @staticmethod
    def calculate_inventory_turnover(cost_of_goods_sold: float, inventory: float) -> float:
        """
        재고자산회전율 계산
        
        Args:
            cost_of_goods_sold: 매출원가
            inventory: 재고자산
            
        Returns:
            재고자산회전율
        """
        if inventory <= 0:
            return 0.0
        return cost_of_goods_sold / inventory
    
    @staticmethod
    def calculate_gross_margin(revenue: float, cost_of_goods_sold: float) -> float:
        """
        매출총이익률 계산
        
        Args:
            revenue: 매출액
            cost_of_goods_sold: 매출원가
            
        Returns:
            매출총이익률
        """
        if revenue <= 0:
            return 0.0
        return (revenue - cost_of_goods_sold) / revenue
    
    @staticmethod
    def calculate_net_margin(net_income: float, revenue: float) -> float:
        """
        순이익률 계산
        
        Args:
            net_income: 순이익
            revenue: 매출액
            
        Returns:
            순이익률
        """
        if revenue <= 0:
            return 0.0
        return net_income / revenue
    
    @staticmethod
    def calculate_ev_ebit(enterprise_value: float, ebit: float) -> float:
        """
        EV/EBIT 계산
        
        Args:
            enterprise_value: 기업가치
            ebit: 영업이익
            
        Returns:
            EV/EBIT
        """
        if ebit <= 0:
            return float('inf')
        return enterprise_value / ebit
    
    @staticmethod
    def calculate_enterprise_value(market_cap: float, total_debt: float, cash: float) -> float:
        """
        기업가치 계산
        
        Args:
            market_cap: 시가총액
            total_debt: 총부채
            cash: 현금 및 현금성자산
            
        Returns:
            기업가치
        """
        return market_cap + total_debt - cash
    
    @staticmethod
    def calculate_beta(returns: pd.Series, market_returns: pd.Series) -> float:
        """
        베타 계산
        
        Args:
            returns: 종목 수익률
            market_returns: 시장 수익률
            
        Returns:
            베타
        """
        if len(returns) != len(market_returns):
            return 1.0
        
        # 공분산과 분산 계산
        covariance = np.cov(returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        
        if market_variance == 0:
            return 1.0
        
        return covariance / market_variance
    
    @staticmethod
    def calculate_alpha(returns: pd.Series, market_returns: pd.Series, 
                       risk_free_rate: float = 0.02) -> float:
        """
        알파 계산
        
        Args:
            returns: 종목 수익률
            market_returns: 시장 수익률
            risk_free_rate: 무위험 수익률
            
        Returns:
            알파
        """
        beta = FinanceUtils.calculate_beta(returns, market_returns)
        
        portfolio_mean = returns.mean() * 252  # 연간화
        market_mean = market_returns.mean() * 252  # 연간화
        
        alpha = portfolio_mean - (risk_free_rate + beta * (market_mean - risk_free_rate))
        return alpha
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        샤프 비율 계산
        
        Args:
            returns: 수익률
            risk_free_rate: 무위험 수익률
            
        Returns:
            샤프 비율
        """
        excess_returns = returns - risk_free_rate / 252  # 일간 무위험 수익률
        volatility = returns.std() * np.sqrt(252)  # 연간 변동성
        
        if volatility == 0:
            return 0.0
        
        return (excess_returns.mean() * 252) / volatility
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        소르티노 비율 계산
        
        Args:
            returns: 수익률
            risk_free_rate: 무위험 수익률
            
        Returns:
            소르티노 비율
        """
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_deviation = downside_returns.std() * np.sqrt(252)
        
        if downside_deviation == 0:
            return 0.0
        
        return (excess_returns.mean() * 252) / downside_deviation
    
    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> float:
        """
        최대 낙폭 계산
        
        Args:
            returns: 수익률
            
        Returns:
            최대 낙폭
        """
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """
        VaR (Value at Risk) 계산
        
        Args:
            returns: 수익률
            confidence_level: 신뢰수준
            
        Returns:
            VaR
        """
        return np.percentile(returns, confidence_level * 100)
    
    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """
        CVaR (Conditional Value at Risk) 계산
        
        Args:
            returns: 수익률
            confidence_level: 신뢰수준
            
        Returns:
            CVaR
        """
        var = FinanceUtils.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    @staticmethod
    def calculate_momentum(prices: pd.Series, period: int = 252) -> float:
        """
        모멘텀 계산
        
        Args:
            prices: 가격 시계열
            period: 계산 기간
            
        Returns:
            모멘텀
        """
        if len(prices) < period:
            return 0.0
        
        current_price = prices.iloc[-1]
        past_price = prices.iloc[-period]
        
        return (current_price - past_price) / past_price
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, window: int = 252) -> float:
        """
        변동성 계산
        
        Args:
            returns: 수익률
            window: 계산 윈도우
            
        Returns:
            변동성 (연간화)
        """
        if len(returns) < window:
            return returns.std() * np.sqrt(252)
        
        rolling_vol = returns.rolling(window=window).std()
        return rolling_vol.iloc[-1] * np.sqrt(252)
    
    @staticmethod
    def calculate_correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        상관관계 행렬 계산
        
        Args:
            returns_df: 수익률 DataFrame (종목별 컬럼)
            
        Returns:
            상관관계 행렬
        """
        return returns_df.corr()
    
    @staticmethod
    def calculate_portfolio_volatility(weights: np.ndarray, 
                                     covariance_matrix: pd.DataFrame) -> float:
        """
        포트폴리오 변동성 계산
        
        Args:
            weights: 가중치 벡터
            covariance_matrix: 공분산 행렬
            
        Returns:
            포트폴리오 변동성
        """
        portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
        return np.sqrt(portfolio_variance)
    
    @staticmethod
    def calculate_portfolio_return(weights: np.ndarray, 
                                 expected_returns: pd.Series) -> float:
        """
        포트폴리오 기대 수익률 계산
        
        Args:
            weights: 가중치 벡터
            expected_returns: 기대 수익률
            
        Returns:
            포트폴리오 기대 수익률
        """
        return np.dot(weights, expected_returns) 