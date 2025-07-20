"""
Joel Greenblatt Magic Formula Agent
조엘 그린블랫의 Magic Formula 전략을 구현한 에이전트
"""

from .base_agent import BaseAgent
import pandas as pd
import numpy as np


class GreenblattAgent(BaseAgent):
    """
    조엘 그린블랫의 Magic Formula 전략 에이전트
    
    Magic Formula 구성요소:
    1. ROIC (Return on Invested Capital) - 높을수록 좋음
    2. EV/EBIT (Enterprise Value to EBIT) - 낮을수록 좋음
    """
    
    def __init__(self):
        super().__init__(
            name="Greenblatt Magic Formula Agent",
            description="조엘 그린블랫의 Magic Formula 전략"
        )
        self.parameters = {
            'min_market_cap': 50000000000,  # 최소 시가총액 (500억원)
            'min_roic': 0.05,  # 최소 ROIC 5%
            'max_ev_ebit': 50,  # 최대 EV/EBIT 50배
        }
    
    def calculate_roic(self, stock_data: pd.Series) -> float:
        """
        ROIC (Return on Invested Capital) 계산
        
        Args:
            stock_data: 개별 종목의 데이터
            
        Returns:
            ROIC 값
        """
        # ROIC = NOPAT / Invested Capital
        # NOPAT = EBIT * (1 - Tax Rate)
        # Invested Capital = Total Assets - Current Liabilities - Cash
        
        ebit = stock_data.get('ebit', 0)
        tax_rate = stock_data.get('effective_tax_rate', 0.25)  # 기본 25%
        total_assets = stock_data.get('total_assets', 0)
        current_liabilities = stock_data.get('current_liabilities', 0)
        cash = stock_data.get('cash', 0)
        
        if total_assets - current_liabilities - cash <= 0:
            return 0
        
        nopat = ebit * (1 - tax_rate)
        invested_capital = total_assets - current_liabilities - cash
        
        return nopat / invested_capital if invested_capital > 0 else 0
    
    def calculate_ev_ebit(self, stock_data: pd.Series) -> float:
        """
        EV/EBIT 계산
        
        Args:
            stock_data: 개별 종목의 데이터
            
        Returns:
            EV/EBIT 값
        """
        market_cap = stock_data.get('market_cap', 0)
        total_debt = stock_data.get('total_debt', 0)
        cash = stock_data.get('cash', 0)
        ebit = stock_data.get('ebit', 0)
        
        if ebit <= 0:
            return float('inf')
        
        enterprise_value = market_cap + total_debt - cash
        return enterprise_value / ebit
    
    def screen_stocks(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Magic Formula 기준으로 스크리닝
        
        Args:
            data: OHLCV 및 재무 데이터
            
        Returns:
            스크리닝된 종목들의 DataFrame
        """
        screened = data.copy()
        
        # ROIC와 EV/EBIT 계산
        roic_values = []
        ev_ebit_values = []
        
        for _, row in screened.iterrows():
            roic = self.calculate_roic(row)
            ev_ebit = self.calculate_ev_ebit(row)
            roic_values.append(roic)
            ev_ebit_values.append(ev_ebit)
        
        screened['roic'] = roic_values
        screened['ev_ebit'] = ev_ebit_values
        
        # 기본 스크리닝 조건
        conditions = [
            screened['market_cap'] >= self.parameters['min_market_cap'],
            screened['roic'] >= self.parameters['min_roic'],
            screened['ev_ebit'] <= self.parameters['max_ev_ebit'],
            screened['ev_ebit'] > 0,  # 음수 EV/EBIT 제외
            screened['roic'].notna(),
            screened['ev_ebit'].notna(),
        ]
        
        # 모든 조건을 만족하는 종목 필터링
        mask = np.logical_and.reduce(conditions)
        screened = screened[mask]
        
        return screened
    
    def calculate_score(self, stock_data: pd.Series) -> float:
        """
        Magic Formula 점수 계산
        ROIC 순위와 EV/EBIT 순위의 합계 (낮을수록 좋음)
        
        Args:
            stock_data: 개별 종목의 데이터
            
        Returns:
            계산된 점수 (낮을수록 좋음, 0-1 범위로 정규화)
        """
        # ROIC 점수 (높을수록 좋음)
        roic_score = min(1.0, stock_data['roic'] / 0.30)  # 30% ROIC를 최대값으로 정규화
        
        # EV/EBIT 점수 (낮을수록 좋음)
        ev_ebit_score = max(0, 1 - (stock_data['ev_ebit'] / 50))  # 50배를 최대값으로 정규화
        
        # 종합 점수 (가중 평균)
        total_score = (0.5 * roic_score + 0.5 * ev_ebit_score)
        
        return total_score 