"""
Richard Driehaus Growth Agent
리처드 드리하우스의 성장 투자 전략을 구현한 에이전트
"""

from .base_agent import BaseAgent
import pandas as pd
import numpy as np


class DriehausAgent(BaseAgent):
    """
    리처드 드리하우스의 성장 투자 전략 에이전트
    
    주요 특징:
    1. 고성장 기업 선별
    2. 모멘텀 활용
    3. 시장 선도 기업 선호
    """
    
    def __init__(self):
        super().__init__(
            name="Driehaus Growth Agent",
            description="리처드 드리하우스의 성장 투자 전략"
        )
        self.parameters = {
            'min_revenue_growth': 0.15,  # 최소 매출 성장률 15%
            'min_earnings_growth': 0.20,  # 최소 이익 성장률 20%
            'min_market_cap': 50000000000,  # 최소 시가총액 (500억원)
            'max_pe': 50,  # 최대 P/E 50배
            'momentum_period': 60,  # 모멘텀 계산 기간
        }
    
    def calculate_growth_score(self, stock_data: pd.Series) -> float:
        """
        성장 점수 계산
        
        Args:
            stock_data: 개별 종목의 데이터
            
        Returns:
            성장 점수 (0-1)
        """
        revenue_growth = stock_data.get('revenue_growth', 0)
        earnings_growth = stock_data.get('earnings_growth', 0)
        roe = stock_data.get('roe', 0)
        
        # 매출 성장 점수
        revenue_score = min(1.0, revenue_growth / 0.50)  # 50% 성장을 최대값으로
        
        # 이익 성장 점수
        earnings_score = min(1.0, earnings_growth / 0.50)  # 50% 성장을 최대값으로
        
        # ROE 점수
        roe_score = min(1.0, roe / 0.30)  # 30% ROE를 최대값으로
        
        # 종합 성장 점수
        growth_score = (0.4 * revenue_score + 0.4 * earnings_score + 0.2 * roe_score)
        
        return growth_score
    
    def calculate_momentum_score(self, stock_data: pd.Series) -> float:
        """
        모멘텀 점수 계산
        
        Args:
            stock_data: 개별 종목의 데이터
            
        Returns:
            모멘텀 점수 (0-1)
        """
        price_momentum = stock_data.get('price_momentum', 0)
        volume_momentum = stock_data.get('volume_momentum', 0)
        
        # 가격 모멘텀 점수
        price_score = min(1.0, max(0, price_momentum / 0.30))  # 30% 상승을 최대값으로
        
        # 거래량 모멘텀 점수
        volume_score = min(1.0, max(0, volume_momentum / 0.50))  # 50% 증가를 최대값으로
        
        # 종합 모멘텀 점수
        momentum_score = (0.7 * price_score + 0.3 * volume_score)
        
        return momentum_score
    
    def screen_stocks(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        성장 투자 기준으로 스크리닝
        
        Args:
            data: OHLCV 및 재무 데이터
            
        Returns:
            스크리닝된 종목들의 DataFrame
        """
        screened = data.copy()
        
        # 성장 점수와 모멘텀 점수 계산
        growth_scores = []
        momentum_scores = []
        
        for _, row in screened.iterrows():
            growth_score = self.calculate_growth_score(row)
            momentum_score = self.calculate_momentum_score(row)
            growth_scores.append(growth_score)
            momentum_scores.append(momentum_score)
        
        screened['growth_score'] = growth_scores
        screened['momentum_score'] = momentum_scores
        
        # 기본 스크리닝 조건
        conditions = [
            screened['market_cap'] >= self.parameters['min_market_cap'],
            screened['revenue_growth'] >= self.parameters['min_revenue_growth'],
            screened['earnings_growth'] >= self.parameters['min_earnings_growth'],
            screened['pe_ratio'] <= self.parameters['max_pe'],
            screened['pe_ratio'] > 0,  # 음수 P/E 제외
            screened['growth_score'].notna(),
            screened['momentum_score'].notna(),
        ]
        
        # 모든 조건을 만족하는 종목 필터링
        mask = np.logical_and.reduce(conditions)
        screened = screened[mask]
        
        return screened
    
    def calculate_score(self, stock_data: pd.Series) -> float:
        """
        드리하우스 점수 계산
        성장 점수와 모멘텀 점수의 조합
        
        Args:
            stock_data: 개별 종목의 데이터
            
        Returns:
            계산된 점수 (높을수록 좋음)
        """
        growth_score = stock_data['growth_score']
        momentum_score = stock_data['momentum_score']
        
        # 성장과 모멘텀의 가중 평균 (성장에 더 높은 가중치)
        total_score = (0.6 * growth_score + 0.4 * momentum_score)
        
        return total_score 