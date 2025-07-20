"""
Piotroski F-Score Agent
피오트로스키의 F-Score 전략을 구현한 에이전트
"""

from .base_agent import BaseAgent
import pandas as pd
import numpy as np


class PiotroskiAgent(BaseAgent):
    """
    피오트로스키 F-Score 전략 에이전트
    
    F-Score 구성요소:
    1. Profitability (3점)
    2. Leverage, Liquidity, Source of Funds (3점)
    3. Operating Efficiency (3점)
    """
    
    def __init__(self):
        super().__init__(
            name="Piotroski F-Score Agent",
            description="피오트로스키의 F-Score 전략"
        )
        self.parameters = {
            'min_f_score': 7,  # 최소 F-Score
            'min_market_cap': 1000000000,  # 최소 시가총액 (10억원)
        }
    
    def calculate_f_score(self, stock_data: pd.Series) -> int:
        """
        F-Score 계산 (0-9점)
        
        Args:
            stock_data: 개별 종목의 데이터
            
        Returns:
            F-Score (0-9)
        """
        f_score = 0
        
        # 1. Profitability (3점)
        # ROA > 0
        if stock_data.get('roa', 0) > 0:
            f_score += 1
        
        # CFO > 0
        if stock_data.get('operating_cash_flow', 0) > 0:
            f_score += 1
        
        # ROA 증가
        if stock_data.get('roa_change', 0) > 0:
            f_score += 1
        
        # 2. Leverage, Liquidity, Source of Funds (3점)
        # 부채비율 감소
        if stock_data.get('debt_ratio_change', 0) < 0:
            f_score += 1
        
        # 유동비율 증가
        if stock_data.get('current_ratio_change', 0) > 0:
            f_score += 1
        
        # 신주 발행 없음
        if stock_data.get('shares_outstanding_change', 0) <= 0:
            f_score += 1
        
        # 3. Operating Efficiency (3점)
        # 총자산회전율 증가
        if stock_data.get('asset_turnover_change', 0) > 0:
            f_score += 1
        
        # 마진 증가
        if stock_data.get('gross_margin_change', 0) > 0:
            f_score += 1
        
        # 매출 증가
        if stock_data.get('revenue_change', 0) > 0:
            f_score += 1
        
        return f_score
    
    def screen_stocks(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        F-Score 기준으로 스크리닝
        
        Args:
            data: OHLCV 및 재무 데이터
            
        Returns:
            스크리닝된 종목들의 DataFrame
        """
        screened = data.copy()
        
        # F-Score 계산
        f_scores = []
        for _, row in screened.iterrows():
            f_score = self.calculate_f_score(row)
            f_scores.append(f_score)
        
        screened['f_score'] = f_scores
        
        # 기본 스크리닝 조건
        conditions = [
            screened['f_score'] >= self.parameters['min_f_score'],
            screened['market_cap'] >= self.parameters['min_market_cap'],
            screened['market_cap'] > 0,
            screened['f_score'].notna(),
        ]
        
        # 모든 조건을 만족하는 종목 필터링
        mask = np.logical_and.reduce(conditions)
        screened = screened[mask]
        
        return screened
    
    def calculate_score(self, stock_data: pd.Series) -> float:
        """
        F-Score 기반 점수 계산
        
        Args:
            stock_data: 개별 종목의 데이터
            
        Returns:
            계산된 점수 (높을수록 좋음)
        """
        # F-Score를 0-1 범위로 정규화
        f_score_normalized = stock_data['f_score'] / 9.0
        
        # 추가 보너스 점수 (ROA, ROE 등)
        roa_bonus = min(0.2, stock_data.get('roa', 0) / 0.20)  # 20% ROA를 최대 보너스
        roe_bonus = min(0.2, stock_data.get('roe', 0) / 0.25)   # 25% ROE를 최대 보너스
        
        # 종합 점수
        total_score = f_score_normalized + roa_bonus + roe_bonus
        
        return min(1.0, total_score)  # 최대 1.0으로 제한 