"""
Benjamin Graham Value Investing Agent
벤자민 그레이엄의 가치투자 전략을 구현한 에이전트
"""

from .base_agent import BaseAgent
import pandas as pd
import numpy as np


class GrahamAgent(BaseAgent):
    """
    벤자민 그레이엄의 가치투자 전략 에이전트
    
    주요 스크리닝 조건:
    1. P/E < 15
    2. P/B < 1.5
    3. Current Ratio > 2
    4. Debt/Equity < 0.5
    5. ROE > 10%
    """
    
    def __init__(self):
        super().__init__(
            name="Graham Value Agent",
            description="벤자민 그레이엄의 가치투자 전략"
        )
        self.parameters = {
            'max_pe': 15,
            'max_pb': 1.5,
            'min_current_ratio': 2.0,
            'max_debt_equity': 0.5,
            'min_roe': 0.10
        }
    
    def screen_stocks(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        그레이엄의 가치투자 기준으로 스크리닝
        
        Args:
            data: OHLCV 및 재무 데이터
            
        Returns:
            스크리닝된 종목들의 DataFrame
        """
        screened = data.copy()
        
        # 기본 스크리닝 조건
        conditions = [
            screened['pe_ratio'] < self.parameters['max_pe'],
            screened['pb_ratio'] < self.parameters['max_pb'],
            screened['current_ratio'] > self.parameters['min_current_ratio'],
            screened['debt_equity_ratio'] < self.parameters['max_debt_equity'],
            screened['roe'] > self.parameters['min_roe'],
            screened['pe_ratio'] > 0,  # 음수 P/E 제외
            screened['pb_ratio'] > 0,  # 음수 P/B 제외
            screened['current_ratio'] > 0,  # 음수 Current Ratio 제외
            screened['roe'].notna(),  # ROE 데이터 있는 종목만
        ]
        
        # 모든 조건을 만족하는 종목 필터링
        mask = np.logical_and.reduce(conditions)
        screened = screened[mask]
        
        return screened
    
    def calculate_score(self, stock_data: pd.Series) -> float:
        """
        그레이엄 점수 계산
        낮은 P/E, P/B일수록 높은 점수
        
        Args:
            stock_data: 개별 종목의 데이터
            
        Returns:
            계산된 점수 (높을수록 좋음)
        """
        # P/E 점수 (낮을수록 높은 점수)
        pe_score = max(0, 1 / stock_data['pe_ratio']) if stock_data['pe_ratio'] > 0 else 0
        
        # P/B 점수 (낮을수록 높은 점수)
        pb_score = max(0, 1 / stock_data['pb_ratio']) if stock_data['pb_ratio'] > 0 else 0
        
        # ROE 점수 (높을수록 높은 점수)
        roe_score = min(1.0, stock_data['roe'] / 0.20)  # 20% ROE를 최대값으로 정규화
        
        # Current Ratio 점수 (높을수록 높은 점수, 단 너무 높으면 불리)
        cr_score = min(1.0, stock_data['current_ratio'] / 3.0)
        
        # 종합 점수 (가중 평균)
        total_score = (0.4 * pe_score + 0.3 * pb_score + 0.2 * roe_score + 0.1 * cr_score)
        
        return total_score 