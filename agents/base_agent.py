"""
Base Agent Template for Quantitative Investment Strategies
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np


class BaseAgent(ABC):
    """
    모든 퀀트 투자 에이전트의 기본 클래스
    """
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.parameters = {}
        
    @abstractmethod
    def screen_stocks(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        주식 스크리닝을 수행하는 추상 메서드
        
        Args:
            data: OHLCV 및 재무 데이터
            
        Returns:
            스크리닝된 종목들의 DataFrame
        """
        pass
    
    @abstractmethod
    def calculate_score(self, stock_data: pd.Series) -> float:
        """
        개별 종목에 대한 점수를 계산하는 추상 메서드
        
        Args:
            stock_data: 개별 종목의 데이터
            
        Returns:
            계산된 점수 (높을수록 좋음)
        """
        pass
    
    def get_top_stocks(self, data: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
        """
        상위 N개 종목을 반환
        
        Args:
            data: 전체 데이터
            top_n: 반환할 상위 종목 수
            
        Returns:
            상위 N개 종목의 DataFrame
        """
        screened_data = self.screen_stocks(data)
        if len(screened_data) == 0:
            return pd.DataFrame()
            
        # 점수 계산
        scores = []
        for _, row in screened_data.iterrows():
            score = self.calculate_score(row)
            scores.append(score)
        
        screened_data['score'] = scores
        screened_data = screened_data.sort_values('score', ascending=False)
        
        return screened_data.head(top_n)
    
    def get_parameters(self) -> Dict[str, Any]:
        """에이전트의 파라미터를 반환"""
        return self.parameters
    
    def set_parameters(self, parameters: Dict[str, Any]):
        """에이전트의 파라미터를 설정"""
        self.parameters.update(parameters) 