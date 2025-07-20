"""
Portfolio Constructor
포트폴리오 구성 및 리밸런싱
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class PortfolioConstraints:
    """포트폴리오 제약조건"""
    max_stocks: int = 20
    max_sector_weight: float = 0.3
    min_stock_weight: float = 0.01
    max_stock_weight: float = 0.1
    min_liquidity: float = 1000000000  # 10억원


class PortfolioConstructor:
    """
    포트폴리오 구성 및 최적화 클래스
    """
    
    def __init__(self, constraints: PortfolioConstraints = None):
        self.constraints = constraints or PortfolioConstraints()
        
    def construct_portfolio(self, selected_stocks: List[Dict[str, Any]], 
                          market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        포트폴리오 구성
        
        Args:
            selected_stocks: 선별된 종목들
            market_data: 시장 데이터
            
        Returns:
            구성된 포트폴리오
        """
        if not selected_stocks:
            return self._empty_portfolio()
        
        # 기본 가중치 계산
        weights = self._calculate_initial_weights(selected_stocks)
        
        # 제약조건 적용
        weights = self._apply_constraints(weights, market_data)
        
        # 최적화
        optimized_weights = self._optimize_portfolio(weights, market_data)
        
        return {
            'weights': optimized_weights,
            'constraints': self.constraints.__dict__,
            'diversification_score': self._calculate_diversification_score(optimized_weights),
            'expected_return': self._calculate_expected_return(optimized_weights, market_data),
            'expected_volatility': self._calculate_expected_volatility(optimized_weights, market_data)
        }
    
    def _calculate_initial_weights(self, selected_stocks: List[Dict[str, Any]]) -> Dict[str, float]:
        """초기 가중치 계산"""
        weights = {}
        total_score = sum(stock.get('weight', 0) for stock in selected_stocks)
        
        if total_score == 0:
            # 균등 가중치
            equal_weight = 1.0 / len(selected_stocks)
            for stock in selected_stocks:
                weights[stock['symbol']] = equal_weight
        else:
            # 점수 기반 가중치
            for stock in selected_stocks:
                weights[stock['symbol']] = stock.get('weight', 0) / total_score
        
        return weights
    
    def _apply_constraints(self, weights: Dict[str, float], 
                          market_data: pd.DataFrame) -> Dict[str, float]:
        """제약조건 적용"""
        constrained_weights = weights.copy()
        
        # 최대 종목 수 제한
        if len(constrained_weights) > self.constraints.max_stocks:
            sorted_stocks = sorted(constrained_weights.items(), 
                                 key=lambda x: x[1], reverse=True)
            constrained_weights = dict(sorted_stocks[:self.constraints.max_stocks])
        
        # 최소/최대 가중치 제한
        for symbol in list(constrained_weights.keys()):
            weight = constrained_weights[symbol]
            if weight < self.constraints.min_stock_weight:
                constrained_weights[symbol] = self.constraints.min_stock_weight
            elif weight > self.constraints.max_stock_weight:
                constrained_weights[symbol] = self.constraints.max_stock_weight
        
        # 유동성 제약
        for symbol in list(constrained_weights.keys()):
            if symbol in market_data.index:
                liquidity = market_data.loc[symbol, 'volume'] * market_data.loc[symbol, 'close']
                if liquidity < self.constraints.min_liquidity:
                    del constrained_weights[symbol]
        
        # 섹터 제약 (간단한 구현)
        sector_weights = self._calculate_sector_weights(constrained_weights, market_data)
        for sector, weight in sector_weights.items():
            if weight > self.constraints.max_sector_weight:
                # 섹터 비중이 높은 종목들의 가중치 조정
                self._adjust_sector_weights(constrained_weights, sector, 
                                         self.constraints.max_sector_weight, market_data)
        
        # 가중치 정규화
        total_weight = sum(constrained_weights.values())
        if total_weight > 0:
            for symbol in constrained_weights:
                constrained_weights[symbol] /= total_weight
        
        return constrained_weights
    
    def _optimize_portfolio(self, weights: Dict[str, float], 
                           market_data: pd.DataFrame) -> Dict[str, float]:
        """포트폴리오 최적화 (간단한 구현)"""
        # 여기서는 기본 가중치를 그대로 사용
        # 실제로는 Mean-Variance 최적화 등을 적용할 수 있음
        return weights
    
    def _calculate_sector_weights(self, weights: Dict[str, float], 
                                 market_data: pd.DataFrame) -> Dict[str, float]:
        """섹터별 가중치 계산"""
        sector_weights = {}
        
        for symbol, weight in weights.items():
            if symbol in market_data.index:
                sector = market_data.loc[symbol, 'sector']
                sector_weights[sector] = sector_weights.get(sector, 0) + weight
        
        return sector_weights
    
    def _adjust_sector_weights(self, weights: Dict[str, float], sector: str, 
                              max_weight: float, market_data: pd.DataFrame):
        """섹터 가중치 조정"""
        sector_stocks = []
        for symbol, weight in weights.items():
            if symbol in market_data.index and market_data.loc[symbol, 'sector'] == sector:
                sector_stocks.append((symbol, weight))
        
        if not sector_stocks:
            return
        
        # 가중치 순으로 정렬
        sector_stocks.sort(key=lambda x: x[1], reverse=True)
        
        # 최대 가중치에 맞춰 조정
        current_weight = sum(weight for _, weight in sector_stocks)
        adjustment_factor = max_weight / current_weight
        
        for symbol, _ in sector_stocks:
            weights[symbol] *= adjustment_factor
    
    def _calculate_diversification_score(self, weights: Dict[str, float]) -> float:
        """다각화 점수 계산"""
        if not weights:
            return 0.0
        
        # Herfindahl-Hirschman Index (HHI) 기반
        hhi = sum(weight ** 2 for weight in weights.values())
        
        # HHI를 0-1 범위의 다각화 점수로 변환
        n = len(weights)
        max_hhi = 1.0  # 완전 집중
        min_hhi = 1.0 / n  # 완전 분산
        
        if n == 1:
            return 0.0
        
        diversification_score = (max_hhi - hhi) / (max_hhi - min_hhi)
        return max(0.0, min(1.0, diversification_score))
    
    def _calculate_expected_return(self, weights: Dict[str, float], 
                                  market_data: pd.DataFrame) -> float:
        """기대 수익률 계산"""
        if not weights:
            return 0.0
        
        expected_return = 0.0
        for symbol, weight in weights.items():
            if symbol in market_data.index:
                # 간단한 과거 수익률 사용
                returns = market_data.loc[symbol, 'returns'] if 'returns' in market_data.columns else 0.0
                expected_return += weight * returns
        
        return expected_return
    
    def _calculate_expected_volatility(self, weights: Dict[str, float], 
                                      market_data: pd.DataFrame) -> float:
        """기대 변동성 계산"""
        if not weights:
            return 0.0
        
        # 간단한 가중 평균 변동성
        volatility = 0.0
        for symbol, weight in weights.items():
            if symbol in market_data.index:
                vol = market_data.loc[symbol, 'volatility'] if 'volatility' in market_data.columns else 0.2
                volatility += weight * vol
        
        return volatility
    
    def _empty_portfolio(self) -> Dict[str, Any]:
        """빈 포트폴리오 반환"""
        return {
            'weights': {},
            'constraints': self.constraints.__dict__,
            'diversification_score': 0.0,
            'expected_return': 0.0,
            'expected_volatility': 0.0
        }
    
    def rebalance_portfolio(self, current_portfolio: Dict[str, Any], 
                           new_selections: List[Dict[str, Any]], 
                           market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        포트폴리오 리밸런싱
        
        Args:
            current_portfolio: 현재 포트폴리오
            new_selections: 새로운 선별 종목들
            market_data: 시장 데이터
            
        Returns:
            리밸런싱된 포트폴리오
        """
        # 새로운 포트폴리오 구성
        new_portfolio = self.construct_portfolio(new_selections, market_data)
        
        # 리밸런싱 비용 계산
        rebalancing_cost = self._calculate_rebalancing_cost(
            current_portfolio.get('weights', {}),
            new_portfolio['weights']
        )
        
        new_portfolio['rebalancing_cost'] = rebalancing_cost
        
        return new_portfolio
    
    def _calculate_rebalancing_cost(self, old_weights: Dict[str, float], 
                                   new_weights: Dict[str, float]) -> float:
        """리밸런싱 비용 계산"""
        cost = 0.0
        
        all_symbols = set(old_weights.keys()) | set(new_weights.keys())
        
        for symbol in all_symbols:
            old_weight = old_weights.get(symbol, 0.0)
            new_weight = new_weights.get(symbol, 0.0)
            
            # 거래량에 따른 비용 계산
            trade_volume = abs(new_weight - old_weight)
            cost += trade_volume * 0.001  # 0.1% 거래비용 가정
        
        return cost 