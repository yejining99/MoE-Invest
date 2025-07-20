"""
Portfolio Evaluator
포트폴리오 성과 평가 및 리스크 분석
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """성과 지표"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    turnover_ratio: float
    win_rate: float


class PortfolioEvaluator:
    """
    포트폴리오 성과 평가 및 리스크 분석 클래스
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
    
    def evaluate_portfolio(self, portfolio_returns: pd.Series, 
                          benchmark_returns: pd.Series = None) -> PerformanceMetrics:
        """
        포트폴리오 성과 평가
        
        Args:
            portfolio_returns: 포트폴리오 수익률 시계열
            benchmark_returns: 벤치마크 수익률 시계열 (선택사항)
            
        Returns:
            성과 지표
        """
        # 기본 성과 지표 계산
        total_return = self._calculate_total_return(portfolio_returns)
        annualized_return = self._calculate_annualized_return(portfolio_returns)
        volatility = self._calculate_volatility(portfolio_returns)
        sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns)
        max_drawdown = self._calculate_max_drawdown(portfolio_returns)
        calmar_ratio = self._calculate_calmar_ratio(annualized_return, max_drawdown)
        turnover_ratio = 0.0  # 실제 구현에서는 거래 데이터 필요
        win_rate = self._calculate_win_rate(portfolio_returns)
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            turnover_ratio=turnover_ratio,
            win_rate=win_rate
        )
    
    def _calculate_total_return(self, returns: pd.Series) -> float:
        """총 수익률 계산"""
        return (1 + returns).prod() - 1
    
    def _calculate_annualized_return(self, returns: pd.Series) -> float:
        """연간화 수익률 계산"""
        total_return = self._calculate_total_return(returns)
        years = len(returns) / 252  # 252 거래일 가정
        return (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    def _calculate_volatility(self, returns: pd.Series) -> float:
        """변동성 계산"""
        return returns.std() * np.sqrt(252)  # 연간화
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """샤프 비율 계산"""
        excess_returns = returns - self.risk_free_rate / 252
        volatility = self._calculate_volatility(returns)
        
        if volatility == 0:
            return 0
        
        return (excess_returns.mean() * 252) / volatility
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """최대 낙폭 계산"""
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()
    
    def _calculate_calmar_ratio(self, annualized_return: float, max_drawdown: float) -> float:
        """칼마 비율 계산"""
        if max_drawdown == 0:
            return 0
        return annualized_return / abs(max_drawdown)
    
    def _calculate_win_rate(self, returns: pd.Series) -> float:
        """승률 계산"""
        positive_returns = returns > 0
        return positive_returns.mean()
    
    def calculate_risk_metrics(self, portfolio_returns: pd.Series) -> Dict[str, float]:
        """
        리스크 지표 계산
        
        Args:
            portfolio_returns: 포트폴리오 수익률
            
        Returns:
            리스크 지표 딕셔너리
        """
        # VaR (Value at Risk) 계산
        var_95 = np.percentile(portfolio_returns, 5)
        var_99 = np.percentile(portfolio_returns, 1)
        
        # CVaR (Conditional VaR) 계산
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        cvar_99 = portfolio_returns[portfolio_returns <= var_99].mean()
        
        # 베타 계산 (벤치마크가 있는 경우)
        beta = 1.0  # 기본값
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'beta': beta,
            'skewness': portfolio_returns.skew(),
            'kurtosis': portfolio_returns.kurtosis()
        }
    
    def calculate_rolling_metrics(self, portfolio_returns: pd.Series, 
                                 window: int = 252) -> pd.DataFrame:
        """
        롤링 성과 지표 계산
        
        Args:
            portfolio_returns: 포트폴리오 수익률
            window: 롤링 윈도우 크기
            
        Returns:
            롤링 지표 DataFrame
        """
        rolling_metrics = pd.DataFrame()
        
        # 롤링 수익률
        rolling_metrics['rolling_return'] = portfolio_returns.rolling(window).apply(
            lambda x: self._calculate_total_return(x)
        )
        
        # 롤링 변동성
        rolling_metrics['rolling_volatility'] = portfolio_returns.rolling(window).std() * np.sqrt(252)
        
        # 롤링 샤프 비율
        rolling_metrics['rolling_sharpe'] = (
            (rolling_metrics['rolling_return'] - self.risk_free_rate) / 
            rolling_metrics['rolling_volatility']
        )
        
        # 롤링 최대 낙폭
        rolling_metrics['rolling_max_dd'] = portfolio_returns.rolling(window).apply(
            lambda x: self._calculate_max_drawdown(x)
        )
        
        return rolling_metrics
    
    def compare_with_benchmark(self, portfolio_returns: pd.Series, 
                              benchmark_returns: pd.Series) -> Dict[str, float]:
        """
        벤치마크 대비 성과 비교
        
        Args:
            portfolio_returns: 포트폴리오 수익률
            benchmark_returns: 벤치마크 수익률
            
        Returns:
            비교 지표 딕셔너리
        """
        # 베타 계산
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
        
        # 알파 계산
        portfolio_mean = portfolio_returns.mean() * 252
        benchmark_mean = benchmark_returns.mean() * 252
        alpha = portfolio_mean - (self.risk_free_rate + beta * (benchmark_mean - self.risk_free_rate))
        
        # 정보 비율 계산
        excess_returns = portfolio_returns - benchmark_returns
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        
        # 추적 오차
        tracking_error = excess_returns.std() * np.sqrt(252)
        
        return {
            'beta': beta,
            'alpha': alpha,
            'information_ratio': information_ratio,
            'tracking_error': tracking_error,
            'correlation': portfolio_returns.corr(benchmark_returns)
        }
    
    def generate_performance_report(self, portfolio_returns: pd.Series, 
                                  benchmark_returns: pd.Series = None) -> Dict[str, Any]:
        """
        종합 성과 보고서 생성
        
        Args:
            portfolio_returns: 포트폴리오 수익률
            benchmark_returns: 벤치마크 수익률 (선택사항)
            
        Returns:
            성과 보고서
        """
        # 기본 성과 지표
        performance_metrics = self.evaluate_portfolio(portfolio_returns, benchmark_returns)
        
        # 리스크 지표
        risk_metrics = self.calculate_risk_metrics(portfolio_returns)
        
        # 롤링 지표
        rolling_metrics = self.calculate_rolling_metrics(portfolio_returns)
        
        report = {
            'performance_metrics': performance_metrics.__dict__,
            'risk_metrics': risk_metrics,
            'rolling_metrics': rolling_metrics,
            'summary': self._generate_summary(performance_metrics, risk_metrics)
        }
        
        # 벤치마크 비교 (있는 경우)
        if benchmark_returns is not None:
            benchmark_comparison = self.compare_with_benchmark(portfolio_returns, benchmark_returns)
            report['benchmark_comparison'] = benchmark_comparison
        
        return report
    
    def _generate_summary(self, performance_metrics: PerformanceMetrics, 
                         risk_metrics: Dict[str, float]) -> Dict[str, str]:
        """성과 요약 생성"""
        summary = {}
        
        # 수익률 평가
        if performance_metrics.annualized_return > 0.15:
            summary['return_assessment'] = "우수"
        elif performance_metrics.annualized_return > 0.08:
            summary['return_assessment'] = "양호"
        elif performance_metrics.annualized_return > 0.05:
            summary['return_assessment'] = "보통"
        else:
            summary['return_assessment'] = "미흡"
        
        # 리스크 평가
        if performance_metrics.sharpe_ratio > 1.0:
            summary['risk_adjusted_return'] = "우수"
        elif performance_metrics.sharpe_ratio > 0.5:
            summary['risk_adjusted_return'] = "양호"
        elif performance_metrics.sharpe_ratio > 0.0:
            summary['risk_adjusted_return'] = "보통"
        else:
            summary['risk_adjusted_return'] = "미흡"
        
        # 변동성 평가
        if performance_metrics.volatility < 0.15:
            summary['volatility_assessment'] = "낮음"
        elif performance_metrics.volatility < 0.25:
            summary['volatility_assessment'] = "보통"
        else:
            summary['volatility_assessment'] = "높음"
        
        return summary 