"""
Result Logger
백테스트 결과 로깅 및 시각화
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os


class ResultLogger:
    """
    백테스트 결과 로깅 및 시각화 클래스
    """
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        self._create_output_dir()
        
    def _create_output_dir(self):
        """출력 디렉토리 생성"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # 하위 디렉토리 생성
        subdirs = ['charts', 'reports', 'data']
        for subdir in subdirs:
            path = os.path.join(self.output_dir, subdir)
            if not os.path.exists(path):
                os.makedirs(path)
    
    def log_backtest_results(self, results: Dict[str, Any], 
                           experiment_name: str = None) -> str:
        """
        백테스트 결과 로깅
        
        Args:
            results: 백테스트 결과
            experiment_name: 실험 이름
            
        Returns:
            로그 파일 경로
        """
        if experiment_name is None:
            experiment_name = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 결과 저장
        log_file = os.path.join(self.output_dir, 'reports', f"{experiment_name}.json")
        self._save_results(results, log_file)
        
        # 요약 보고서 생성
        summary_file = os.path.join(self.output_dir, 'reports', f"{experiment_name}_summary.txt")
        self._create_summary_report(results, summary_file)
        
        # 차트 생성
        charts_dir = os.path.join(self.output_dir, 'charts', experiment_name)
        self._create_charts(results, charts_dir)
        
        print(f"결과가 저장되었습니다: {log_file}")
        return log_file
    
    def _save_results(self, results: Dict[str, Any], file_path: str):
        """결과를 JSON 파일로 저장"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    def _create_summary_report(self, results: Dict[str, Any], file_path: str):
        """요약 보고서 생성"""
        performance = results.get('performance_report', {})
        performance_metrics = performance.get('performance_metrics', {})
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("=== MOE-INVEST 백테스트 결과 요약 ===\n\n")
            
            # 기본 성과 지표
            f.write("1. 성과 지표\n")
            f.write(f"   - 총 수익률: {performance_metrics.get('total_return', 0):.2%}\n")
            f.write(f"   - 연간화 수익률: {performance_metrics.get('annualized_return', 0):.2%}\n")
            f.write(f"   - 변동성: {performance_metrics.get('volatility', 0):.2%}\n")
            f.write(f"   - 샤프 비율: {performance_metrics.get('sharpe_ratio', 0):.2f}\n")
            f.write(f"   - 최대 낙폭: {performance_metrics.get('max_drawdown', 0):.2%}\n")
            f.write(f"   - 승률: {performance_metrics.get('win_rate', 0):.2%}\n\n")
            
            # 에이전트별 성과
            agent_performance = results.get('agent_performance', {})
            f.write("2. 에이전트별 성과\n")
            for agent_name, perf in agent_performance.items():
                f.write(f"   - {agent_name}:\n")
                f.write(f"     * 총 추천 종목 수: {perf.get('total_recommendations', 0)}\n")
                f.write(f"     * 평균 신뢰도: {perf.get('avg_confidence', 0):.3f}\n")
            f.write("\n")
            
            # 최종 자본
            final_capital = results.get('final_capital', 0)
            f.write(f"3. 최종 자본: {final_capital:,.0f}원\n")
    
    def _create_charts(self, results: Dict[str, Any], charts_dir: str):
        """차트 생성"""
        if not os.path.exists(charts_dir):
            os.makedirs(charts_dir)
        
        # 1. 포트폴리오 수익률 차트
        self._plot_portfolio_returns(results, charts_dir)
        
        # 2. 에이전트별 성과 차트
        self._plot_agent_performance(results, charts_dir)
        
        # 3. 리스크 지표 차트
        self._plot_risk_metrics(results, charts_dir)
        
        # 4. 포트폴리오 구성 차트
        self._plot_portfolio_composition(results, charts_dir)
    
    def _plot_portfolio_returns(self, results: Dict[str, Any], charts_dir: str):
        """포트폴리오 수익률 차트"""
        performance = results.get('performance_report', {})
        rolling_metrics = performance.get('rolling_metrics', pd.DataFrame())
        
        if rolling_metrics.empty:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 누적 수익률
        cumulative_returns = (1 + rolling_metrics['rolling_return']).cumprod()
        axes[0, 0].plot(cumulative_returns.index, cumulative_returns.values)
        axes[0, 0].set_title('누적 수익률')
        axes[0, 0].set_ylabel('누적 수익률')
        axes[0, 0].grid(True)
        
        # 변동성
        axes[0, 1].plot(rolling_metrics.index, rolling_metrics['rolling_volatility'])
        axes[0, 1].set_title('롤링 변동성')
        axes[0, 1].set_ylabel('변동성')
        axes[0, 1].grid(True)
        
        # 샤프 비율
        axes[1, 0].plot(rolling_metrics.index, rolling_metrics['rolling_sharpe'])
        axes[1, 0].set_title('롤링 샤프 비율')
        axes[1, 0].set_ylabel('샤프 비율')
        axes[1, 0].grid(True)
        
        # 최대 낙폭
        axes[1, 1].plot(rolling_metrics.index, rolling_metrics['rolling_max_dd'])
        axes[1, 1].set_title('롤링 최대 낙폭')
        axes[1, 1].set_ylabel('최대 낙폭')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, 'portfolio_returns.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_agent_performance(self, results: Dict[str, Any], charts_dir: str):
        """에이전트별 성과 차트"""
        agent_performance = results.get('agent_performance', {})
        
        if not agent_performance:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 추천 종목 수
        agent_names = list(agent_performance.keys())
        recommendation_counts = [perf.get('total_recommendations', 0) for perf in agent_performance.values()]
        
        axes[0].bar(agent_names, recommendation_counts)
        axes[0].set_title('에이전트별 추천 종목 수')
        axes[0].set_ylabel('추천 종목 수')
        axes[0].tick_params(axis='x', rotation=45)
        
        # 평균 신뢰도
        avg_confidences = [perf.get('avg_confidence', 0) for perf in agent_performance.values()]
        
        axes[1].bar(agent_names, avg_confidences)
        axes[1].set_title('에이전트별 평균 신뢰도')
        axes[1].set_ylabel('평균 신뢰도')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, 'agent_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_risk_metrics(self, results: Dict[str, Any], charts_dir: str):
        """리스크 지표 차트"""
        performance = results.get('performance_report', {})
        risk_metrics = performance.get('risk_metrics', {})
        
        if not risk_metrics:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # VaR
        var_values = [risk_metrics.get('var_95', 0), risk_metrics.get('var_99', 0)]
        var_labels = ['VaR 95%', 'VaR 99%']
        
        axes[0, 0].bar(var_labels, var_values)
        axes[0, 0].set_title('Value at Risk')
        axes[0, 0].set_ylabel('VaR')
        
        # CVaR
        cvar_values = [risk_metrics.get('cvar_95', 0), risk_metrics.get('cvar_99', 0)]
        cvar_labels = ['CVaR 95%', 'CVaR 99%']
        
        axes[0, 1].bar(cvar_labels, cvar_values)
        axes[0, 1].set_title('Conditional Value at Risk')
        axes[0, 1].set_ylabel('CVaR')
        
        # 분포 지표
        skewness = risk_metrics.get('skewness', 0)
        kurtosis = risk_metrics.get('kurtosis', 0)
        
        axes[1, 0].bar(['Skewness', 'Kurtosis'], [skewness, kurtosis])
        axes[1, 0].set_title('분포 지표')
        axes[1, 0].set_ylabel('값')
        
        # 베타
        beta = risk_metrics.get('beta', 1.0)
        axes[1, 1].bar(['Beta'], [beta])
        axes[1, 1].set_title('베타')
        axes[1, 1].set_ylabel('베타')
        
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, 'risk_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_portfolio_composition(self, results: Dict[str, Any], charts_dir: str):
        """포트폴리오 구성 차트"""
        portfolio_history = results.get('portfolio_history', [])
        
        if not portfolio_history:
            return
        
        # 마지막 포트폴리오 구성
        last_portfolio = portfolio_history[-1]['portfolio']
        weights = last_portfolio.get('weights', {})
        
        if not weights:
            return
        
        # 상위 10개 종목만 표시
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10]
        symbols = [item[0] for item in sorted_weights]
        weight_values = [item[1] for item in sorted_weights]
        
        plt.figure(figsize=(12, 8))
        plt.pie(weight_values, labels=symbols, autopct='%1.1f%%')
        plt.title('포트폴리오 구성 (상위 10개 종목)')
        
        plt.savefig(os.path.join(charts_dir, 'portfolio_composition.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_comparison_report(self, results_list: List[Dict[str, Any]], 
                                experiment_names: List[str]) -> str:
        """
        여러 실험 결과 비교 보고서 생성
        
        Args:
            results_list: 실험 결과 리스트
            experiment_names: 실험 이름 리스트
            
        Returns:
            비교 보고서 파일 경로
        """
        comparison_file = os.path.join(self.output_dir, 'reports', 'comparison_report.txt')
        
        with open(comparison_file, 'w', encoding='utf-8') as f:
            f.write("=== 실험 결과 비교 보고서 ===\n\n")
            
            for i, (results, name) in enumerate(zip(results_list, experiment_names)):
                performance = results.get('performance_report', {})
                metrics = performance.get('performance_metrics', {})
                
                f.write(f"{i+1}. {name}\n")
                f.write(f"   - 총 수익률: {metrics.get('total_return', 0):.2%}\n")
                f.write(f"   - 샤프 비율: {metrics.get('sharpe_ratio', 0):.2f}\n")
                f.write(f"   - 최대 낙폭: {metrics.get('max_drawdown', 0):.2%}\n")
                f.write(f"   - 변동성: {metrics.get('volatility', 0):.2%}\n\n")
        
        return comparison_file 