"""
Backtest Runner
백테스트 실행 및 결과 분석
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 상위 디렉토리 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.graham import GrahamAgent
from agents.piotroski import PiotroskiAgent
from agents.greenblatt import GreenblattAgent
from agents.carlisle import CarlisleAgent
from agents.driehaus import DriehausAgent
from meta_agent.cot_reasoner import CoTReasoner
from portfolio.constructor import PortfolioConstructor, PortfolioConstraints
from portfolio.evaluator import PortfolioEvaluator
from utils.data_loader import DataLoader


class BacktestRunner:
    """
    백테스트 실행 및 결과 분석 클래스
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agents = self._initialize_agents()
        self.cot_reasoner = CoTReasoner()
        self.portfolio_constructor = PortfolioConstructor()
        self.portfolio_evaluator = PortfolioEvaluator()
        self.data_loader = DataLoader()
        
    def _initialize_agents(self) -> Dict[str, Any]:
        """에이전트 초기화"""
        agents = {
            'graham': GrahamAgent(),
            'piotroski': PiotroskiAgent(),
            'greenblatt': GreenblattAgent(),
            'carlisle': CarlisleAgent(),
            'driehaus': DriehausAgent()
        }
        return agents
    
    def run_backtest(self, start_date: str, end_date: str, 
                    initial_capital: float = 100000000) -> Dict[str, Any]:
        """
        백테스트 실행
        
        Args:
            start_date: 시작일 (YYYY-MM-DD)
            end_date: 종료일 (YYYY-MM-DD)
            initial_capital: 초기 자본금
            
        Returns:
            백테스트 결과
        """
        print(f"백테스트 시작: {start_date} ~ {end_date}")
        
        # 데이터 로딩
        market_data = self.data_loader.load_market_data(start_date, end_date)
        
        # 백테스트 기간 설정
        rebalancing_dates = self._generate_rebalancing_dates(start_date, end_date)
        
        # 백테스트 실행
        portfolio_history = []
        agent_results_history = []
        
        current_portfolio = None
        current_capital = initial_capital
        
        for i, rebalancing_date in enumerate(rebalancing_dates):
            print(f"리밸런싱 {i+1}/{len(rebalancing_dates)}: {rebalancing_date}")
            
            # 현재 시점까지의 데이터로 에이전트 실행
            current_data = market_data[market_data.index <= rebalancing_date]
            
            # 각 에이전트 실행
            agent_results = self._run_agents(current_data)
            agent_results_history.append({
                'date': rebalancing_date,
                'results': agent_results
            })
            
            # CoT 추론을 통한 최종 결정
            market_context = self._get_market_context(current_data, rebalancing_date)
            final_decision = self.cot_reasoner.reason_about_investment(
                agent_results, market_context
            )
            
            # 포트폴리오 구성
            selected_stocks = final_decision['final_decision']['selected_stocks']
            new_portfolio = self.portfolio_constructor.construct_portfolio(
                selected_stocks, current_data
            )
            
            # 포트폴리오 성과 계산
            if current_portfolio is not None:
                portfolio_return = self._calculate_portfolio_return(
                    current_portfolio, new_portfolio, current_data, rebalancing_date
                )
                current_capital *= (1 + portfolio_return)
            
            # 포트폴리오 히스토리 저장
            portfolio_history.append({
                'date': rebalancing_date,
                'portfolio': new_portfolio,
                'capital': current_capital,
                'decision': final_decision
            })
            
            current_portfolio = new_portfolio
        
        # 최종 결과 분석
        backtest_results = self._analyze_backtest_results(
            portfolio_history, agent_results_history, market_data
        )
        
        return backtest_results
    
    def _generate_rebalancing_dates(self, start_date: str, end_date: str) -> List[str]:
        """리밸런싱 날짜 생성"""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        rebalancing_frequency = self.config.get('rebalancing_frequency', 'monthly')
        
        if rebalancing_frequency == 'monthly':
            dates = pd.date_range(start, end, freq='M')
        elif rebalancing_frequency == 'quarterly':
            dates = pd.date_range(start, end, freq='Q')
        else:  # weekly
            dates = pd.date_range(start, end, freq='W')
        
        return [date.strftime('%Y-%m-%d') for date in dates]
    
    def _run_agents(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """에이전트 실행"""
        agent_results = {}
        
        for agent_name, agent in self.agents.items():
            try:
                # 각 에이전트로 상위 종목 선별
                top_stocks = agent.get_top_stocks(data, top_n=50)
                agent_results[agent_name] = top_stocks
            except Exception as e:
                print(f"에이전트 {agent_name} 실행 중 오류: {e}")
                agent_results[agent_name] = pd.DataFrame()
        
        return agent_results
    
    def _get_market_context(self, data: pd.DataFrame, date: str) -> Dict[str, Any]:
        """시장 상황 정보 생성"""
        # 간단한 시장 상황 계산
        recent_data = data.tail(60)  # 최근 60일
        
        volatility = recent_data['returns'].std() if 'returns' in recent_data.columns else 0.2
        market_return = recent_data['returns'].mean() if 'returns' in recent_data.columns else 0.0
        
        return {
            'volatility': volatility,
            'market_return': market_return,
            'date': date,
            'data_points': len(recent_data)
        }
    
    def _calculate_portfolio_return(self, old_portfolio: Dict[str, Any], 
                                  new_portfolio: Dict[str, Any],
                                  market_data: pd.DataFrame, 
                                  date: str) -> float:
        """포트폴리오 수익률 계산"""
        # 간단한 구현: 가중 평균 수익률
        total_return = 0.0
        
        old_weights = old_portfolio.get('weights', {})
        
        for symbol, weight in old_weights.items():
            if symbol in market_data.index:
                # 해당 날짜의 수익률 계산
                stock_data = market_data.loc[symbol]
                if 'returns' in stock_data:
                    total_return += weight * stock_data['returns']
        
        return total_return
    
    def _analyze_backtest_results(self, portfolio_history: List[Dict[str, Any]],
                                 agent_results_history: List[Dict[str, Any]],
                                 market_data: pd.DataFrame) -> Dict[str, Any]:
        """백테스트 결과 분석"""
        # 포트폴리오 수익률 시계열 생성
        portfolio_returns = []
        dates = []
        
        for i, history in enumerate(portfolio_history):
            if i > 0:  # 첫 번째는 제외 (이전 포트폴리오가 없음)
                prev_capital = portfolio_history[i-1]['capital']
                curr_capital = history['capital']
                return_rate = (curr_capital - prev_capital) / prev_capital
                portfolio_returns.append(return_rate)
                dates.append(history['date'])
        
        portfolio_returns_series = pd.Series(portfolio_returns, index=dates)
        
        # 성과 평가
        performance_report = self.portfolio_evaluator.generate_performance_report(
            portfolio_returns_series
        )
        
        # 에이전트별 성과 분석
        agent_performance = self._analyze_agent_performance(agent_results_history)
        
        return {
            'performance_report': performance_report,
            'agent_performance': agent_performance,
            'portfolio_history': portfolio_history,
            'final_capital': portfolio_history[-1]['capital'] if portfolio_history else 0,
            'total_return': (portfolio_history[-1]['capital'] / 100000000 - 1) if portfolio_history else 0
        }
    
    def _analyze_agent_performance(self, agent_results_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """에이전트별 성과 분석"""
        agent_performance = {}
        
        for agent_name in self.agents.keys():
            agent_performance[agent_name] = {
                'total_recommendations': 0,
                'consensus_participation': 0,
                'avg_confidence': 0.0
            }
        
        for history in agent_results_history:
            results = history['results']
            
            for agent_name, result_df in results.items():
                if len(result_df) > 0:
                    agent_performance[agent_name]['total_recommendations'] += len(result_df)
                    
                    # 평균 신뢰도 계산
                    if 'score' in result_df.columns:
                        avg_score = result_df['score'].mean()
                        agent_performance[agent_name]['avg_confidence'] += avg_score
        
        # 평균 계산
        for agent_name in agent_performance:
            if agent_performance[agent_name]['total_recommendations'] > 0:
                agent_performance[agent_name]['avg_confidence'] /= len(agent_results_history)
        
        return agent_performance
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """결과 저장"""
        import json
        
        # JSON으로 저장 가능한 형태로 변환
        serializable_results = self._make_serializable(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        print(f"결과가 {output_path}에 저장되었습니다.")
    
    def _make_serializable(self, obj: Any) -> Any:
        """객체를 JSON 직렬화 가능한 형태로 변환"""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return obj 