"""
Chain of Thought (CoT) Reasoner for Investment Decisions
투자 결정을 위한 CoT 기반 판단자
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from .prompt_templates import PromptTemplates


class CoTReasoner:
    """
    Chain of Thought 기반 투자 판단자
    여러 에이전트의 의견을 종합하여 최종 투자 결정을 내림
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.prompt_templates = PromptTemplates()
        
    def analyze_agent_opinions(self, agent_results: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        각 에이전트의 의견을 분석
        
        Args:
            agent_results: 에이전트별 결과 딕셔너리
            
        Returns:
            분석 결과
        """
        analysis = {
            'consensus_stocks': [],
            'agent_agreement': {},
            'risk_assessment': {},
            'final_recommendations': []
        }
        
        # 모든 에이전트가 추천한 종목들 수집
        all_stocks = set()
        agent_stock_lists = {}
        
        for agent_name, result_df in agent_results.items():
            if len(result_df) > 0:
                stocks = set(result_df.index.tolist())
                agent_stock_lists[agent_name] = stocks
                all_stocks.update(stocks)
        
        # 에이전트 간 합의도 계산
        for stock in all_stocks:
            agreement_count = sum(1 for stocks in agent_stock_lists.values() if stock in stocks)
            analysis['agent_agreement'][stock] = agreement_count / len(agent_results)
        
        # 합의도가 높은 종목들을 합의 종목으로 선별
        consensus_threshold = 0.5  # 50% 이상의 에이전트가 추천
        consensus_stocks = [
            stock for stock, agreement in analysis['agent_agreement'].items()
            if agreement >= consensus_threshold
        ]
        analysis['consensus_stocks'] = consensus_stocks
        
        return analysis
    
    def generate_reasoning_prompt(self, analysis: Dict[str, Any], market_context: Dict[str, Any]) -> str:
        """
        CoT 추론을 위한 프롬프트 생성
        
        Args:
            analysis: 에이전트 분석 결과
            market_context: 시장 상황 정보
            
        Returns:
            생성된 프롬프트
        """
        prompt = self.prompt_templates.get_cot_prompt(
            consensus_stocks=analysis['consensus_stocks'],
            agent_agreement=analysis['agent_agreement'],
            market_context=market_context
        )
        
        return prompt
    
    def reason_about_investment(self, agent_results: Dict[str, pd.DataFrame], 
                              market_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        투자 결정에 대한 CoT 추론 수행
        
        Args:
            agent_results: 에이전트별 결과
            market_context: 시장 상황 정보
            
        Returns:
            추론 결과
        """
        # 에이전트 의견 분석
        analysis = self.analyze_agent_opinions(agent_results)
        
        # CoT 프롬프트 생성
        prompt = self.generate_reasoning_prompt(analysis, market_context)
        
        # LLM을 통한 추론 (실제 LLM이 없는 경우 시뮬레이션)
        if self.llm_client:
            reasoning_result = self.llm_client.generate(prompt)
        else:
            reasoning_result = self._simulate_reasoning(analysis, market_context)
        
        # 최종 투자 결정
        final_decision = self._make_final_decision(analysis, reasoning_result)
        
        return {
            'analysis': analysis,
            'reasoning': reasoning_result,
            'final_decision': final_decision
        }
    
    def _simulate_reasoning(self, analysis: Dict[str, Any], 
                           market_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        LLM이 없는 경우 추론 시뮬레이션
        
        Args:
            analysis: 분석 결과
            market_context: 시장 상황
            
        Returns:
            시뮬레이션된 추론 결과
        """
        reasoning = {
            'thought_process': [
                "1. 각 에이전트의 투자 철학을 고려하여 의견을 종합",
                "2. 합의도가 높은 종목들을 우선 검토",
                "3. 시장 상황과 리스크를 고려한 최종 선별",
                "4. 포트폴리오 다각화 원칙 적용"
            ],
            'key_considerations': [
                "에이전트 간 합의도",
                "시장 변동성",
                "섹터 분산",
                "리스크 조정 수익률"
            ],
            'recommendations': []
        }
        
        # 합의도 기반 추천
        consensus_stocks = analysis['consensus_stocks']
        for stock in consensus_stocks[:10]:  # 상위 10개
            reasoning['recommendations'].append({
                'stock': stock,
                'reason': f"여러 에이전트가 추천한 합의 종목 (합의도: {analysis['agent_agreement'][stock]:.2f})",
                'confidence': analysis['agent_agreement'][stock]
            })
        
        return reasoning
    
    def _make_final_decision(self, analysis: Dict[str, Any], 
                            reasoning: Dict[str, Any]) -> Dict[str, Any]:
        """
        최종 투자 결정
        
        Args:
            analysis: 분석 결과
            reasoning: 추론 결과
            
        Returns:
            최종 투자 결정
        """
        final_stocks = []
        
        # 추천 종목들을 신뢰도 순으로 정렬
        recommendations = reasoning.get('recommendations', [])
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 상위 종목들을 최종 선별
        for rec in recommendations[:20]:  # 최대 20개 종목
            final_stocks.append({
                'symbol': rec['stock'],
                'weight': rec['confidence'],
                'reason': rec['reason']
            })
        
        return {
            'selected_stocks': final_stocks,
            'total_weight': sum(stock['weight'] for stock in final_stocks),
            'diversification_score': self._calculate_diversification_score(final_stocks)
        }
    
    def _calculate_diversification_score(self, stocks: List[Dict[str, Any]]) -> float:
        """
        포트폴리오 다각화 점수 계산
        
        Args:
            stocks: 선별된 종목들
            
        Returns:
            다각화 점수 (0-1)
        """
        if not stocks:
            return 0.0
        
        # 가중치의 표준편차 (낮을수록 더 균등한 분배)
        weights = [stock['weight'] for stock in stocks]
        weight_std = np.std(weights)
        weight_mean = np.mean(weights)
        
        # 변동계수 (낮을수록 더 균등)
        cv = weight_std / weight_mean if weight_mean > 0 else 1.0
        
        # 다각화 점수 (변동계수가 낮을수록 높은 점수)
        diversification_score = max(0, 1 - cv)
        
        return diversification_score 