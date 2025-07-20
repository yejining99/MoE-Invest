"""
Prompt Templates for LLM-based Investment Decisions
LLM 기반 투자 결정을 위한 프롬프트 템플릿
"""

from typing import Dict, List, Any


class PromptTemplates:
    """
    LLM 프롬프트 템플릿 클래스
    """
    
    def __init__(self):
        self.system_prompt = self._get_system_prompt()
    
    def _get_system_prompt(self) -> str:
        """시스템 프롬프트 반환"""
        return """
당신은 전문적인 투자 분석가입니다. 여러 퀀트 투자 에이전트의 의견을 종합하여 
최적의 투자 결정을 내리는 것이 목표입니다.

각 에이전트의 투자 철학:
- Graham Agent: 가치투자, 낮은 P/E, P/B 선호
- Piotroski Agent: 재무 건전성, F-Score 기반
- Greenblatt Agent: ROIC와 EV/EBIT 기반 Magic Formula
- Carlisle Agent: 모멘텀 기반 투자
- Driehaus Agent: 성장 투자, 고성장 기업 선별

분석 시 고려사항:
1. 에이전트 간 합의도
2. 시장 상황과 리스크
3. 포트폴리오 다각화
4. 리스크 조정 수익률
"""
    
    def get_cot_prompt(self, consensus_stocks: List[str], 
                       agent_agreement: Dict[str, float],
                       market_context: Dict[str, Any]) -> str:
        """
        Chain of Thought 추론을 위한 프롬프트 생성
        
        Args:
            consensus_stocks: 합의 종목 리스트
            agent_agreement: 에이전트 간 합의도
            market_context: 시장 상황 정보
            
        Returns:
            생성된 프롬프트
        """
        prompt = f"""
{self.system_prompt}

현재 시장 상황:
- 시장 변동성: {market_context.get('volatility', 'N/A')}
- 섹터별 성과: {market_context.get('sector_performance', 'N/A')}
- 경제 지표: {market_context.get('economic_indicators', 'N/A')}

에이전트 분석 결과:
합의 종목들: {consensus_stocks[:10]}  # 상위 10개

에이전트 간 합의도 (상위 5개):
{self._format_agreement_table(agent_agreement, 5)}

다음 단계로 추론해주세요:

1단계: 각 에이전트의 투자 철학을 고려하여 합의 종목들을 분석
2단계: 현재 시장 상황과 리스크를 고려한 종목 선별
3단계: 포트폴리오 다각화 원칙을 적용한 최종 선별
4단계: 최종 투자 결정과 근거 제시

각 단계별로 상세한 추론 과정을 보여주세요.
"""
        return prompt
    
    def get_risk_assessment_prompt(self, selected_stocks: List[str],
                                  market_context: Dict[str, Any]) -> str:
        """
        리스크 평가를 위한 프롬프트 생성
        
        Args:
            selected_stocks: 선별된 종목들
            market_context: 시장 상황 정보
            
        Returns:
            생성된 프롬프트
        """
        prompt = f"""
{self.system_prompt}

선별된 종목들: {selected_stocks}

시장 상황:
- 변동성: {market_context.get('volatility', 'N/A')}
- 섹터 분포: {market_context.get('sector_distribution', 'N/A')}
- 거시경제 지표: {market_context.get('macro_indicators', 'N/A')}

다음 관점에서 리스크를 평가해주세요:

1. 개별 종목 리스크
   - 재무 건전성
   - 업종별 리스크
   - 유동성 리스크

2. 포트폴리오 리스크
   - 섹터 집중도
   - 상관관계 분석
   - 변동성 분석

3. 시장 리스크
   - 거시경제 영향
   - 섹터 순환
   - 이벤트 리스크

각 리스크 요소에 대한 점수(1-10)와 대응 방안을 제시해주세요.
"""
        return prompt
    
    def get_portfolio_optimization_prompt(self, stock_weights: Dict[str, float],
                                        constraints: Dict[str, Any]) -> str:
        """
        포트폴리오 최적화를 위한 프롬프트 생성
        
        Args:
            stock_weights: 종목별 가중치
            constraints: 제약조건
            
        Returns:
            생성된 프롬프트
        """
        prompt = f"""
{self.system_prompt}

현재 포트폴리오 가중치:
{self._format_weight_table(stock_weights)}

제약조건:
- 최대 종목 수: {constraints.get('max_stocks', 20)}
- 최대 섹터 비중: {constraints.get('max_sector_weight', 0.3)}
- 최소 유동성: {constraints.get('min_liquidity', 'N/A')}

포트폴리오 최적화 목표:
1. 리스크 조정 수익률 최대화
2. 다각화 효과 극대화
3. 거래비용 최소화

다음 관점에서 최적화 방안을 제시해주세요:

1. 가중치 조정 방안
2. 리밸런싱 전략
3. 리스크 관리 방안
4. 성과 모니터링 지표
"""
        return prompt
    
    def _format_agreement_table(self, agreement: Dict[str, float], top_n: int) -> str:
        """합의도 테이블 포맷팅"""
        sorted_agreement = sorted(agreement.items(), key=lambda x: x[1], reverse=True)
        
        table = "종목\t\t합의도\n"
        table += "-" * 30 + "\n"
        
        for stock, agreement_rate in sorted_agreement[:top_n]:
            table += f"{stock}\t\t{agreement_rate:.2f}\n"
        
        return table
    
    def _format_weight_table(self, weights: Dict[str, float]) -> str:
        """가중치 테이블 포맷팅"""
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        
        table = "종목\t\t가중치\n"
        table += "-" * 30 + "\n"
        
        for stock, weight in sorted_weights:
            table += f"{stock}\t\t{weight:.3f}\n"
        
        return table
    
    def get_summary_prompt(self, analysis_results: Dict[str, Any]) -> str:
        """
        분석 결과 요약을 위한 프롬프트 생성
        
        Args:
            analysis_results: 분석 결과
            
        Returns:
            생성된 프롬프트
        """
        prompt = f"""
{self.system_prompt}

분석 결과 요약:

1. 에이전트별 추천 종목 수:
{self._format_agent_summary(analysis_results.get('agent_results', {}))}

2. 합의 종목 분석:
- 총 합의 종목 수: {len(analysis_results.get('consensus_stocks', []))}
- 평균 합의도: {self._calculate_avg_agreement(analysis_results.get('agent_agreement', {})):.2f}

3. 최종 투자 결정:
{self._format_final_decision(analysis_results.get('final_decision', {}))}

이 결과를 바탕으로 투자 전략의 핵심 포인트와 향후 모니터링 방안을 요약해주세요.
"""
        return prompt
    
    def _format_agent_summary(self, agent_results: Dict[str, Any]) -> str:
        """에이전트별 요약 포맷팅"""
        summary = ""
        for agent_name, result in agent_results.items():
            stock_count = len(result) if hasattr(result, '__len__') else 0
            summary += f"- {agent_name}: {stock_count}개 종목\n"
        return summary
    
    def _calculate_avg_agreement(self, agreement: Dict[str, float]) -> float:
        """평균 합의도 계산"""
        if not agreement:
            return 0.0
        return sum(agreement.values()) / len(agreement)
    
    def _format_final_decision(self, decision: Dict[str, Any]) -> str:
        """최종 결정 포맷팅"""
        if not decision:
            return "결정 없음"
        
        selected_stocks = decision.get('selected_stocks', [])
        total_weight = decision.get('total_weight', 0)
        diversification_score = decision.get('diversification_score', 0)
        
        summary = f"- 선별 종목 수: {len(selected_stocks)}개\n"
        summary += f"- 총 가중치: {total_weight:.3f}\n"
        summary += f"- 다각화 점수: {diversification_score:.3f}\n"
        
        return summary 