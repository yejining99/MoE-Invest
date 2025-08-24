"""
Master Portfolio Allocation Agent (LangGraph 기반)
저장된 개별 에이전트들의 분석 결과를 읽어서 종합하여 최적 포트폴리오 비중을 결정하는 메타 에이전트
"""

import os
import pandas as pd
from typing import Dict, List, Any, Optional, TypedDict
import numpy as np
import json
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.runnables import RunnableParallel, RunnableLambda

from langgraph.graph import StateGraph, END

class MasterAgentState(TypedDict):
    """마스터 에이전트 상태"""
    start_date: str
    end_date: str
    top_n: int
    agent_results: Dict[str, Any]
    consensus_analysis: Dict[str, Any]
    cot_analysis: str
    summary: str


class InvestmentMemory:
    """투자 결정 메모리 관리 클래스"""
    
    def __init__(self, memory_file: str = "results/meta_analysis/investment_memory.json"):
        self.memory_file = memory_file
        self.memories = self._load_memories()
    
    def _load_memories(self) -> List[Dict]:
        """저장된 메모리 로드"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"메모리 로드 실패: {e}")
        return []
    
    def save_memory(self, decision_summary: str, market_context: str, 
                   recommendation: str, outcomes: str = ""):
        """새로운 투자 결정 메모리 저장"""
        memory_entry = {
            "timestamp": datetime.now().isoformat(),
            "decision_summary": decision_summary,
            "market_context": market_context,
            "recommendation": recommendation,
            "outcomes": outcomes
        }
        
        self.memories.append(memory_entry)
        self._save_memories()
    
    def get_relevant_memories(self, current_context: str, n_matches: int = 2) -> List[Dict]:
        """현재 상황과 유사한 과거 메모리 검색"""
        if not self.memories:
            return []
        
        # 간단한 키워드 매칭 (실제로는 더 정교한 임베딩 검색 사용 가능)
        context_keywords = set(current_context.lower().split())
        
        scored_memories = []
        for memory in self.memories[-10:]:  # 최근 10개만 검색
            memory_text = f"{memory['market_context']} {memory['decision_summary']}".lower()
            memory_keywords = set(memory_text.split())
            
            # 키워드 겹치는 비율로 점수 계산
            overlap = len(context_keywords.intersection(memory_keywords))
            total_keywords = len(context_keywords.union(memory_keywords))
            score = overlap / total_keywords if total_keywords > 0 else 0
            
            scored_memories.append((score, memory))
        
        # 점수 높은 순으로 정렬하여 상위 n개 반환
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [memory for score, memory in scored_memories[:n_matches]]
    
    def _save_memories(self):
        """메모리를 파일에 저장"""
        try:
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.memories, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"메모리 저장 실패: {e}")


class MasterInvestmentAgent:
    """
    LangGraph 기반 포트폴리오 구성 에이전트
    저장된 개별 투자 에이전트들의 분석 결과를 읽어서 종합하여 최적 비중 할당을 결정하는 메타 에이전트
    
    각 에이전트는 results/{agent_name}_agent/ 경로에 저장된 분석 결과 파일을 참조:
    - BenjaminGraham -> results/graham_agent/
    - JosephPiotroski -> results/piotroski_agent/  
    - JoelGreenblatt -> results/greenblatt_agent/
    - EdwardAltman -> results/altman_agent/
    - WarrenBuffett -> results/buffett_agent/
    """
    
    def __init__(self, llm=None):
        """
        포트폴리오 구성 메타 에이전트 초기화
        """
        # LLM 설정
        api_key = os.getenv('OPENAI_API_KEY')
        self.llm = llm or ChatOpenAI(temperature=0, model="gpt-4o", api_key=api_key)
        
        # 메모리 초기화
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.investment_memory = InvestmentMemory()
        
        # 에이전트별 결과 경로 매핑
        self.agent_result_paths = {
            'BenjaminGraham': 'results/graham_agent',
            'JosephPiotroski': 'results/piotroski_agent', 
            'JoelGreenblatt': 'results/greenblatt_agent',
            'EdwardAltman': 'results/altman_agent',
            'WarrenBuffett': 'results/buffett_agent'
        }
        
        # 에이전트별 설명
        self.agent_descriptions = {
            'BenjaminGraham': 'Benjamin Graham\'s deep value investing strategy (P/E, P/B, financial safety)',
            'JosephPiotroski': 'Joseph Piotroski F-Score based quality investing (9-point financial score)',
            'JoelGreenblatt': 'Joel Greenblatt\'s Magic Formula investing (ROIC + EV/EBIT)',
            'EdwardAltman': 'Edward Altman Z-Score based bankruptcy prediction and distressed investing',
            'WarrenBuffett': 'Warren Buffett\'s quality value investing (moats, owner earnings, compounding)'
        }
        self.agents = list(self.agent_result_paths.keys())
        
        # CoT 분석용 LLMChain 구성
        self.cot_chain = self._build_cot_chain()
        
        # LangGraph 워크플로우 구성
        self.workflow = self._build_workflow()
    
    def check_available_results(self, start_date: str, end_date: str) -> Dict[str, bool]:
        """
        지정된 기간에 대한 각 에이전트의 결과 파일 가용성 확인
        
        Args:
            start_date: 분석 시작일 (YYYY-MM-DD)
            end_date: 분석 종료일 (YYYY-MM-DD)
            
        Returns:
            Dict[str, bool]: 각 에이전트별 결과 파일 존재 여부
        """
        availability = {}
        
        for agent_name, result_path in self.agent_result_paths.items():
            agent_prefix = {
                'BenjaminGraham': 'graham',
                'JosephPiotroski': 'piotroski', 
                'JoelGreenblatt': 'greenblatt',
                'EdwardAltman': 'altman',
                'WarrenBuffett': 'buffett'
            }[agent_name]
            
            analysis_file = f"{result_path}/{agent_prefix}_analysis_{start_date}_{end_date}.json"
            availability[agent_name] = os.path.exists(analysis_file)
        
        return availability
    
    def _build_cot_chain(self) -> LLMChain:
        """
        Chain of Thought 분석용 LLMChain 구성
        """
        cot_prompt = PromptTemplate.from_template("""
You are a senior portfolio manager responsible for constructing optimal investment portfolios based on analysis from multiple investment experts. 
Your role is to evaluate the collective wisdom of different investment strategies and determine the optimal allocation weights for each recommended stock.

## Investment Expert Analysis Summary

### Expert Recommendations:
{agent_summary}

### Highest Consensus Stocks:
{consensus_stocks}

## Your Portfolio Construction Framework

As the portfolio manager, you will ALWAYS invest (BUY), but you must determine the optimal allocation weights for each stock. 
Your goal is to construct a balanced portfolio that maximizes expected returns while managing risk through diversification.

Your analysis should be conversational and natural, as if you're explaining your portfolio construction process to the investment committee. 
Focus on the most compelling arguments for each allocation decision.

### Investment Expert Philosophies:
- **Benjamin Graham**: Deep value investing, margin of safety, financial strength analysis
- **Joseph Piotroski**: Financial quality screening via F-Score metrics (9-point system)  
- **Joel Greenblatt**: Magic Formula combining high ROIC with attractive valuation
- **Edward Altman**: Credit risk analysis via Z-Score, distressed opportunity identification
- **Warren Buffett**: Quality compounders with moats, owner earnings, long-term value creation

### Critical Portfolio Questions:
1. **Which stocks deserve the highest allocation?** Based on expert consensus and conviction levels
2. **How should we balance concentration vs diversification?** Risk management through position sizing
3. **What sector/style exposures are we creating?** Ensure balanced portfolio construction
4. **How do we weight expert opinions?** Consider track record and market environment fit

### Required Portfolio Construction Format:

**PORTFOLIO ALLOCATION DECISION**

**INVESTMENT PHILOSOPHY:**
[Explain your 2-3 core principles for this portfolio construction based on expert analysis]

**KEY ALLOCATION DRIVERS:**
1. [Primary factor influencing allocation decisions]
2. [Secondary factor for portfolio balance]
3. [Risk management consideration]

**FINAL PORTFOLIO ALLOCATION:**
| Rank | Stock | Weight(%) | Supporting Experts | Key Investment Thesis | Risk Level |
|------|-------|-----------|-------------------|----------------------|------------|
| 1    | AAPL  | 15.0      | Graham, Piotroski | Strong fundamentals + growth | Medium |
| 2    | MSFT  | 12.5      | Greenblatt, Driehaus | High ROIC + momentum | Medium |
| ...  | ...   | ...       | ...               | ...                  | ...    |

**PORTFOLIO CHARACTERISTICS:**
- Total Allocation: 100%
- Number of Holdings: [X] stocks  
- Top 5 Holdings: [X]% of portfolio
- Sector Concentration: [Describe main sector exposures]

**RISK MANAGEMENT:**
- Key portfolio risks: [List main concentration/sector risks]
- Monitoring metrics: [What to watch for rebalancing]
- Rebalancing triggers: [When to adjust allocations]

Take into account your past portfolio construction decisions and performance outcomes. Use these insights to refine your allocation methodology.

Previous analysis history:
{history}

Remember: You are always investing 100% of the portfolio. Focus on optimal weight allocation, not whether to invest.
""")
        
        return LLMChain(
            llm=self.llm,
            prompt=cot_prompt,
            verbose=True
        )
    
    def _build_workflow(self) -> StateGraph:
        """
        LangGraph 워크플로우 구성
        """
        # 상태 그래프 생성
        workflow = StateGraph(MasterAgentState)
        
        # 노드 추가 
        workflow.add_node("run_agents_parallel", self._run_agents_parallel)
        workflow.add_node("analyze_consensus", self._analyze_consensus) 
        workflow.add_node("generate_cot_analysis", self._generate_cot_analysis)
        workflow.add_node("save_results", self._save_results)
        
        # 엣지 추가 (플로우 정의)
        workflow.set_entry_point("run_agents_parallel")
        workflow.add_edge("run_agents_parallel", "analyze_consensus")
        workflow.add_edge("analyze_consensus", "generate_cot_analysis")
        workflow.add_edge("generate_cot_analysis", "save_results")
        workflow.add_edge("save_results", END)
        
        return workflow.compile()
    
    def _run_agents_parallel(self, state: MasterAgentState) -> MasterAgentState:
        """
        저장된 에이전트 결과 파일들을 읽어서 분석
        """
        print("📂 저장된 투자 에이전트 결과들을 읽어오는 중...")
        
        formatted_results = {}
        
        for agent_name, result_path in self.agent_result_paths.items():
            try:
                print(f"  📖 {agent_name} 결과 읽는 중...")
                
                # 파일명 생성 (agent_name을 소문자로 변환)
                agent_prefix = {
                    'BenjaminGraham': 'graham',
                    'JosephPiotroski': 'piotroski', 
                    'JoelGreenblatt': 'greenblatt',
                    'EdwardAltman': 'altman',
                    'WarrenBuffett': 'buffett'
                }[agent_name]
                
                analysis_file = f"{result_path}/{agent_prefix}_analysis_{state['start_date']}_{state['end_date']}.json"
                
                # 파일 존재 확인 및 읽기
                if os.path.exists(analysis_file):
                    with open(analysis_file, 'r', encoding='utf-8') as f:
                        analysis_data = json.load(f)
                    
                    # final_output 추출
                    analysis_result = analysis_data.get('final_output', '분석 결과가 없습니다.')
                    execution_time = analysis_data.get('execution_time_seconds', 0)
                    
                    # 결과에서 종목 수 추출 (간단한 추정)
                    stock_lines = [line for line in analysis_result.split('\n') if '|' in line and any(word in line.upper() for word in ['TICKER', 'SYMBOL', 'STOCK'])]
                    stock_count = max(0, len(stock_lines) - 1) if stock_lines else 0  # 헤더 제외
                    
                    formatted_results[agent_name] = {
                        'llm_explanation': analysis_result,
                        'stock_count': stock_count,
                        'description': self.agent_descriptions[agent_name],
                        'execution_time': execution_time,
                        'file_path': analysis_file
                    }
                    
                    print(f"  ✅ {agent_name}: 결과 로드 완료 (추정 {stock_count}개 종목, {execution_time:.1f}초)")
                    
                else:
                    formatted_results[agent_name] = {
                        'llm_explanation': f"결과 파일을 찾을 수 없습니다: {analysis_file}",
                        'stock_count': 0,
                        'description': self.agent_descriptions[agent_name],
                        'execution_time': 0,
                        'file_path': analysis_file
                    }
                    print(f"  ⚠️ {agent_name}: 파일 없음 - {analysis_file}")
                    
            except Exception as e:
                formatted_results[agent_name] = {
                    'llm_explanation': f"파일 읽기 오류: {str(e)}",
                    'stock_count': 0,
                    'description': self.agent_descriptions[agent_name],
                    'execution_time': 0,
                    'file_path': f"{result_path}/{agent_prefix}_analysis_{state['start_date']}_{state['end_date']}.json"
                }
                print(f"  ❌ {agent_name}: 파일 읽기 오류 - {str(e)}")
        
        state["agent_results"] = formatted_results
        
        # 성공적으로 로드된 에이전트 수 출력
        successful_loads = len([r for r in formatted_results.values() if 'file_path' in r and os.path.exists(r['file_path'])])
        total_agents = len(formatted_results)
        print(f"\n📊 {successful_loads}/{total_agents}개 에이전트 결과 로드 완료!")
        
        return state
    
    def _analyze_consensus(self, state: MasterAgentState) -> MasterAgentState:
        """
        에이전트 간 합의 종목 분석 (분석 결과 텍스트에서 티커 추출)
        """
        print("\n🎯 에이전트 간 합의 분석 중...")
        
        agent_results = state["agent_results"]
        
        # 일반적인 주식 티커 패턴 (3-5글자 대문자)
        import re
        ticker_pattern = r'\b[A-Z]{1,5}\b'
        
        # 각 에이전트의 추천 종목 수집
        agent_recommendations = {}
        all_stocks = set()
        
        for agent_name, result in agent_results.items():
            if result['llm_explanation']:
                # 분석 결과에서 티커 추출
                analysis_text = result['llm_explanation']
                found_tickers = re.findall(ticker_pattern, analysis_text)
                
                # 일반적인 단어들 제외 (NOT, AND, THE 등)
                excluded_words = {'THE', 'AND', 'OR', 'NOT', 'BUT', 'FOR', 'IN', 'ON', 'AT', 'TO', 'BY', 'UP', 'OF', 'IS', 'ARE', 'WAS', 'WERE', 'BE', 'BEEN', 'HAVE', 'HAS', 'HAD', 'DO', 'DOES', 'DID', 'WILL', 'WOULD', 'COULD', 'SHOULD', 'MAY', 'MIGHT', 'CAN', 'MUST', 'SHALL'}
                valid_tickers = [ticker for ticker in found_tickers if ticker not in excluded_words and len(ticker) >= 2]
                
                # 빈도 기반 상위 종목 선택 (상위 10개)
                from collections import Counter
                ticker_counts = Counter(valid_tickers)
                top_tickers = [ticker for ticker, count in ticker_counts.most_common(10)]
                
                agent_recommendations[agent_name] = set(top_tickers)
                all_stocks.update(top_tickers)
                
                print(f"  📊 {agent_name}: {len(top_tickers)}개 종목 추출 ({', '.join(list(top_tickers)[:5])}{'...' if len(top_tickers) > 5 else ''})")
            else:
                agent_recommendations[agent_name] = set()
        
        # 합의도 계산
        consensus_analysis = {}
        for stock in all_stocks:
            recommending_agents = [agent for agent, stocks in agent_recommendations.items() 
                                 if stock in stocks]
            consensus_rate = len(recommending_agents) / len(self.agents)
            
            consensus_analysis[stock] = {
                'consensus_rate': consensus_rate,
                'recommending_agents': recommending_agents,
                'agent_count': len(recommending_agents)
            }
        
        # 합의도별 정렬
        sorted_consensus = dict(sorted(consensus_analysis.items(), 
                                     key=lambda x: x[1]['consensus_rate'], 
                                     reverse=True))
        
        state["consensus_analysis"] = {
            'consensus_stocks': sorted_consensus,
            'total_unique_stocks': len(all_stocks),
            'agent_recommendations': agent_recommendations
        }
        
        print(f"  🎯 총 {len(all_stocks)}개 고유 종목 발견, 최고 합의율: {max([data['consensus_rate'] for data in consensus_analysis.values()]) * 100:.1f}%" if consensus_analysis else "  ⚠️ 추출된 종목 없음")
        
        return state
    
    def _generate_cot_analysis(self, state: MasterAgentState) -> MasterAgentState:
        """
        CoT 분석 생성 (LLMChain 사용, 과거 메모리 활용)
        """
        print("\n🧠 Chain of Thought 분석 중...")
        
        agent_results = state["agent_results"]
        consensus_analysis = state["consensus_analysis"]
        
        # 에이전트 요약 생성
        agent_summary = ""
        for agent_name, result in agent_results.items():
            agent_summary += f"- **{agent_name.title()}** ({result['description']}): {result['stock_count']} stocks screened\n"
        
        # 합의 종목 생성
        consensus_stocks = ""
        for i, (stock, data) in enumerate(list(consensus_analysis['consensus_stocks'].items())[:10]):
            agents_str = ', '.join(data['recommending_agents'])
            consensus_stocks += f"{i+1}. **{stock}**: {data['consensus_rate']:.1%} consensus rate ({agents_str})\n"
        
        # 현재 시장 상황 요약 (과거 메모리 검색을 위해)
        current_situation = f"{agent_summary}\n{consensus_stocks}"
        
        # 과거 유사한 상황의 투자 결정 검색
        past_memories = self.investment_memory.get_relevant_memories(current_situation, n_matches=2)
        
        # 과거 메모리를 프롬프트에 포함할 형태로 포맷팅
        past_memory_str = ""
        if past_memories:
            for i, memory in enumerate(past_memories, 1):
                past_memory_str += f"Past Decision {i} ({memory['timestamp'][:10]}):\n"
                past_memory_str += f"Context: {memory['market_context']}\n"
                past_memory_str += f"Decision: {memory['recommendation']}\n"
                if memory['outcomes']:
                    past_memory_str += f"Outcomes: {memory['outcomes']}\n"
                past_memory_str += "\n"
        else:
            past_memory_str = "No relevant past decisions found."
        
        # CoT 분석 실행 (과거 메모리 포함)
        analysis = self.cot_chain.run({
            "agent_summary": agent_summary,
            "consensus_stocks": consensus_stocks,
            "history": past_memory_str
        })
        state["cot_analysis"] = analysis
        
        # 현재 포트폴리오 구성을 메모리에 저장
        self.investment_memory.save_memory(
            decision_summary=f"Portfolio allocation for {state['consensus_analysis']['total_unique_stocks']} stocks from {state['start_date']} to {state['end_date']}",
            market_context=current_situation,
            recommendation=analysis[:500] + "..." if len(analysis) > 500 else analysis  # 요약본만 저장
        )
        
        return state
    
    def _save_results(self, state: MasterAgentState) -> MasterAgentState:
        """
        분석 결과 저장
        """
        print("\n💾 분석 결과 저장 중...")
        
        try:
            # 결과 저장 디렉토리 생성
            result_dir = f'results/meta_analysis'
            os.makedirs(result_dir, exist_ok=True)

            # 1. analysis 저장
            with open(f'{result_dir}/answer_{state["start_date"]}_{state["end_date"]}.md', 'w', encoding='utf-8') as f:
                f.write(state["cot_analysis"])


            # 1. 포트폴리오 할당 결과 저장
            with open(f'{result_dir}/portfolio_allocation_{state["start_date"]}_{state["end_date"]}.md', 'w', encoding='utf-8') as f:
                f.write(f"# 📊 최적 포트폴리오 구성 보고서\n")
                f.write(f"**분석 기간**: {state['start_date']} ~ {state['end_date']}\n")
                f.write(f"**구성 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # 포트폴리오 매니저 최종 구성 (CoT 분석)
                f.write("## 💼 포트폴리오 매니저 최적 할당 결정\n\n")
                f.write(state["cot_analysis"])
                f.write("\n\n---\n\n")
                
                # 상세 분석 배경
                f.write("## 📊 분석 배경 정보\n\n")
                
                # 에이전트별 요약
                f.write("### 투자 전문가별 분석 요약\n\n")
                for agent_name, result in state["agent_results"].items():
                    f.write(f"**{agent_name.title()}** ({result['description']})\n")
                    f.write(f"- 스크리닝 종목 수: {result['stock_count']}개\n")
                    if result['llm_explanation']:
                        # 첫 100자만 미리보기
                        preview = result['llm_explanation'][:100].replace('\n', ' ')
                        f.write(f"- 분석 미리보기: {preview}...\n")
                    f.write("\n")
                
                # 합의 분석
                f.write("### 전문가 간 합의 분석\n\n")
                f.write(f"- **총 고유 추천 종목**: {state['consensus_analysis']['total_unique_stocks']}개\n")
                f.write(f"- **상위 합의 종목들**:\n\n")
                
                for i, (stock, data) in enumerate(list(state['consensus_analysis']['consensus_stocks'].items())[:15]):
                    agents_str = ', '.join(data['recommending_agents'])
                    f.write(f"{i+1}. **{stock}**: {data['consensus_rate']:.1%} 합의율 ({agents_str})\n")
            
            # 2. 합의 종목 상세 데이터 저장
            consensus_df = pd.DataFrame([
                {
                    'stock': stock,
                    'consensus_rate': data['consensus_rate'],
                    'agent_count': data['agent_count'],
                    'recommending_agents': ', '.join(data['recommending_agents'])
                }
                for stock, data in state['consensus_analysis']['consensus_stocks'].items()
            ])
            consensus_df.to_csv(f'{result_dir}/consensus_stocks_{state["start_date"]}_{state["end_date"]}.csv', index=False, encoding='utf-8')
            
            # 요약 생성
            summary = f"포트폴리오 최적 구성 완료: {state['consensus_analysis']['total_unique_stocks']}개 투자 후보에서 포트폴리오 구성"
            state["summary"] = summary
            
            print(f"✅ 포트폴리오 구성 보고서가 {result_dir}/ 디렉토리에 저장되었습니다.")
            print(f"📊 메인 리포트: {result_dir}/{state['start_date']}_{state['end_date']}_portfolio_allocation.md")
            print(f"🧠 상세 분석: {result_dir}/{state['start_date']}_{state['end_date']}_answer.md")
            
        except Exception as e:
            print(f"❌ 결과 저장 중 오류: {str(e)}")
            state["summary"] = f"분석 완료, 저장 중 오류 발생: {str(e)}"
        
        return state
    
    def run_comprehensive_analysis(self, start_date: str, end_date: str, 
                                 top_n: int = 10) -> Dict[str, Any]:
        """
        저장된 에이전트 결과들을 종합하여 최적 포트폴리오 구성 실행 (LangGraph 워크플로우 사용)
        """
        print("🚀 MoE-Invest 포트폴리오 매니저: 저장된 분석 결과 종합 및 최적 비중 할당! (LangGraph 기반)\n")
        print(f"📅 분석 기간: {start_date} ~ {end_date}")
        print(f"📂 결과 경로: results/{{agent_name}}_agent/")
        
        # 결과 파일 가용성 확인
        availability = self.check_available_results(start_date, end_date)
        available_count = sum(availability.values())
        total_count = len(availability)
        
        print(f"📊 사용 가능한 에이전트 결과: {available_count}/{total_count}개")
        for agent_name, available in availability.items():
            status = "✅" if available else "❌"
            print(f"  {status} {agent_name}")
        print("")
        
        # 초기 상태 설정
        initial_state = {
            "start_date": start_date,
            "end_date": end_date,
            "top_n": top_n,
            "agent_results": {},
            "consensus_analysis": {},
            "cot_analysis": "",
            "summary": ""
        }
        
        # 워크플로우 실행
        final_state = self.workflow.invoke(initial_state)
        
        print("\n🎉 포트폴리오 매니저의 최적 비중 할당이 완료되었습니다!")
        print("📊 투자 준비 완료! 각 종목별 정확한 할당 비중이 결정되었습니다!")
        
        return {
            'agent_results': final_state["agent_results"],
            'consensus_analysis': final_state["consensus_analysis"],
            'cot_analysis': final_state["cot_analysis"],
            'summary': final_state["summary"]
        } 