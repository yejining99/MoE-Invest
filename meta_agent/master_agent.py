"""
Master Investment Agent (LangGraph 기반)
모든 개별 에이전트들의 의견을 종합하는 메타 에이전트
"""

import os
import pandas as pd
from typing import Dict, List, Any, Optional, TypedDict
import numpy as np

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.runnables import RunnableParallel, RunnableLambda

from langgraph.graph import StateGraph, END

# 개별 에이전트 import
from agents.graham import GrahamAgent
from agents.piotroski import PiotroskiAgent
from agents.greenblatt import GreenblattAgent
from agents.carlisle import CarlisleAgent
from agents.driehaus import DriehausAgent
from agents.base_agent import AgentInput, AgentOutput


class MasterAgentState(TypedDict):
    """마스터 에이전트 상태"""
    tickers: List[str]
    start_date: str
    end_date: str
    top_n: int
    agent_results: Dict[str, AgentOutput]
    consensus_analysis: Dict[str, Any]
    cot_analysis: str
    summary: str


class MasterInvestmentAgent:
    """
    LangGraph 기반 마스터 투자 에이전트
    모든 개별 투자 에이전트들의 의견을 종합하는 메타 에이전트
    """
    
    def __init__(self, llm=None):
        """
        메타 에이전트 초기화
        """
        # LLM 설정
        api_key = os.getenv('OPENAI_API_KEY')
        self.llm = llm or ChatOpenAI(temperature=0, model="gpt-4o", api_key=api_key)
        
        # 메모리 초기화
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # 개별 에이전트들 초기화
        self.agents = {
            'graham': GrahamAgent(llm=self.llm),
            'piotroski': PiotroskiAgent(llm=self.llm),
            'greenblatt': GreenblattAgent(llm=self.llm),
            'carlisle': CarlisleAgent(llm=self.llm),
            'driehaus': DriehausAgent(llm=self.llm)
        }
        
        # 에이전트별 설명
        self.agent_descriptions = {
            'graham': 'Benjamin Graham\'s value investing strategy (P/E, P/B, financial health)',
            'piotroski': 'Piotroski F-Score based quality investing (9-point financial score)',
            'greenblatt': 'Joel Greenblatt\'s Magic Formula (ROIC + EV/EBIT)',
            'carlisle': 'Tobias Carlisle\'s momentum strategy (volatility-adjusted momentum)',
            'driehaus': 'Richard Driehaus\'s growth investing (revenue/profit growth rate)'
        }
        
        # CoT 분석용 LLMChain 구성
        self.cot_chain = self._build_cot_chain()
        
        # LangGraph 워크플로우 구성
        self.workflow = self._build_workflow()
    
    def _build_cot_chain(self) -> LLMChain:
        """
        Chain of Thought 분석용 LLMChain 구성
        """
        cot_prompt = PromptTemplate.from_template("""
You are a professional investment analyst. Your goal is to synthesize opinions from multiple quantitative investment agents 
to make optimal investment decisions.

Each agent's investment philosophy:
- Graham Agent: Value investing, prefers low P/E and P/B ratios
- Piotroski Agent: Financial health focus, F-Score based analysis  
- Greenblatt Agent: Magic Formula based on ROIC and EV/EBIT
- Carlisle Agent: Momentum-based investing
- Driehaus Agent: Growth investing, targeting high-growth companies

## Agent Analysis Results Summary

### Recommendations by Each Agent:
{agent_summary}

### Top Consensus Stocks (Top 10):
{consensus_stocks}

## Chain of Thought Analysis Steps

Please analyze systematically through the following 4 steps:

### Step 1: Analyze consensus stocks from each investment strategy perspective
- Graham (Value): Assess undervaluation
- Piotroski (Quality): Financial health evaluation
- Greenblatt (Efficiency): Capital efficiency and valuation analysis
- Carlisle (Momentum): Price trends and technical signals
- Driehaus (Growth): Growth potential and momentum

### Step 2: Consider Current Market Environment
- Evaluate the effectiveness of each strategy in the current market
- Consider macroeconomic environment and sector rotation

### Step 3: Portfolio Construction Perspective
- Sector and style diversification
- Risk dispersion effects
- Correlation analysis

### Step 4: Final Investment Decision and Weight Allocation
- Select top 10-15 stocks
- Determine investment weight for each stock (%)
- Explain investment rationale and risk factors

Show detailed reasoning process for each step, and finally present the final portfolio in the following format:

| Rank | Stock | Weight(%) | # of Recommending Agents | Investment Rationale | Key Risks |
|------|-------|-----------|--------------------------|---------------------|-----------|
| 1    | AAPL  | 8.5       | 3                        | ...                 | ...       |

Please construct the portfolio so that total weight equals 100%.

Chat History:
{history}
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
        # agent들 중에서 이 상황에 맞는 애를 고르는 노드 추가
        workflow.add_node("select_agents", self._select_agents)
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
        모든 에이전트를 병렬로 실행 (RunnableParallel 사용)
        """
        print("🤖 모든 투자 에이전트들을 병렬로 실행 중입니다...")
        
        # AgentInput 생성
        agent_input = AgentInput(
            tickers=state["tickers"],
            start_date=state["start_date"], 
            end_date=state["end_date"],
            top_n=state["top_n"]
        )
        
        # RunnableParallel로 모든 에이전트 병렬 실행
        parallel_agents = RunnableParallel({
            agent_name: agent.chain for agent_name, agent in self.agents.items()
        })
        
        try:
            # 병렬 실행
            results = parallel_agents.invoke(agent_input)
            
            # 결과 포맷팅
            formatted_results = {}
            for agent_name, result in results.items():
                if isinstance(result, AgentOutput):
                    formatted_results[agent_name] = {
                        'screened_data': result.screened_data,
                        'top_stocks': result.top_stocks,
                        'llm_explanation': result.llm_explanation,
                        'stock_count': result.stock_count,
                        'description': self.agent_descriptions[agent_name]
                    }
                    # saving the results of agents
                    with open(f'{agent_name}_script/{state["start_date"]}_{state["end_date"]}_answer.md', 'w', encoding='utf-8') as f:
                        f.write(result.llm_explanation)

                    print(f"  ✅ {agent_name.title()}: {result.stock_count}개 종목 스크리닝 완료")
                else:
                    # 오류 처리
                    formatted_results[agent_name] = {
                        'screened_data': pd.DataFrame(),
                        'top_stocks': pd.DataFrame(),
                        'llm_explanation': f"실행 오류: {str(result)}",
                        'stock_count': 0,
                        'description': self.agent_descriptions[agent_name]
                    }
                    print(f"  ❌ {agent_name.title()}: 실행 오류")
            
            state["agent_results"] = formatted_results
            
        except Exception as e:
            print(f"❌ 에이전트 병렬 실행 중 오류: {str(e)}")
            # 빈 결과로 초기화
            state["agent_results"] = {
                agent_name: {
                    'screened_data': pd.DataFrame(),
                    'top_stocks': pd.DataFrame(), 
                    'llm_explanation': f"오류 발생: {str(e)}",
                    'stock_count': 0,
                    'description': self.agent_descriptions[agent_name]
                } for agent_name in self.agents.keys()
            }
        
        return state
    
    def _analyze_consensus(self, state: MasterAgentState) -> MasterAgentState:
        """
        에이전트 간 합의 종목 분석
        """
        print("\n🎯 에이전트 간 합의 분석 중...")
        
        agent_results = state["agent_results"]
        
        # 각 에이전트의 추천 종목 수집
        agent_recommendations = {}
        all_stocks = set()
        
        for agent_name, result in agent_results.items():
            if len(result['top_stocks']) > 0:
                # ticker 컬럼이 있으면 그것을 사용, 없으면 인덱스 사용
                if 'ticker' in result['top_stocks'].columns:
                    stocks = set(result['top_stocks']['ticker'].tolist())
                else:
                    stocks = set(result['top_stocks'].index.tolist())
                agent_recommendations[agent_name] = stocks
                all_stocks.update(stocks)
        
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
        
        return state
    
    def _generate_cot_analysis(self, state: MasterAgentState) -> MasterAgentState:
        """
        CoT 분석 생성 (LLMChain 사용)
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
        
        # CoT 분석 실행
        analysis = self.cot_chain.run({
            "agent_summary": agent_summary,
            "consensus_stocks": consensus_stocks,
            "history": str(self.memory.chat_memory.messages) if self.memory.chat_memory.messages else "No prior analysis."
        })
        state["cot_analysis"] = analysis
        
        return state
    
    def _save_results(self, state: MasterAgentState) -> MasterAgentState:
        """
        분석 결과 저장
        """
        print("\n💾 분석 결과 저장 중...")
        
        try:
            # 결과 저장 디렉토리 생성
            result_dir = f'data/meta_analysis'
            os.makedirs(result_dir, exist_ok=True)

            # 1. analysis 저장
            with open(f'{result_dir}/{state["start_date"]}_{state["end_date"]}_answer.md', 'w', encoding='utf-8') as f:
                f.write(state["cot_analysis"])


            # 1. 종합 분석 결과 저장
            with open(f'{result_dir}/{state["start_date"]}_{state["end_date"]}_comprehensive_analysis_.md', 'w', encoding='utf-8') as f:
                f.write(f"# 종합 투자 분석 결과\n")
                f.write(f"**분석 기간**: {state['start_date']} ~ {state['end_date']}\n\n")
                
                # 에이전트별 요약
                f.write("## 에이전트별 분석 요약\n\n")
                for agent_name, result in state["agent_results"].items():
                    f.write(f"### {agent_name.title()}\n")
                    f.write(f"- **전략**: {result['description']}\n")
                    f.write(f"- **스크리닝 종목 수**: {result['stock_count']}개\n\n")
                
                # 합의 분석
                f.write("## 에이전트 간 합의 분석\n\n")
                f.write(f"- **총 고유 종목 수**: {state['consensus_analysis']['total_unique_stocks']}개\n")
                f.write(f"- **상위 합의 종목들**:\n\n")
                
                for i, (stock, data) in enumerate(list(state['consensus_analysis']['consensus_stocks'].items())[:10]):
                    agents_str = ', '.join(data['recommending_agents'])
                    f.write(f"{i+1}. **{stock}**: {data['consensus_rate']:.1%} ({agents_str})\n")
                
                # CoT 분석
                f.write("\n## Chain of Thought 종합 분석\n\n")
                f.write(state["cot_analysis"])
            
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
            consensus_df.to_csv(f'{result_dir}/{state["start_date"]}_{state["end_date"]}_consensus_stocks.csv', index=False, encoding='utf-8')
            
            # 요약 생성
            summary = f"총 {len(state['tickers'])}개 종목 분석, {state['consensus_analysis']['total_unique_stocks']}개 고유 추천 종목 발견"
            state["summary"] = summary
            
            print(f"✅ 분석 결과가 {result_dir}/ 디렉토리에 저장되었습니다.")
            
        except Exception as e:
            print(f"❌ 결과 저장 중 오류: {str(e)}")
            state["summary"] = f"분석 완료, 저장 중 오류 발생: {str(e)}"
        
        return state
    
    def run_comprehensive_analysis(self, tickers: List[str], start_date: str, end_date: str, 
                                 top_n: int = 10) -> Dict[str, Any]:
        """
        포괄적인 투자 분석 실행 (LangGraph 워크플로우 사용)
        """
        print("🚀 MoE-Invest 종합 투자 분석을 시작합니다! (LangGraph 기반)\n")
        
        # 초기 상태 설정
        initial_state = {
            "tickers": tickers,
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
        
        print("\n🎉 MoE-Invest 종합 분석이 완료되었습니다!")
        
        return {
            'agent_results': final_state["agent_results"],
            'consensus_analysis': final_state["consensus_analysis"],
            'cot_analysis': final_state["cot_analysis"],
            'summary': final_state["summary"]
        } 