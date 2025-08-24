"""
Integrated MoE-Invest System
Router Agent + Selected Agents + Meta Agent 통합 시스템
"""

import os
import pandas as pd
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from langchain_openai import ChatOpenAI

from agents.router_agent import RouterAgent
from meta_agent.master_agent import MasterInvestmentAgent

from dotenv import load_dotenv
load_dotenv()

class IntegratedMoESystem:
    """
    Router Agent + Selected Single Agents + Meta Agent를 통합한 완전한 MoE-Invest 시스템
    
    Flow:
    1. Router Agent: 시장 분석 → 적절한 투자 전략 선택
    2. Selected Agents: 선택된 에이전트들만 실행 및 결과 저장  
    3. Meta Analysis: 저장된 결과 파일들을 읽어서 최종 포트폴리오 구성
    """
    
    def __init__(self, llm=None):
        """통합 시스템 초기화"""
        self.llm = llm or ChatOpenAI(temperature=0, model="gpt-4o")
        
        # Router Agent 초기화
        self.router_agent = RouterAgent(llm=self.llm)
        
        # Master Agent 초기화 (메타 분석용)
        self.master_agent = MasterInvestmentAgent(llm=self.llm)
        
        # 사용 가능한 투자 에이전트 정의
        self.available_agents = {
            'WarrenBuffett': {
                'class_name': 'WarrenBuffettInvestmentAnalyzer',
                'module_path': 'agents.WarrenBuffett_agent',
                'description': 'Warren Buffett 가치투자 전략',
                'file_prefix': 'buffett'
            },
            'BenjaminGraham': {
                'class_name': 'GrahamInvestmentAnalyzer',
                'module_path': 'agents.BenjaminGraham_agent', 
                'description': 'Benjamin Graham 순자산 가치투자',
                'file_prefix': 'graham'
            },
            'JosephPiotroski': {
                'class_name': 'PiotroskiInvestmentAnalyzer',
                'module_path': 'agents.JosephPiotroski_agent',
                'description': 'Joseph Piotroski F-Score 재무건전성 분석',
                'file_prefix': 'piotroski'
            },
            'JoelGreenblatt': {
                'class_name': 'GreenblattInvestmentAnalyzer', 
                'module_path': 'agents.JoelGreenblatt_agent',
                'description': 'Joel Greenblatt Magic Formula',
                'file_prefix': 'greenblatt'
            },
            'EdwardAltman': {
                'class_name': 'AltmanInvestmentAnalyzer',
                'module_path': 'agents.EdwardAltman_agent', 
                'description': 'Edward Altman Z-Score 신용위험 분석',
                'file_prefix': 'altman'
            }
        }
    

    
    def _check_agent_results_exist(self, agent_name: str, start_date: str, end_date: str) -> bool:
        """특정 에이전트의 결과 파일이 이미 존재하는지 확인"""
        try:
            agent_info = self.available_agents.get(agent_name)
            if not agent_info:
                return False
            
            file_prefix = agent_info['file_prefix']
            
            # 결과 파일 경로 생성
            analysis_file = f"results/{file_prefix}_agent/{file_prefix}_analysis_{start_date}_{end_date}.json"
            portfolio_file = f"results/{file_prefix}_agent/{file_prefix}_portfolio_{start_date}_{end_date}.csv"
            
            # 두 파일 모두 존재하는지 확인
            analysis_exists = os.path.exists(analysis_file)
            portfolio_exists = os.path.exists(portfolio_file)
            
            return analysis_exists and portfolio_exists
            
        except Exception as e:
            print(f"❌ {agent_name} 파일 존재 확인 실패: {str(e)}")
            return False
    
    def _import_and_create_agent(self, agent_name: str):
        """동적으로 에이전트를 import하고 생성"""
        try:
            agent_info = self.available_agents[agent_name]
            module_name = agent_info['module_path']
            class_name = agent_info['class_name']
            
            # 동적 import
            module = __import__(module_name, fromlist=[class_name])
            agent_class = getattr(module, class_name)
            
            # 에이전트 인스턴스 생성
            return agent_class(llm=self.llm)
            
        except Exception as e:
            print(f"❌ {agent_name} 에이전트 로딩 실패: {str(e)}")
            return None
    
    def run_selected_agents(self, selected_agents: List[str], start_date: str, end_date: str) -> Dict[str, Any]:
        """선택된 에이전트들만 실행 (중복 실행 방지)"""
        print(f"\n🚀 선택된 {len(selected_agents)}개 에이전트 실행 시작...")
        
        agent_results = {}
        successful_agents = []
        skipped_agents = []
        
        for agent_name in selected_agents:
            try:
                # 기존 결과 파일 존재 여부 확인
                if self._check_agent_results_exist(agent_name, start_date, end_date):
                    print(f"\n⏭️  {agent_name} 에이전트: 기존 결과 파일 발견, 실행 건너뛰기")
                    
                    # 기존 결과를 successful로 간주
                    agent_results[agent_name] = {
                        'analysis_result': f"기존 결과 파일 사용: {start_date}_{end_date}",
                        'description': self.available_agents[agent_name]['description'],
                        'status': 'success',
                        'timestamp': datetime.now().isoformat(),
                        'skipped': True
                    }
                    
                    successful_agents.append(agent_name)
                    skipped_agents.append(agent_name)
                    print(f"✅ {agent_name} 기존 결과 사용")
                    continue
                
                print(f"\n⚡ {agent_name} 에이전트 실행 중...")
                
                # 에이전트 동적 로딩 및 생성
                agent = self._import_and_create_agent(agent_name)
                if agent is None:
                    continue
                
                # 에이전트 실행
                result = agent.analyze(start_date, end_date)
                
                agent_results[agent_name] = {
                    'analysis_result': result,
                    'description': self.available_agents[agent_name]['description'],
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'skipped': False
                }
                
                successful_agents.append(agent_name)
                print(f"✅ {agent_name} 새로 실행 완료")
                
            except Exception as e:
                print(f"❌ {agent_name} 실행 실패: {str(e)}")
                agent_results[agent_name] = {
                    'analysis_result': f"실행 오류: {str(e)}",
                    'description': self.available_agents[agent_name]['description'],
                    'status': 'error', 
                    'timestamp': datetime.now().isoformat(),
                    'skipped': False
                }
        
        executed_count = len(successful_agents) - len(skipped_agents)
        print(f"\n📊 에이전트 실행 완료:")
        print(f"   - 새로 실행: {executed_count}개")
        print(f"   - 기존 결과 사용: {len(skipped_agents)}개")
        print(f"   - 실행 실패: {len(selected_agents) - len(successful_agents)}개")
        
        return {
            'agent_results': agent_results,
            'successful_agents': successful_agents,
            'skipped_agents': skipped_agents,
            'execution_summary': {
                'total_selected': len(selected_agents),
                'successful': len(successful_agents),
                'newly_executed': executed_count,
                'skipped_existing': len(skipped_agents),
                'failed': len(selected_agents) - len(successful_agents)
            }
        }
    
    def run_meta_analysis(self, market_analysis: str, selected_agents: List[str], 
                         agent_results: Dict[str, Any], selection_reasons: List[str], 
                         start_date: str, end_date: str) -> Dict[str, Any]:
        """메타 분석 실행 - Master Agent가 저장된 결과 파일들을 읽어서 최종 포트폴리오 구성"""
        print("\n🧠 메타 분석 실행 중... (Master Agent 파일 기반 포트폴리오 구성)")
        
        try:
            # 1. 실행된 에이전트들의 결과 파일 확인
            print("📂 에이전트 결과 파일 가용성 확인...")
            availability = self.master_agent.check_available_results(start_date, end_date)
            
            # 선택되고 성공적으로 실행된 에이전트들만 필터링
            available_selected_agents = []
            for agent_name in selected_agents:
                if (agent_name in agent_results and 
                    agent_results[agent_name]['status'] == 'success' and
                    availability.get(agent_name, False)):
                    available_selected_agents.append(agent_name)
                    print(f"  ✅ {agent_name}: 결과 파일 준비됨")
                else:
                    print(f"  ❌ {agent_name}: 결과 파일 없음 또는 실행 실패")
            
            if not available_selected_agents:
                error_msg = "선택된 에이전트들의 결과 파일을 찾을 수 없습니다."
                print(f"❌ {error_msg}")
                return {
                    'portfolio_analysis': error_msg,
                    'available_agents': available_selected_agents,
                    'router_context': {
                        'market_analysis': market_analysis,
                        'selection_reasons': selection_reasons,
                        'selected_agents': selected_agents
                    }
                }
            
            # 2. Master Agent의 종합 분석 실행 (저장된 파일들 기반)
            print(f"🎯 {len(available_selected_agents)}개 에이전트 결과로 포트폴리오 구성 중...")
            
            comprehensive_result = self.master_agent.run_comprehensive_analysis(
                start_date=start_date, 
                end_date=end_date,
                top_n=10
            )
            
            # 3. Router Agent 컨텍스트와 결합
            meta_result = {
                'portfolio_analysis': comprehensive_result.get('cot_analysis', '포트폴리오 분석 결과 없음'),
                'consensus_analysis': comprehensive_result.get('consensus_analysis', {}),
                'agent_results_summary': comprehensive_result.get('agent_results', {}),
                'available_agents': available_selected_agents,
                'router_context': {
                    'market_analysis': market_analysis,
                    'selection_reasons': selection_reasons,
                    'selected_agents': selected_agents,
                    'execution_summary': f"{len(available_selected_agents)}/{len(selected_agents)} agents successfully analyzed"
                },
                'summary': comprehensive_result.get('summary', 'No summary available')
            }
            
            print("✅ 메타 분석 완료 (Master Agent 파일 기반 포트폴리오)")
            return meta_result
            
        except Exception as e:
            print(f"❌ 메타 분석 실패: {str(e)}")
            return {
                'portfolio_analysis': f"메타 분석 중 오류 발생: {str(e)}",
                'available_agents': [],
                'router_context': {
                    'market_analysis': market_analysis,
                    'selection_reasons': selection_reasons,
                    'selected_agents': selected_agents
                },
                'error': str(e)
            }
    
    def _extract_consensus_from_selected_agents(self, agent_results: Dict[str, Any], selected_agents: List[str], tickers: List[str]) -> str:
        """선택된 에이전트들의 결과에서 합의 종목 추출"""
        try:
            # 각 에이전트의 결과에서 추천 종목들을 추출
            all_recommendations = {}
            
            for agent_name in selected_agents:
                if agent_name in agent_results and agent_results[agent_name]['status'] == 'success':
                    result_text = agent_results[agent_name]['analysis_result']
                    
                    # 결과 텍스트에서 종목들을 파싱 (간단한 방법)
                    # 실제로는 더 정교한 파싱이 필요할 수 있음
                    
                    # 분석 대상 종목들에서 에이전트 결과에 포함된 종목 찾기
                    
                    found_tickers = []
                    for ticker in tickers:  # 분석 대상 종목들에서 찾기
                        if ticker in result_text:
                            found_tickers.append(ticker)
                    
                    all_recommendations[agent_name] = found_tickers
            
            # 합의 종목 생성
            consensus_text = "## Top Consensus Stocks from Selected Agents:\n\n"
            
            # 종목별 추천 에이전트 수 계산
            ticker_counts = {}
            for agent_name, tickers in all_recommendations.items():
                for ticker in tickers:
                    if ticker not in ticker_counts:
                        ticker_counts[ticker] = []
                    ticker_counts[ticker].append(agent_name)
            
            # 추천 수가 많은 순으로 정렬
            sorted_tickers = sorted(ticker_counts.items(), key=lambda x: len(x[1]), reverse=True)
            
            for i, (ticker, recommending_agents) in enumerate(sorted_tickers[:10]):
                consensus_rate = len(recommending_agents) / len(selected_agents)
                agents_str = ', '.join(recommending_agents)
                consensus_text += f"{i+1}. **{ticker}**: {consensus_rate:.1%} consensus rate ({agents_str})\n"
            
            if not sorted_tickers:
                consensus_text += "No clear consensus stocks identified from selected agents.\n"
                consensus_text += "Proceeding with individual agent recommendations for portfolio construction.\n"
            
            return consensus_text
            
        except Exception as e:
            return f"Error extracting consensus: {str(e)}\nProceeding with available agent results for analysis."
    
    def run_complete_analysis(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        완전한 통합 분석 실행
        
        Flow:
        1. Router Agent: 시장 분석 → 에이전트 선택
        2. Selected Agents: 선택된 에이전트들만 실행 및 결과 저장
        3. Meta Analysis: 저장된 결과 파일들을 읽어서 최종 포트폴리오 구성
        4. Results: 모든 결과 저장
        """
        
        print("🎯" + "="*60)
        print("  MoE-Invest 통합 시스템 실행")
        print("  Router → Selected Agents → Meta Analysis")
        print("="*64)
        print(f"📅 분석 기간: {start_date} ~ {end_date}")
        
        start_time = datetime.now()
        
        # === 1단계: Router Agent - 시장 분석 & 에이전트 선택 ===
        print("\n" + "="*50)
        print("1️⃣  STEP 1: Router Agent - 시장 분석 & 에이전트 선택")
        print("="*50)
        
        routing_result = self.router_agent.analyze_and_route(start_date, end_date)
        selected_agents = routing_result['selected_agents']
        
        print(f"✅ 선택된 에이전트: {', '.join(selected_agents)}")
        print(f"📋 선택 이유: {'; '.join(routing_result['selection_reasons'])}")
        
        # === 2단계: Selected Agents - 선택된 에이전트들만 실행 및 저장 ===
        print("\n" + "="*50)
        print("2️⃣  STEP 2: Selected Agents - 선택된 전략 실행 및 저장")
        print("="*50)
        
        execution_result = self.run_selected_agents(selected_agents, start_date, end_date)
        
        # === 3단계: Meta Analysis - 파일 기반 포트폴리오 구성 ===
        print("\n" + "="*50)
        print("3️⃣  STEP 3: Meta Analysis - 저장된 결과로 포트폴리오 구성")
        print("="*50)
        
        meta_analysis = self.run_meta_analysis(
            market_analysis=routing_result['market_analysis'],
            selected_agents=selected_agents,
            agent_results=execution_result['agent_results'],
            selection_reasons=routing_result['selection_reasons'],
            start_date=start_date,
            end_date=end_date
        )
        
        # === 4단계: Results - 통합 결과 생성 및 저장 ===
        execution_time = (datetime.now() - start_time).total_seconds()
        
        final_result = {
            'timestamp': datetime.now().isoformat(),
            'execution_time_seconds': execution_time,
            'input_parameters': {
                'start_date': start_date,
                'end_date': end_date
            },
            'step1_routing': routing_result,
            'step2_execution': execution_result,
            'step3_meta_analysis': meta_analysis,
            'summary': {
                'total_agents_available': len(self.available_agents),
                'agents_selected_by_router': len(selected_agents),
                'agents_executed_successfully': execution_result['execution_summary']['successful'],
                'final_recommendation': 'Generated' if meta_analysis and 'error' not in meta_analysis else 'Failed'
            }
        }
        
        # 결과 저장 (JSON, Markdown, CSV)
        self._save_integrated_results(final_result, start_date, end_date)
        
        # 최종 요약 출력
        print("\n" + "🎉" + "="*60)
        print(f"  통합 분석 완료! (실행시간: {execution_time:.2f}초)")
        print("="*64)
        print(f"📈 Router 선택: {len(selected_agents)}개 에이전트")
        print(f"✅ 총 성공: {execution_result['execution_summary']['successful']}개")
        print(f"🔄 새로 실행: {execution_result['execution_summary']['newly_executed']}개")
        print(f"⏭️  기존 결과 사용: {execution_result['execution_summary']['skipped_existing']}개")
        print(f"🧠 포트폴리오 구성: {'완료' if final_result['summary']['final_recommendation'] == 'Generated' else '실패'}")
        
        return final_result
    
    def _save_integrated_results(self, results: Dict[str, Any], start_date: str, end_date: str):
        """통합 분석 결과 저장 (JSON, Markdown, CSV)"""
        try:
            # 결과 저장 디렉토리 생성
            os.makedirs("results/meta_analysis", exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # JSON 결과 저장
            json_filename = f"results/meta_analysis/meta_analysis_{start_date}_{end_date}.json"
            with open(json_filename, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
            # CSV 포트폴리오 저장
            csv_filename = f"results/meta_analysis/meta_portfolio_{start_date}_{end_date}.csv"
            self._save_portfolio_csv(results, csv_filename)
            
            # 종합 리포트 저장 (Markdown)
            report_filename = f"results/meta_analysis/meta_report_{start_date}_{end_date}.md"
            with open(report_filename, "w", encoding="utf-8") as f:
                f.write("# MoE-Invest 통합 시스템 분석 리포트\n\n")
                f.write(f"**분석 시간**: {results['timestamp']}\n")
                f.write(f"**실행 시간**: {results['execution_time_seconds']:.2f}초\n")
                f.write(f"**분석 기간**: {start_date} ~ {end_date}\n")
                
                # Step 1: Router Analysis
                f.write("## 1️⃣ Router Agent 분석\n\n")
                f.write("### 시장 분석 결과\n")
                f.write(results['step1_routing']['market_analysis'])
                f.write("\n\n### 선택된 에이전트\n")
                for agent in results['step1_routing']['selected_agents']:
                    details = results['step1_routing']['agent_details'].get(agent, {})
                    f.write(f"- **{agent}**: {details.get('description', 'N/A')}\n")
                f.write(f"\n### 선택 이유\n")
                for reason in results['step1_routing']['selection_reasons']:
                    f.write(f"- {reason}\n")
                
                # Step 2: Agent Execution
                f.write("\n## 2️⃣ 선택된 에이전트 실행 결과\n\n")
                execution_summary = results['step2_execution']['execution_summary']
                f.write(f"**실행 요약**:\n")
                f.write(f"- 총 선택 에이전트: {execution_summary['total_selected']}개\n")
                f.write(f"- 성공 (총): {execution_summary['successful']}개\n")
                f.write(f"- 새로 실행: {execution_summary['newly_executed']}개\n")
                f.write(f"- 기존 결과 사용: {execution_summary['skipped_existing']}개\n")
                f.write(f"- 실행 실패: {execution_summary['failed']}개\n\n")
                
                for agent_name, result in results['step2_execution']['agent_results'].items():
                    f.write(f"### {agent_name}\n")
                    f.write(f"- **상태**: {result['status']}\n")
                    f.write(f"- **설명**: {result['description']}\n")
                    if result['status'] == 'success':
                        if result.get('skipped', False):
                            f.write(f"- **결과**: 기존 결과 파일 사용\n")
                        else:
                            f.write(f"- **결과**: 새로 실행 완료\n")
                    else:
                        f.write(f"- **오류**: {result['analysis_result']}\n")
                    f.write("\n")
                
                # Step 3: Meta Analysis
                f.write("## 3️⃣ 메타 분석 및 최종 포트폴리오 구성\n\n")
                meta_data = results['step3_meta_analysis']
                
                if isinstance(meta_data, dict):
                    # Router Context
                    if 'router_context' in meta_data:
                        f.write("### Router Agent 분석 컨텍스트\n")
                        f.write(f"**선택된 에이전트**: {', '.join(meta_data['router_context']['selected_agents'])}\n")
                        f.write(f"**선택 이유**: {'; '.join(meta_data['router_context']['selection_reasons'])}\n\n")
                    
                    # Portfolio Analysis
                    f.write("### 최종 포트폴리오 구성 결과\n")
                    f.write(meta_data.get('portfolio_analysis', '포트폴리오 분석 결과 없음'))
                    f.write("\n\n")
                    
                    # Available Agents Summary
                    if 'available_agents' in meta_data:
                        f.write("### 포트폴리오 구성에 사용된 에이전트\n")
                        for agent in meta_data['available_agents']:
                            f.write(f"- {agent}\n")
                        f.write("\n")
                    
                    # Summary
                    if 'summary' in meta_data:
                        f.write("### 포트폴리오 매니저 요약\n")
                        f.write(meta_data['summary'])
                        f.write("\n")
                else:
                    f.write(str(meta_data))
                
                # Summary
                f.write("\n## 📊 실행 요약\n\n")
                summary = results['summary']
                f.write(f"- **전체 에이전트 수**: {summary['total_agents_available']}개\n")
                f.write(f"- **Router 선택 에이전트**: {summary['agents_selected_by_router']}개\n")
                f.write(f"- **성공적 실행**: {summary['agents_executed_successfully']}개\n")
                f.write(f"- **최종 추천**: {summary['final_recommendation']}\n")
            
            print(f"\n💾 통합 시스템 결과 저장 완료:")
            print(f"   📄 JSON: {json_filename}")
            print(f"   📝 Report: {report_filename}")
            print(f"   📊 Portfolio CSV: {csv_filename}")
            
        except Exception as e:
            print(f"❌ 결과 저장 실패: {str(e)}")
    
    def _save_portfolio_csv(self, results: Dict[str, Any], csv_filename: str):
        """포트폴리오를 CSV 형식으로 저장 (Piotroski 형식 참고)"""
        try:
            # 메타 분석 결과에서 포트폴리오 정보 추출
            meta_analysis = results.get('step3_meta_analysis', {})
            
            # 합의 분석 또는 포트폴리오 분석에서 종목 정보 추출
            portfolio_stocks = []
            
            # 1. 합의 분석에서 종목 추출 시도
            consensus_analysis = meta_analysis.get('consensus_analysis', {})
            if consensus_analysis and 'consensus_stocks' in consensus_analysis:
                consensus_stocks = consensus_analysis['consensus_stocks']
                total_stocks = len(consensus_stocks)
                
                for i, (ticker, stock_data) in enumerate(consensus_stocks.items()):
                    consensus_rate = stock_data.get('consensus_rate', 0.0)
                    agent_count = stock_data.get('agent_count', 0)
                    recommending_agents = stock_data.get('recommending_agents', [])
                    
                    # 기본 균등 비중 할당, 합의도가 높은 종목에 약간 더 높은 비중
                    base_weight = 100.0 / total_stocks
                    consensus_bonus = consensus_rate * 20  # 합의도 20% 가중치
                    weight = base_weight + consensus_bonus
                    
                    portfolio_stocks.append({
                        'ticker': ticker,
                        'score': round(consensus_rate, 3),
                        'weight': round(weight, 1),
                        'reason': f'Consensus from {agent_count} agents: {", ".join(recommending_agents)}'
                    })
                
                # 비중 정규화 (총합 100%로)
                if portfolio_stocks:
                    total_weight = sum(s['weight'] for s in portfolio_stocks)
                    if total_weight > 0:
                        for stock in portfolio_stocks:
                            stock['weight'] = round(stock['weight'] * 100.0 / total_weight, 1)
            
            # 2. 메타 분석 텍스트에서 종목 정보 파싱 시도
            if not portfolio_stocks:
                portfolio_text = meta_analysis.get('portfolio_analysis', '')
                portfolio_stocks = self._parse_portfolio_from_text(portfolio_text, results)
            
            # 3. 선택된 에이전트들의 결과에서 종목 추출 (fallback)
            if not portfolio_stocks:
                portfolio_stocks = self._extract_portfolio_from_agents(results)
            
            # CSV 파일 생성
            if portfolio_stocks:
                df = pd.DataFrame(portfolio_stocks)
                # 컬럼 순서 조정 (Piotroski 형식에 맞춰)
                df = df[['ticker', 'score', 'weight', 'reason']]
                df.columns = ['Ticker', 'Score', 'Weight (%)', 'Reason']
                
                # CSV 저장
                df.to_csv(csv_filename, index=False, encoding='utf-8')
                print(f"✅ 포트폴리오 CSV 저장 완료: {len(portfolio_stocks)}개 종목")
            else:
                # 빈 템플릿 생성
                empty_df = pd.DataFrame(columns=['Ticker', 'Score', 'Weight (%)', 'Reason'])
                empty_df.to_csv(csv_filename, index=False, encoding='utf-8')
                print("⚠️  포트폴리오 종목 정보 없음 - 빈 CSV 템플릿 생성")
                
        except Exception as e:
            print(f"❌ CSV 저장 실패: {str(e)}")
            # 오류 시 빈 템플릿이라도 생성
            try:
                empty_df = pd.DataFrame(columns=['Ticker', 'Score', 'Weight (%)', 'Reason'])
                empty_df.to_csv(csv_filename, index=False, encoding='utf-8')
            except:
                pass
    
    def _parse_portfolio_from_text(self, portfolio_text: str, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """포트폴리오 분석 텍스트에서 종목 정보 파싱"""
        stocks = []
        try:
            # 메타 분석에서 발견된 종목들 추출
            meta_analysis = results.get('step3_meta_analysis', {})
            consensus_stocks = meta_analysis.get('consensus_analysis', {}).get('consensus_stocks', {})
            input_tickers = list(consensus_stocks.keys()) if consensus_stocks else []
            
            # 텍스트에서 종목 찾기 및 점수/비중 추정
            for i, ticker in enumerate(input_tickers):
                if ticker in portfolio_text:
                    # 기본 점수 및 비중 할당 (실제 구현에서는 더 정교한 파싱 필요)
                    base_score = 0.8  # 기본 점수
                    base_weight = 100.0 / len(input_tickers)  # 균등 비중
                    
                    # 텍스트에서 긍정적/부정적 언급 확인
                    ticker_context = self._extract_ticker_context(portfolio_text, ticker)
                    
                    # 점수 조정
                    if any(word in ticker_context.lower() for word in ['강력', 'strong', 'recommend', '추천', '우수']):
                        score = min(0.95, base_score + 0.15)
                        weight_multiplier = 1.2
                    elif any(word in ticker_context.lower() for word in ['약함', 'weak', '주의', 'caution', '위험']):
                        score = max(0.3, base_score - 0.2)
                        weight_multiplier = 0.8
                    else:
                        score = base_score
                        weight_multiplier = 1.0
                    
                    stocks.append({
                        'ticker': ticker,
                        'score': round(score, 2),
                        'weight': round(base_weight * weight_multiplier, 1),
                        'reason': f'Meta analysis consensus ({len(results.get("step1_routing", {}).get("selected_agents", []))} agents)'
                    })
            
            # 비중 정규화 (총합 100%로)
            if stocks:
                total_weight = sum(s['weight'] for s in stocks)
                if total_weight > 0:
                    for stock in stocks:
                        stock['weight'] = round(stock['weight'] * 100.0 / total_weight, 1)
            
        except Exception as e:
            print(f"포트폴리오 텍스트 파싱 오류: {e}")
        
        return stocks
    
    def _extract_ticker_context(self, text: str, ticker: str, context_length: int = 200) -> str:
        """텍스트에서 특정 종목 주변 문맥 추출"""
        try:
            ticker_pos = text.find(ticker)
            if ticker_pos == -1:
                return ""
            
            start = max(0, ticker_pos - context_length)
            end = min(len(text), ticker_pos + len(ticker) + context_length)
            
            return text[start:end]
        except:
            return ""
    
    def _extract_portfolio_from_agents(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """에이전트 결과에서 포트폴리오 추출 (fallback)"""
        stocks = []
        try:
            # 성공적으로 실행된 에이전트들
            successful_agents = results.get('step2_execution', {}).get('successful_agents', [])
            # 메타 분석에서 발견된 종목들 추출
            meta_analysis = results.get('step3_meta_analysis', {})
            consensus_stocks = meta_analysis.get('consensus_analysis', {}).get('consensus_stocks', {})
            input_tickers = list(consensus_stocks.keys()) if consensus_stocks else []
            
            if not successful_agents or not input_tickers:
                return stocks
            
            # 각 종목에 대한 기본 정보 생성
            for i, ticker in enumerate(input_tickers):
                stocks.append({
                    'ticker': ticker,
                    'score': round(0.7 + (i % 3) * 0.1, 2),  # 0.7-0.9 범위
                    'weight': round(100.0 / len(input_tickers), 1),  # 균등 비중
                    'reason': f'Selected by {len(successful_agents)} agents: {", ".join(successful_agents[:2])}{"..." if len(successful_agents) > 2 else ""}'
                })
                
        except Exception as e:
            print(f"에이전트 결과에서 포트폴리오 추출 오류: {e}")
        
        return stocks


if __name__ == "__main__":

    print("🔧 MoE-Invest 통합 시스템 테스트")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    system = IntegratedMoESystem(llm=llm)

    # 분기별 달력 날짜 범위 생성
    def quarter_date_range(year: int, q: int):
        # 시작일
        start_month = (q - 1) * 3 + 1
        start_date = f"{year}-{start_month:02d}-01"
        # 종료일(달의 마지막 날은 고정)
        quarter_end = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}[q]
        end_date = f"{year}-{quarter_end}"
        return start_date, end_date

    # 루프 실행 + 결과 패널 저장
    def run_three_years_with_dates(agent, years=(2025, 2025)):
        for y in years:
            for q in (1, 2):
                s, e = quarter_date_range(y, q)
                print(f"▶ {y}Q{q}: {s} ~ {e}")
                try:
                    out = agent.run_complete_analysis(s, e)
                except:
                    print(f"Error: {y}Q{q}")
                    continue

    run_three_years_with_dates(system)


