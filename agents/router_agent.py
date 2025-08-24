"""
Router Agent for MoE-Invest System
시장 분석 후 적절한 투자 에이전트를 선택하고 handoff하는 라우터 에이전트
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain.schema import SystemMessage, HumanMessage

# 기술적 지표 계산용
try:
    import talib
except ImportError:
    print("Warning: TA-Lib not installed. Using basic calculations.")
    talib = None


class RouterAgent:
    """
    시장 분석 후 적절한 투자 에이전트를 선택하는 라우터 에이전트
    """
    
    def __init__(self, llm=None):
        """Router Agent 초기화"""
        self.llm = llm or ChatOpenAI(temperature=0, model="gpt-4o")
        
        # 분석할 시장 지수들 (대표 ETF/Index)
        self.market_indices = {
            'SPY': 'S&P 500 ETF',
            'QQQ': 'NASDAQ 100 ETF', 
            'IWM': 'Russell 2000 ETF',
            'VIX': 'Volatility Index',
            'TLT': '20+ Year Treasury Bond ETF',
            'GLD': 'Gold ETF'
        }
        
        # 투자 에이전트별 특성 정의
        self.agent_characteristics = {
            'WarrenBuffett': {
                'style': 'Value',
                'market_preference': ['bull', 'stable'],
                'volatility_preference': 'low',
                'description': 'Warren Buffett 가치투자 전략 - 우량주, 저평가, 장기투자',
                'best_conditions': ['경제 안정기', '시장 과열 후 조정기', '우량주 저평가 시기']
            },
            'BenjaminGraham': {
                'style': 'Deep Value',
                'market_preference': ['bear', 'sideways'],
                'volatility_preference': 'high',
                'description': 'Benjamin Graham 순자산 가치투자 - 극도 저평가 종목',
                'best_conditions': ['약세장', '시장 공황기', '유동성 위기 시기']
            },
            'JosephPiotroski': {
                'style': 'Quality',
                'market_preference': ['bear', 'volatile'],
                'volatility_preference': 'medium',
                'description': 'Joseph Piotroski F-Score 재무건전성 분석',
                'best_conditions': ['시장 불안정기', '신용 위기 시기', '재무건전성 중요 시기']
            },
            'JoelGreenblatt': {
                'style': 'Magic Formula',
                'market_preference': ['bull', 'recovery'],
                'volatility_preference': 'medium',
                'description': 'Joel Greenblatt Magic Formula - 수익률과 밸류에이션 균형',
                'best_conditions': ['경기 회복기', '시장 전환기', '중소형주 선호 시기']
            },
            'EdwardAltman': {
                'style': 'Credit Risk',
                'market_preference': ['bear', 'crisis'],
                'volatility_preference': 'high',
                'description': 'Edward Altman Z-Score 신용위험 분석',
                'best_conditions': ['경기 침체기', '신용 경색 시기', '부실기업 정리 시기']
            }
        }
        
        # 분석 기간 초기화
        self._analysis_start_date = None
        self._analysis_end_date = None
        
        # 도구 설정
        self.tools = self._create_tools()
        
        # 에이전트 생성
        self.agent = self._create_agent()
    
    def _create_tools(self) -> List[Tool]:
        """라우터 에이전트가 사용할 도구들 생성"""
        
        def get_market_data(symbols: str = "SPY,QQQ,IWM,TLT,GLD", period: str = "6mo", start_date: str = None, end_date: str = None) -> str:
            """시장 지수 데이터 조회"""
            try:
                symbol_list = [s.strip() for s in symbols.split(',')]
                results = {}
                
                for symbol in symbol_list:
                    ticker = yf.Ticker(symbol)
                    if start_date and end_date:
                        hist = ticker.history(start=start_date, end=end_date)
                    else:
                        hist = ticker.history(period=period)
                    if not hist.empty:
                        latest = hist.iloc[-1]
                        first = hist.iloc[0]
                        
                        # 수익률 계산
                        returns = (latest['Close'] / first['Close'] - 1) * 100
                        
                        # 변동성 계산 (20일 기준)
                        volatility = hist['Close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100
                        
                        results[symbol] = {
                            'current_price': latest['Close'],
                            'period_return': returns,
                            'volatility_annualized': volatility,
                            'avg_volume': hist['Volume'].mean(),
                            'latest_date': hist.index[-1].strftime('%Y-%m-%d')
                        }
                
                return json.dumps(results, indent=2)
                
            except Exception as e:
                return f"Error getting market data: {str(e)}"
        
        def calculate_technical_indicators(symbol: str = "SPY", period: str = "6mo", start_date: str = None, end_date: str = None) -> str:
            """기술적 지표 계산"""
            try:
                ticker = yf.Ticker(symbol)
                if start_date and end_date:
                    hist = ticker.history(start=start_date, end=end_date)
                else:
                    hist = ticker.history(period=period)
                
                if hist.empty:
                    return f"No data available for {symbol}"
                
                close_prices = hist['Close'].values
                high_prices = hist['High'].values
                low_prices = hist['Low'].values
                volume = hist['Volume'].values
                
                results = {}
                
                # RSI 계산
                if talib:
                    results['RSI'] = float(talib.RSI(close_prices)[-1])
                    results['MACD'], results['MACD_signal'], results['MACD_hist'] = talib.MACD(close_prices)
                    results['MACD'] = float(results['MACD'][-1])
                    results['MACD_signal'] = float(results['MACD_signal'][-1])
                    results['MACD_hist'] = float(results['MACD_hist'][-1])
                    
                    # Bollinger Bands
                    bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices)
                    results['BB_upper'] = float(bb_upper[-1])
                    results['BB_middle'] = float(bb_middle[-1])
                    results['BB_lower'] = float(bb_lower[-1])
                    
                    # ATR
                    results['ATR'] = float(talib.ATR(high_prices, low_prices, close_prices)[-1])
                else:
                    # 기본 계산 (TA-Lib 없을 때)
                    # RSI 계산
                    delta = pd.Series(close_prices).diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    results['RSI'] = float(100 - (100 / (1 + rs.iloc[-1])))
                
                # 이동평균
                results['SMA_20'] = float(pd.Series(close_prices).rolling(20).mean().iloc[-1])
                results['SMA_50'] = float(pd.Series(close_prices).rolling(50).mean().iloc[-1])
                results['SMA_200'] = float(pd.Series(close_prices).rolling(200).mean().iloc[-1])
                
                # 현재 가격 대비 이동평균 위치
                current_price = close_prices[-1]
                results['price_vs_SMA20'] = (current_price / results['SMA_20'] - 1) * 100
                results['price_vs_SMA50'] = (current_price / results['SMA_50'] - 1) * 100
                results['price_vs_SMA200'] = (current_price / results['SMA_200'] - 1) * 100
                
                return json.dumps(results, indent=2)
                
            except Exception as e:
                return f"Error calculating technical indicators: {str(e)}"
        
        def analyze_market_regime(start_date: str = None, end_date: str = None) -> str:
            """시장 체제 분석"""
            try:
                # 주요 지수들의 상관관계와 트렌드 분석 (VIX 포함)
                indices = ['SPY', 'QQQ', 'IWM', '^VIX']
                data = {}
                
                for symbol in indices:
                    ticker = yf.Ticker(symbol)
                    if start_date and end_date:
                        # 사용자 지정 기간 사용
                        hist = ticker.history(start=start_date, end=end_date)
                    else:
                        # 기본 3개월 데이터
                        hist = ticker.history(period="3mo")
                    
                    if not hist.empty:
                        returns = hist['Close'].pct_change().dropna()
                        data[symbol] = returns
                
                if not data:
                    return "No data available for market regime analysis"
                
                # 데이터프레임으로 변환
                df = pd.DataFrame(data)
                
                # 상관관계 계산
                correlation_matrix = df.corr()
                
                # 트렌드 강도 계산 (최근 분석 기간 내 수익률)
                recent_returns = {}
                analysis_days = min(30, len(df))  # 최대 30일 또는 데이터 길이
                
                for symbol in indices:
                    if symbol in df.columns:
                        period_return = df[symbol].tail(analysis_days).sum() * 100
                        recent_returns[symbol] = period_return
                
                # 변동성 수준 분석 (실제 VIX 데이터 사용)
                if '^VIX' in df.columns:
                    current_vix = df['^VIX'].iloc[-1] if not df['^VIX'].empty else 20
                    volatility_level = "low" if current_vix < 15 else "high" if current_vix > 25 else "medium"
                    vix_data = {'current_vix': current_vix}
                else:
                    # VIX 데이터가 없으면 SPY 변동성으로 대체
                    spy_volatility = df['SPY'].std() * np.sqrt(252) * 100 if 'SPY' in df.columns else 20
                    volatility_level = "low" if spy_volatility < 15 else "high" if spy_volatility > 25 else "medium"
                    vix_data = {'spy_volatility_annualized': spy_volatility}
                
                analysis = {
                    'correlations': correlation_matrix.to_dict(),
                    'recent_returns': recent_returns,
                    'volatility_regime': volatility_level,
                    'volatility_data': vix_data,
                    'market_breadth': 'strong' if recent_returns.get('IWM', 0) > recent_returns.get('SPY', 0) else 'weak',
                    'analysis_period': f"{analysis_days} days"
                }
                
                return json.dumps(analysis, indent=2, default=str)
                
            except Exception as e:
                return f"Error in market regime analysis: {str(e)}"
        
        def get_economic_indicators(start_date: str = None, end_date: str = None) -> str:
            """경제 지표 조회 (간접적으로 시장에서 파악)"""
            try:
                # 채권, 달러, 원자재 ETF를 통한 경제 상황 파악
                indicators = {
                    'TLT': '장기 국채 (금리 환경)',
                    'UUP': '달러 강도',
                    'GLD': '금 (안전자산)',
                    'DJP': '원자재',
                    'XLF': '금융섹터'
                }
                
                results = {}
                
                for symbol, description in indicators.items():
                    try:
                        ticker = yf.Ticker(symbol)
                        if start_date and end_date:
                            hist = ticker.history(start=start_date, end=end_date)
                        else:
                            hist = ticker.history(period="3mo")
                        if not hist.empty:
                            returns_3m = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100
                            volatility = hist['Close'].pct_change().std() * np.sqrt(252) * 100
                            
                            results[symbol] = {
                                'description': description,
                                '3month_return': returns_3m,
                                'volatility': volatility
                            }
                    except:
                        continue
                
                return json.dumps(results, indent=2)
                
            except Exception as e:
                return f"Error getting economic indicators: {str(e)}"
        
        def select_agents_for_market(market_analysis: str) -> str:
            """시장 분석 결과를 바탕으로 적절한 에이전트 선택 (LLM 기반)"""
            try:
                # LLM이 시장 분석을 바탕으로 직접 에이전트를 선택하도록 개선
                agent_selection_prompt = f"""
다음 시장 분석 결과를 바탕으로 가장 적합한 투자 에이전트 2-5명을 선택해주세요.

시장 분석 결과:
{market_analysis}

사용 가능한 투자 에이전트:
1. WarrenBuffett: 가치투자 전략, 우량주 장기투자, 안정적 시장에서 효과적
   - 적합한 상황: 경제 안정기, 시장 과열 후 조정기, 우량주 저평가 시기
   
2. BenjaminGraham: 순자산 가치투자, 극도 저평가 종목 발굴, 약세장에서 강점
   - 적합한 상황: 약세장, 시장 공황기, 유동성 위기 시기
   
3. JosephPiotroski: F-Score 재무건전성 분석, 질적 분석 중심, 불안정한 시장에서 유용
   - 적합한 상황: 시장 불안정기, 신용 위기 시기, 재무건전성 중요 시기
   
4. JoelGreenblatt: Magic Formula, 수익률과 밸류에이션 균형, 중소형주 강점
   - 적합한 상황: 경기 회복기, 시장 전환기, 중소형주 선호 시기
   
5. EdwardAltman: Z-Score 신용위험 분석, 부실 위험 평가, 위기 상황에서 필수
   - 적합한 상황: 경기 침체기, 신용 경색 시기, 부실기업 정리 시기

선택 기준:
- 현재 시장의 변동성 수준
- 경제 환경 및 트렌드 방향
- 금리 환경 및 유동성 상황
- 섹터별 강약 패턴
- 투자자 심리 및 리스크 선호도

반드시 다음 JSON 형식으로 응답해주세요:
{{
  "selected_agents": ["에이전트1", "에이전트2", ...],
  "reasons": ["선택 이유1", "선택 이유2", ...]
}}

주의사항:
- 최소 2명, 최대 5명 선택
- 각 에이전트의 특성과 현재 시장 상황의 적합성을 고려
- 서로 다른 관점을 제공할 수 있는 에이전트들의 조합 고려
- 선택 이유는 구체적이고 분석 결과에 근거해야 함
"""
                
                # LLM을 통해 에이전트 선택
                response = self.llm.invoke([
                    SystemMessage(content="당신은 투자 전문가로서 시장 분석 결과를 바탕으로 최적의 투자 에이전트 조합을 선택하는 역할을 합니다."),
                    HumanMessage(content=agent_selection_prompt)
                ])
                
                # LLM 응답에서 JSON 추출
                response_text = response.content
                
                # JSON 부분만 추출 시도
                try:
                    # JSON 블록 찾기 (```json 또는 { 시작)
                    if "```json" in response_text:
                        start_idx = response_text.find("```json") + 7
                        end_idx = response_text.find("```", start_idx)
                        json_text = response_text[start_idx:end_idx].strip()
                    elif "{" in response_text:
                        start_idx = response_text.find("{")
                        # 마지막 } 찾기
                        end_idx = response_text.rfind("}") + 1
                        json_text = response_text[start_idx:end_idx].strip()
                    else:
                        raise ValueError("JSON 형식을 찾을 수 없습니다")
                    
                    # JSON 파싱
                    llm_selection = json.loads(json_text)
                    selected_agents = llm_selection.get('selected_agents', [])
                    reasons = llm_selection.get('reasons', [])
                    
                    # 유효성 검사
                    valid_agents = list(self.agent_characteristics.keys())
                    selected_agents = [agent for agent in selected_agents if agent in valid_agents]
                    
                    # 최소 2명, 최대 5명 보장
                    if len(selected_agents) < 2:
                        selected_agents = ['WarrenBuffett', 'JosephPiotroski']
                        reasons = ['LLM 선택 실패로 인한 기본 조합: 가치투자 + 재무건전성 분석']
                    elif len(selected_agents) > 5:
                        selected_agents = selected_agents[:5]
                        reasons = reasons[:5] if len(reasons) >= 5 else reasons
                    
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    print(f"LLM 응답 파싱 실패: {e}")
                    print(f"응답 내용: {response_text}")
                    # 파싱 실패 시 기본값 사용
                    selected_agents = ['WarrenBuffett', 'JosephPiotroski']
                    reasons = ['LLM 응답 파싱 실패로 인한 기본 조합']
                
                # 중복 제거
                selected_agents = list(dict.fromkeys(selected_agents))  # 순서 유지하면서 중복 제거
                
                result = {
                    'selected_agents': selected_agents,
                    'reasons': reasons,
                    'agent_details': {
                        agent: self.agent_characteristics.get(agent, {})
                        for agent in selected_agents
                    },
                    'llm_response': response_text  # 디버깅용
                }
                
                return json.dumps(result, indent=2)
                
            except Exception as e:
                print(f"에이전트 선택 중 오류: {e}")
                # 오류 시 기본 조합 반환
                result = {
                    'selected_agents': ['WarrenBuffett', 'JosephPiotroski'],
                    'reasons': [f'오류로 인한 기본 조합: {str(e)}'],
                    'agent_details': {
                        'WarrenBuffett': self.agent_characteristics.get('WarrenBuffett', {}),
                        'JosephPiotroski': self.agent_characteristics.get('JosephPiotroski', {})
                    }
                }
                return json.dumps(result, indent=2)
        
        # 도구들을 래핑해서 기간 정보를 자동으로 전달
        def get_market_data_wrapper(symbols: str = "SPY,QQQ,IWM,TLT,GLD,^VIX") -> str:
            # VIX를 ^VIX로 변환 (Yahoo Finance 형식)
            symbol_list = [s.strip() for s in symbols.split(',')]
            converted_symbols = ['^VIX' if s.upper() in ['VIX', '$VIX'] else s for s in symbol_list]
            clean_symbols = ','.join(converted_symbols)
            return get_market_data(clean_symbols, start_date=self._analysis_start_date, end_date=self._analysis_end_date)
        
        def calculate_technical_indicators_wrapper(symbol: str = "SPY") -> str:
            # 여러 종목이 쉼표로 구분되어 들어오면 첫 번째 종목만 사용
            if ',' in symbol:
                symbol = symbol.split(',')[0].strip()
            return calculate_technical_indicators(symbol, start_date=self._analysis_start_date, end_date=self._analysis_end_date)
        
        def analyze_market_regime_wrapper(unused_param: str = "") -> str:
            # 파라미터는 무시하고 사전 설정된 기간으로 분석
            return analyze_market_regime(start_date=self._analysis_start_date, end_date=self._analysis_end_date)
        
        def get_economic_indicators_wrapper(unused_param: str = "") -> str:
            # 파라미터는 무시하고 사전 설정된 기간으로 분석
            return get_economic_indicators(start_date=self._analysis_start_date, end_date=self._analysis_end_date)
        
        return [
            Tool(
                name="get_market_data",
                description="시장 지수들의 현재 데이터를 조회합니다. symbols는 쉼표로 구분 (예: SPY,QQQ,IWM,TLT,GLD,VIX)",
                func=get_market_data_wrapper
            ),
            Tool(
                name="calculate_technical_indicators", 
                description="단일 종목의 기술적 지표를 계산합니다 (RSI, MACD, 볼린저밴드 등). 반드시 하나의 종목만 입력하세요 (예: SPY)",
                func=calculate_technical_indicators_wrapper
            ),
            Tool(
                name="analyze_market_regime",
                description="시장 체제를 분석합니다 (상관관계, 트렌드 강도, 변동성 수준 등). 빈 문자열로 호출하세요",
                func=analyze_market_regime_wrapper
            ),
            Tool(
                name="get_economic_indicators",
                description="경제 지표들을 간접적으로 파악합니다 (채권, 달러, 원자재 등). 빈 문자열로 호출하세요",
                func=get_economic_indicators_wrapper
            ),
            Tool(
                name="select_agents_for_market",
                description="시장 분석 결과를 바탕으로 적절한 투자 에이전트들을 선택합니다",
                func=select_agents_for_market
            )
        ]
    
    def _create_agent(self) -> AgentExecutor:
        """Router Agent 생성"""
        
        system_prompt = """당신은 MoE-Invest 시스템의 Router Agent입니다. 
        
주요 역할:
1. 현재 시장 상황을 종합적으로 분석합니다
2. 시장 상황에 가장 적합한 투자 에이전트 2-5명을 지능적으로 선택합니다
3. 선택 이유를 명확하고 논리적으로 설명합니다

분석 과정:
1. 주요 시장 지수 데이터 조회 (SPY, QQQ, IWM, TLT, GLD, VIX 등)
2. 기술적 지표 분석 (RSI, MACD, 이동평균, 볼린저밴드 등) - SPY 기준
3. 시장 체제 분석 (상관관계, 변동성 수준, 트렌드 강도, 시장 폭)
4. 경제 환경 파악 (채권, 달러, 원자재, 금융섹터)
5. 시장 분석 데이터를 종합하여 최적 에이전트 조합 선택

도구 사용 주의사항:
- get_market_data: 주요 지수 사용 (예: SPY,QQQ,IWM,TLT,GLD,VIX)
- calculate_technical_indicators: 반드시 단일 종목만 분석 (예: SPY)
- analyze_market_regime: 빈 문자열 ""로 호출
- get_economic_indicators: 빈 문자열 ""로 호출
- select_agents_for_market: 시장 분석 결과를 종합한 텍스트 전달

사용 가능한 투자 에이전트:
- WarrenBuffett: 가치투자 전략, 우량주 장기투자, 안정적 시장에서 효과적
- BenjaminGraham: 순자산 가치투자, 극도 저평가 종목 발굴, 약세장/위기시 강점  
- JosephPiotroski: F-Score 재무건전성 분석, 질적 분석 중심, 불안정한 시장에서 유용
- JoelGreenblatt: Magic Formula, 수익률과 밸류에이션 균형, 경기 회복기에 적합
- EdwardAltman: Z-Score 신용위험 분석, 부실 위험 평가, 경기 침체/위기시 필수

에이전트 선택 전략:
- 시장 변동성과 트렌드 방향을 고려
- 서로 다른 투자 철학과 관점을 제공하는 조합 선택
- 현재 경제/금융 환경에 특화된 전문성 활용
- 리스크 관리와 수익 창출의 균형 고려
- 최소 2명, 최대 5명까지 선택 가능

중요 지침:
- 모든 분석은 사용자 지정 기간에 맞춰 수행
- 데이터 기반의 논리적 에이전트 선택 필수
- 각 에이전트 선택에 대한 명확한 근거 제시
- 시장 상황 변화에 유연하게 대응할 수 있는 조합 구성
- 마지막에 반드시 select_agents_for_market 도구를 사용하여 최종 선택 확정"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])
        
        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            return_intermediate_steps=True,
            max_iterations=10
        )
    
    def analyze_and_route(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        시장 분석 후 적절한 에이전트들을 선택하고 실행
        
        Args:
            start_date: 분석 시작일
            end_date: 분석 종료일
            
        Returns:
            분석 결과와 선택된 에이전트 정보
        """
        
        print("🔍 Router Agent: 시장 분석 및 에이전트 선택을 시작합니다...")
        
        # 분석 기간 설정 (도구들이 사용할 수 있도록)
        self._analysis_start_date = start_date
        self._analysis_end_date = end_date
        
        # 현재 시장 상황 분석 요청
        request = f"""
현재 시장 상황을 분석하고 투자 분석에 적합한 에이전트들을 선택해주세요.

분석 대상:
- 기간: {start_date} ~ {end_date}

다음 순서로 분석해주세요:
1. 주요 시장 지수 현황 파악
2. 기술적 지표 분석
3. 시장 체제 분석  
4. 경제 환경 분석
5. 종합 분석 및 최적 에이전트 선택

최종적으로 선택된 에이전트 목록과 각각의 선택 이유를 제시해주세요.
"""
        
        try:
            # Router Agent 실행
            result = self.agent.invoke({"input": request})
            
            # 중간 단계에서 선택된 에이전트 정보 추출
            selected_agents = []
            selection_reasons = []
            
            # 마지막 도구 실행 결과에서 선택된 에이전트 정보 파싱
            steps = result.get("intermediate_steps", [])
            for action, observation in steps:
                if hasattr(action, 'tool') and action.tool == "select_agents_for_market":
                    try:
                        agent_selection = json.loads(observation)
                        selected_agents = agent_selection.get('selected_agents', [])
                        selection_reasons = agent_selection.get('reasons', [])
                        break
                    except json.JSONDecodeError:
                        continue
            
            # 기본값 설정 (분석 실패 시) - 더 강화된 기본 조합
            if not selected_agents:
                selected_agents = ['WarrenBuffett', 'JosephPiotroski', 'JoelGreenblatt']
                selection_reasons = ['분석 실패로 인한 기본 조합: 가치투자 + 재무건전성 + 매직포뮬러 균형 전략']
            
            print(f"\n✅ 선택된 에이전트: {', '.join(selected_agents)}")
            print(f"📋 선택 이유: {'; '.join(selection_reasons)}")
            
            return {
                'market_analysis': result['output'],
                'selected_agents': selected_agents,
                'selection_reasons': selection_reasons,
                'agent_details': {
                    agent: self.agent_characteristics.get(agent, {})
                    for agent in selected_agents
                },
                'intermediate_steps': result.get('intermediate_steps', [])
            }
            
        except Exception as e:
            print(f"❌ Router Agent 실행 중 오류: {str(e)}")
            
            # 오류 시 기본 에이전트 반환 - 더 강화된 기본 조합
            return {
                'market_analysis': f"시장 분석 중 오류 발생: {str(e)}",
                'selected_agents': ['WarrenBuffett', 'JosephPiotroski', 'JoelGreenblatt'],
                'selection_reasons': ['오류로 인한 기본 조합: 가치투자 + 재무건전성 + 균형 전략'],
                'agent_details': {
                    'WarrenBuffett': self.agent_characteristics['WarrenBuffett'],
                    'JosephPiotroski': self.agent_characteristics['JosephPiotroski'],
                    'JoelGreenblatt': self.agent_characteristics['JoelGreenblatt']
                },
                'intermediate_steps': []
            }
    
    def execute_selected_agents(self, selected_agents: List[str], 
                              start_date: str, end_date: str) -> Dict[str, Any]:
        """
        선택된 에이전트들을 실행
        
        Args:
            selected_agents: 선택된 에이전트 이름 리스트
            start_date: 분석 시작일
            end_date: 분석 종료일
            
        Returns:
            각 에이전트의 실행 결과
        """
        
        print(f"\n🚀 선택된 에이전트들 실행: {', '.join(selected_agents)}")
        
        # 에이전트별 실행 결과 저장
        agent_results = {}
        
        for agent_name in selected_agents:
            try:
                print(f"\n⚡ {agent_name} 에이전트 실행 중...")
                
                # 각 에이전트의 모듈을 동적으로 import하고 실행
                if agent_name == 'WarrenBuffett':
                    from .WarrenBuffett_agent import WarrenBuffettInvestmentAnalyzer
                    agent = WarrenBuffettInvestmentAnalyzer()
                    result = agent.analyze(start_date, end_date)
                    
                elif agent_name == 'BenjaminGraham':
                    from .BenjaminGraham_agent import GrahamInvestmentAnalyzer  
                    agent = GrahamInvestmentAnalyzer()
                    result = agent.analyze(start_date, end_date)
                    
                elif agent_name == 'JosephPiotroski':
                    from .JosephPiotroski_agent import PiotroskiInvestmentAnalyzer
                    agent = PiotroskiInvestmentAnalyzer()
                    result = agent.analyze(start_date, end_date)
                    
                elif agent_name == 'JoelGreenblatt':
                    from .JoelGreenblatt_agent import GreenblattInvestmentAnalyzer
                    agent = GreenblattInvestmentAnalyzer()
                    result = agent.analyze(start_date, end_date)
                    
                elif agent_name == 'EdwardAltman':
                    from .EdwardAltman_agent import AltmanInvestmentAnalyzer
                    agent = AltmanInvestmentAnalyzer()
                    result = agent.analyze(start_date, end_date)
                    
                else:
                    result = f"Unknown agent: {agent_name}"
                
                agent_results[agent_name] = {
                    'result': result,
                    'status': 'success',
                    'timestamp': datetime.now().isoformat()
                }
                
                print(f"✅ {agent_name} 완료")
                
            except Exception as e:
                print(f"❌ {agent_name} 실행 실패: {str(e)}")
                agent_results[agent_name] = {
                    'result': f"Error: {str(e)}",
                    'status': 'error',
                    'timestamp': datetime.now().isoformat()
                }
        
        return agent_results
    
    def run_complete_analysis(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        전체 분석 프로세스 실행: 시장 분석 → 에이전트 선택 → 에이전트 실행
        
        Args:
            start_date: 분석 시작일  
            end_date: 분석 종료일
            
        Returns:
            전체 분석 결과
        """
        
        print("🎯 MoE-Invest Router Agent: 전체 분석을 시작합니다!")
        print(f"📅 분석 기간: {start_date} ~ {end_date}\n")
        
        start_time = datetime.now()
        
        # 1단계: 시장 분석 및 에이전트 선택
        routing_result = self.analyze_and_route(start_date, end_date)
        
        # 2단계: 선택된 에이전트들 실행  
        agent_results = self.execute_selected_agents(
            routing_result['selected_agents'],
            start_date, 
            end_date
        )
        
        # 3단계: 결과 종합
        execution_time = (datetime.now() - start_time).total_seconds()
        
        final_result = {
            'timestamp': datetime.now().isoformat(),
            'execution_time_seconds': execution_time,
            'input_parameters': {
                'start_date': start_date,
                'end_date': end_date
            },
            'routing_analysis': routing_result,
            'agent_results': agent_results,
            'summary': {
                'total_agents_selected': len(routing_result['selected_agents']),
                'successful_executions': len([r for r in agent_results.values() if r['status'] == 'success']),
                'failed_executions': len([r for r in agent_results.values() if r['status'] == 'error'])
            }
        }
        
        # 4단계: 결과 저장
        self._save_router_results(final_result, start_date, end_date)
        
        print(f"\n🎉 전체 분석 완료! (실행시간: {execution_time:.2f}초)")
        print(f"📈 선택된 에이전트: {len(routing_result['selected_agents'])}개")
        print(f"✅ 성공: {final_result['summary']['successful_executions']}개")
        print(f"❌ 실패: {final_result['summary']['failed_executions']}개")
        
        return final_result
    
    def _save_router_results(self, results: Dict[str, Any], start_date: str, end_date: str):
        """Router Agent 결과 저장"""
        try:
            # 결과 저장 디렉토리 생성
            os.makedirs("results/router_agent", exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # JSON 결과 저장
            json_filename = f"results/router_agent/router_analysis_{start_date}_{end_date}_{timestamp}.json"
            with open(json_filename, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
            # 요약 리포트 저장
            report_filename = f"results/router_agent/router_report_{start_date}_{end_date}_{timestamp}.md"
            with open(report_filename, "w", encoding="utf-8") as f:
                f.write("# MoE-Invest Router Agent 분석 리포트\n\n")
                f.write(f"**분석 시간**: {results['timestamp']}\n")
                f.write(f"**실행 시간**: {results['execution_time_seconds']:.2f}초\n")
                f.write(f"**분석 기간**: {start_date} ~ {end_date}\n\n")
                
                f.write("## 시장 분석 결과\n\n")
                f.write(results['routing_analysis']['market_analysis'])
                f.write("\n\n")
                
                f.write("## 선택된 에이전트\n\n")
                for agent in results['routing_analysis']['selected_agents']:
                    details = results['routing_analysis']['agent_details'].get(agent, {})
                    f.write(f"### {agent}\n")
                    f.write(f"- **투자 스타일**: {details.get('style', 'N/A')}\n")
                    f.write(f"- **설명**: {details.get('description', 'N/A')}\n")
                    f.write(f"- **적합한 시장**: {', '.join(details.get('best_conditions', []))}\n\n")
                
                f.write("## 선택 이유\n\n")
                for reason in results['routing_analysis']['selection_reasons']:
                    f.write(f"- {reason}\n")
                
                f.write("\n## 에이전트 실행 결과\n\n")
                for agent_name, result in results['agent_results'].items():
                    f.write(f"### {agent_name}\n")
                    f.write(f"- **상태**: {result['status']}\n")
                    f.write(f"- **실행 시간**: {result['timestamp']}\n\n")
            
            print(f"\n💾 Router Agent 결과 저장 완료:")
            print(f"   📄 JSON: {json_filename}")
            print(f"   📝 Report: {report_filename}")
            
        except Exception as e:
            print(f"❌ 결과 저장 실패: {str(e)}")


if __name__ == "__main__":
    # 테스트 실행
    router = RouterAgent()
    
    # 예시 실행
    test_start = "2024-06-01"
    test_end = "2024-12-01"
    
    result = router.run_complete_analysis(test_start, test_end)
    print("\n" + "="*50)
    print("🔍 Router Agent 테스트 완료!")
    print("="*50)