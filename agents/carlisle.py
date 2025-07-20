"""
Carlisle Momentum Agent
칼라일의 모멘텀 전략을 구현한 에이전트
"""

from .base_agent import BaseAgent
import pandas as pd
import numpy as np
import os

class CarlisleAgent(BaseAgent):
    """
    칼라일의 모멘텀 전략 에이전트
    
    주요 특징:
    1. 상대 모멘텀 (Relative Momentum)
    2. 절대 모멘텀 (Absolute Momentum)
    3. 변동성 조정
    """
    
    def __init__(self):
        super().__init__(
            name="Carlisle Momentum Agent",
            description="칼라일의 모멘텀 전략"
        )
        self.parameters = {
            'momentum_period': 252,  # 모멘텀 계산 기간 (1년)
            'min_momentum': 0.05,  # 최소 모멘텀 5%
            'volatility_period': 60,  # 변동성 계산 기간
            'min_market_cap': 10000000000,  # 최소 시가총액 (100억원)
        }
    
    def calculate_momentum(self, prices: pd.Series, period: int) -> float:
        """
        모멘텀 계산
        
        Args:
            prices: 가격 시계열
            period: 계산 기간
            
        Returns:
            모멘텀 값
        """
        if len(prices) < period:
            return 0
        
        current_price = prices.iloc[-1]
        past_price = prices.iloc[-period]
        
        return (current_price - past_price) / past_price
    
    def calculate_volatility(self, returns: pd.Series, period: int) -> float:
        """
        변동성 계산
        
        Args:
            returns: 수익률 시계열
            period: 계산 기간
            
        Returns:
            변동성 값
        """
        if len(returns) < period:
            return 0
        
        return returns.rolling(window=period).std().iloc[-1]
    
    def screen_stocks(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        모멘텀 기준으로 스크리닝
        
        Args:
            data: OHLCV 및 재무 데이터
            
        Returns:
            스크리닝된 종목들의 DataFrame
        """
        screened = data.copy()
        
        # 모멘텀과 변동성 계산
        momentum_values = []
        volatility_values = []
        
        for _, row in screened.iterrows():
            # 가격 데이터가 있다고 가정
            if 'price_history' in row:
                prices = pd.Series(row['price_history'])
                returns = prices.pct_change().dropna()
                
                momentum = self.calculate_momentum(prices, self.parameters['momentum_period'])
                volatility = self.calculate_volatility(returns, self.parameters['volatility_period'])
            else:
                # 실제 데이터가 없는 경우 기본값 사용
                momentum = row.get('momentum', 0)
                volatility = row.get('volatility', 0.2)
            
            momentum_values.append(momentum)
            volatility_values.append(volatility)
        
        screened['momentum'] = momentum_values
        screened['volatility'] = volatility_values
        
        # 기본 스크리닝 조건
        conditions = [
            screened['market_cap'] >= self.parameters['min_market_cap'],
            screened['momentum'] >= self.parameters['min_momentum'],
            screened['momentum'].notna(),
            screened['volatility'].notna(),
        ]
        
        # 모든 조건을 만족하는 종목 필터링
        mask = np.logical_and.reduce(conditions)
        screened = screened[mask]
        
        return screened
    
    def calculate_score(self, stock_data: pd.Series) -> float:
        """
        모멘텀 기반 점수 계산
        모멘텀을 변동성으로 조정한 Sharpe 비율과 유사한 지표
        
        Args:
            stock_data: 개별 종목의 데이터
            
        Returns:
            계산된 점수 (높을수록 좋음)
        """
        momentum = stock_data['momentum']
        volatility = stock_data['volatility']
        
        # 변동성 조정 모멘텀 (Sharpe 비율과 유사)
        if volatility > 0:
            risk_adjusted_momentum = momentum / volatility
        else:
            risk_adjusted_momentum = 0
        
        # 0-1 범위로 정규화 (2.0을 최대값으로 가정)
        normalized_score = min(1.0, max(0, risk_adjusted_momentum / 2.0))
        
        return normalized_score 
    
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from langchain.tools import tool
from data.get_ohlcv import get_ohlcv_data

import re

class CarlisleLLMAgent(CarlisleAgent):
    def __init__(self, llm=None):
        super().__init__()
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        api_key = os.getenv('OPENAI_API_KEY')
        self.llm = llm or ChatOpenAI(temperature=0, model="gpt-4o", api_key=api_key)

        self.explanation_prompt = PromptTemplate.from_template("""
You are Tobias Carlisle's AI assistant.

Your task is to analyze the following top {top_n} stocks based on the momentum strategy and assign investment weights to construct a portfolio.

## Strategy Background
- Momentum = (Current Price - Price 1 year ago) / Price 1 year ago
- Volatility = StdDev of daily returns over the past 60 days
- Score = Momentum / Volatility
- Stocks with higher scores are preferred, but diversification matters.

## Past Reasoning History:
{history}

## Candidates:
{stock_list}

### Your task:
1. Briefly explain the strengths/weaknesses of each stock.
2. Rank them from best to worst according to the strategy.
3. Assign portfolio weights (in %) that sum up to 100, considering both score and diversification.
4. Present the final portfolio in the following markdown table format:

| Ticker | Score | Weight (%) | Reason |
|--------|-------|------------|--------|
| AAPL   | 0.87  | 25         | ...    |
...

Explain your reasoning step-by-step before showing the table.
""")

        self.tools = [tool("get_ohlcv_data")(get_ohlcv_data)]
    
    def enrich_data(self, tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
        """
        종목 리스트에 대해 OHLCV 데이터를 수집
        
        Args:
            tickers (list): 종목 리스트
            start_date (str): 시작일 (YYYY-MM-DD)
            end_date (str): 종료일 (YYYY-MM-DD)
        """
        enriched = []
        for t in tickers:
            stock = get_ohlcv_data(t, start_date, end_date)
            enriched.append(stock)
        return pd.DataFrame(enriched)

    def explain_topN_with_llm(self, screened_df: pd.DataFrame, top_n: int = 10) -> str:
        screened_df['score'] = screened_df.apply(self.calculate_score, axis=1)
        top = screened_df.sort_values(by="score", ascending=False).head(top_n)

        stock_list_text = "| Ticker | Momentum | Volatility | Score |\n|--------|----------|------------|-------|\n"
        for _, row in top.iterrows():
            stock_list_text += f"| {row['ticker']} | {row['momentum']:.2f} | {row['volatility']:.2f} | {self.calculate_score(row):.2f} |\n"

        prompt = self.explanation_prompt.format(
            history=self.memory.chat_memory.messages or "No prior reasoning.",
            top_n=top_n,
            stock_list=stock_list_text
        )

        response = self.llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        self.portfolio_table = self.extract_portfolio_table(response_text)

        self.memory.save_context(
            {"input": f"Explain Top {top_n} Carlisle picks"},
            {"output": self.portfolio_table or "No summary table found."}
        )

        return response_text  # 문자열 반환

    def extract_portfolio_table(self, text: str) -> str:
        """
        LLM 응답 텍스트에서 포트폴리오 테이블만 추출

        Returns:
            Markdown 테이블 문자열 (없을 경우 빈 문자열)
        """

        # 개선된 정규식: 'Ticker' 포함한 헤더 ~ 다음 빈 줄 or 끝까지
        pattern = r"(\| *Ticker *\|.*?\n\|[-| ]+\|\n(?:\|.*\n?)+)"
        match = re.search(pattern, text, re.MULTILINE)
        return match.group(1).strip() if match else ""





