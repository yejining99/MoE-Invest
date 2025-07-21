"""
Tobias Carlisle's Deep Value and Momentum Agent
토비아스 칼라일의 딥 밸류 & 모멘텀 전략을 구현한 에이전트
"""

from .base_agent import BaseAgent, AgentInput, AgentOutput
import pandas as pd
import numpy as np
import os
import re

from langchain.prompts import PromptTemplate
from data.get_ohlcv import get_ohlcv_data


class CarlisleAgent(BaseAgent):
    """
    토비아스 칼라일의 딥 밸류 & 모멘텀 전략 에이전트 (RunnableSequence 기반)
    
    전략 구성요소:
    - Deep Value: EV/EBIT 기반 저평가 종목 선별
    - Momentum: 최근 성과 기반 모멘텀 필터링
    - Volatility-adjusted momentum
    """
    
    def __init__(self, llm=None):
        super().__init__(
            name="Carlisle Deep Value Agent",
            description="토비아스 칼라일의 딥 밸류 & 모멘텀 전략",
            llm=llm
        )
        self.parameters = {
            'min_ev_ebit_percentile': 0.10,  # 하위 10% EV/EBIT (저평가)
            'min_momentum_percentile': 0.70,  # 상위 30% 모멘텀
            'lookback_period': 252,          # 모멘텀 측정 기간 (1년)
        }

        self.explanation_prompt = PromptTemplate.from_template("""
You are Tobias Carlisle's AI assistant.

Your task is to analyze the following top {top_n} stocks based on the Deep Value + Momentum strategy and assign investment weights to construct a portfolio.

## Strategy Background
- Deep Value: Select stocks with low EV/EBIT ratios (cheapest 10-20%)
- Momentum Filter: Among value stocks, select those with positive momentum (top 30%)
- Volatility-adjusted momentum to account for risk
- Focus on undervalued companies showing recent strength
- Avoid value traps through momentum confirmation

## Past Reasoning History:
{history}

## Candidates:
{stock_list}

### Your task:
1. Analyze the value characteristics (EV/EBIT) and momentum of each stock.
2. Rank them from best to worst according to deep value + momentum principles.
3. Assign portfolio weights (in %) that sum up to 100, balancing value and momentum.
4. Present the final portfolio in the following markdown table format:

| Ticker | EV/EBIT | 12M Return | Volatility | Adj Momentum | Score | Weight (%) | Reason |
|--------|---------|------------|------------|--------------|-------|------------|--------|
| AAPL   | 6.2     | 25%        | 18%        | 1.39         | 0.82  | 25         | ...    |
...

Explain your reasoning step-by-step before showing the table.
""")
    
    def enrich_data(self, tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
        """종목 리스트에 대해 OHLCV 데이터를 수집"""
        self.start_date = start_date
        self.end_date = end_date
        enriched = []
        for t in tickers:
            stock = get_ohlcv_data(t, start_date, end_date)
            enriched.append(stock)
        return pd.DataFrame(enriched)

    def screen_stocks(self, data: pd.DataFrame) -> pd.DataFrame:
        """딥 밸류 & 모멘텀 기준으로 스크리닝"""
        screened = data.copy()
        
        # 필요한 컬럼이 없으면 기본값으로 채우기
        required_columns = {
            'ev_ebit': 5.5,           # EV/EBIT 5.5배 (저평가)
            'return_12m': 0.20,       # 12개월 수익률 20%
            'volatility': 0.25,       # 변동성 25%
            'adj_momentum': 0.80,     # 조정 모멘텀
            'ebit': 500000000         # EBIT 5억달러 기본값
        }
        
        for col, default_value in required_columns.items():
            if col not in screened.columns:
                screened[col] = default_value
            else:
                screened[col] = screened[col].fillna(default_value)
        
        # 스크리닝 조건
        conditions = [
            screened['ev_ebit'] <= screened['ev_ebit'].quantile(0.20),  # 하위 20% EV/EBIT
            screened['return_12m'] >= screened['return_12m'].quantile(0.60),  # 상위 40% 모멘텀
            screened['ebit'] > 0,  # 양의 EBIT
            screened['volatility'] < 0.60,  # 변동성이 너무 크지 않은 종목
        ]
        
        mask = np.logical_and.reduce(conditions)
        screened = screened[mask]
        
        return screened
    
    def calculate_score(self, stock_data: pd.Series) -> float:
        """딥 밸류 & 모멘텀 점수 계산"""
        # Value 점수 (EV/EBIT가 낮을수록 좋음)
        value_score = max(0, 1 - (stock_data['ev_ebit'] - 3) / 20)  # 3배를 기준으로 정규화
        
        # Momentum 점수 (높을수록 좋음)
        momentum_score = min(1.0, (stock_data['return_12m'] + 0.20) / 0.80)  # -20%~60% 범위로 정규화
        
        # 조정 모멘텀 점수 (변동성 고려)
        adj_momentum_score = min(1.0, stock_data['adj_momentum'] / 2.0)
        
        # 변동성 점수 (낮을수록 좋음)
        volatility_score = max(0, 1 - stock_data['volatility'] / 0.50)
        
        # 종합 점수 (밸류와 모멘텀을 중시)
        total_score = (0.3 * value_score + 0.3 * momentum_score + 0.3 * adj_momentum_score + 0.1 * volatility_score)
        
        return total_score

    def explain_topN_with_llm(self, screened_df: pd.DataFrame, top_n: int = 10,
                             start_date: str = None, end_date: str = None) -> str:
        """상위 N개 종목에 대한 LLM 설명 생성"""
        if len(screened_df) == 0:
            return "스크리닝된 종목이 없습니다."
            
        screened_df_copy = screened_df.copy()
        if 'score' not in screened_df_copy.columns:
            screened_df_copy['score'] = screened_df_copy.apply(self.calculate_score, axis=1)
            
        top = screened_df_copy.sort_values(by="score", ascending=False).head(top_n)

        stock_list_text = "| Ticker | EV/EBIT | 12M Return | Volatility | Adj Momentum | Score |\n"
        stock_list_text += "|--------|---------|------------|------------|--------------|-------|\n"
        
        for _, row in top.iterrows():
            ev_ebit = row.get('ev_ebit', 0)
            return_12m = row.get('return_12m', 0)
            volatility = row.get('volatility', 0)
            adj_momentum = row.get('adj_momentum', 0)
            score = row.get('score', 0)
            
            stock_list_text += f"| {row['ticker']} | {ev_ebit:.1f} | {return_12m:.1%} | {volatility:.1%} | {adj_momentum:.2f} | {score:.2f} |\n"

        prompt = self.explanation_prompt.format(
            history=self.memory.chat_memory.messages or "No prior reasoning.",
            top_n=top_n,
            stock_list=stock_list_text
        )

        # 프롬프트와 테이블 저장
        os.makedirs('data/carlisle_script', exist_ok=True)
        with open(f'data/carlisle_script/{start_date}_{end_date}_prompt.md', 'w') as f:
            f.write(prompt)
        
        response = self.llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # 포트폴리오 테이블 추출 및 저장
        portfolio_table = self.extract_portfolio_table(response_text)
        with open(f'data/carlisle_script/{start_date}_{end_date}_table.md', 'w') as f:
            f.write(portfolio_table)

        # 메모리에 저장
        self.memory.save_context(
            {"input": f"Explain Top {top_n} Deep Value + Momentum picks"},
            {"output": portfolio_table or "No summary table found."}
        )

        return response_text

    def extract_portfolio_table(self, text: str) -> str:
        """LLM 응답 텍스트에서 포트폴리오 테이블만 추출"""
        pattern = r"(\| *Ticker *\|.*?\n\|[-| ]+\|\n(?:\|.*\n?)+)"
        match = re.search(pattern, text, re.MULTILINE)
        return match.group(1).strip() if match else ""


# 편의를 위한 별칭
CarlisleLLMAgent = CarlisleAgent





