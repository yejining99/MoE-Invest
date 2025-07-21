"""
Joel Greenblatt's Magic Formula Agent
조엘 그린블랫의 마법공식 전략을 구현한 에이전트
"""

from .base_agent import BaseAgent, AgentInput, AgentOutput
import pandas as pd
import numpy as np
import os
import re

from langchain.prompts import PromptTemplate
from data.get_ohlcv import get_ohlcv_data


class GreenblattAgent(BaseAgent):
    """
    조엘 그린블랫의 마법공식 전략 에이전트 (RunnableSequence 기반)
    
    마법공식 = ROIC (투하자본수익률) + EY (수익수익률)
    - ROIC = EBIT / 투하자본
    - EY = EBIT / 기업가치(EV)
    """
    
    def __init__(self, llm=None):
        super().__init__(
            name="Greenblatt Magic Formula Agent",
            description="조엘 그린블랫의 마법공식 전략",
            llm=llm
        )
        self.parameters = {
            'min_roic': 0.15,      # 최소 ROIC 15%
            'min_earnings_yield': 0.10,  # 최소 수익수익률 10%
            'min_market_cap': 100000000   # 최소 시가총액 1억달러
        }
        
        self.explanation_prompt = PromptTemplate.from_template("""
You are Joel Greenblatt's AI assistant.

Your task is to analyze the following top {top_n} stocks based on the Magic Formula strategy and assign investment weights to construct a portfolio.

## Strategy Background
- Magic Formula = ROIC Rank + Earnings Yield Rank (lower combined rank is better)
- ROIC (Return on Invested Capital): EBIT / Invested Capital (higher is better)
- Earnings Yield: EBIT / Enterprise Value (higher is better)
- Focus on companies with both high returns on capital and attractive valuations
- Minimum ROIC ≥ 15%, Earnings Yield ≥ 10%

## Past Reasoning History:
{history}

## Candidates:
{stock_list}

### Your task:
1. Analyze the ROIC and Earnings Yield characteristics of each stock.
2. Rank them from best to worst according to the Magic Formula principles.
3. Assign portfolio weights (in %) that sum up to 100, prioritizing low combined ranks.
4. Present the final portfolio in the following markdown table format:

| Ticker | ROIC | Earnings Yield | EV/EBIT | Combined Rank | Score | Weight (%) | Reason |
|--------|------|----------------|---------|---------------|-------|------------|--------|
| AAPL   | 28%  | 12%           | 8.3     | 15           | 0.82  | 25         | ...    |
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
        """마법공식 기준으로 스크리닝"""
        screened = data.copy()
        
        # 필요한 컬럼이 없으면 기본값으로 채우기
        required_columns = {
            'roic': 0.20,              # ROIC 20% 기본값
            'earnings_yield': 0.12,    # 수익수익률 12% 기본값
            'ev_ebit': 8.0,           # EV/EBIT 8배 기본값
            'market_cap': 5000000000,  # 시가총액 50억달러 기본값
            'ebit': 1000000000        # EBIT 10억달러 기본값
        }
        
        for col, default_value in required_columns.items():
            if col not in screened.columns:
                screened[col] = default_value
            else:
                screened[col] = screened[col].fillna(default_value)
        
        # 스크리닝 조건
        conditions = [
            screened['roic'] >= self.parameters['min_roic'],
            screened['earnings_yield'] >= self.parameters['min_earnings_yield'],
            screened['market_cap'] >= self.parameters['min_market_cap'],
            screened['ebit'] > 0,  # 양의 EBIT
            screened['ev_ebit'] > 0,  # 양의 EV/EBIT
        ]
        
        mask = np.logical_and.reduce(conditions)
        screened = screened[mask]
        
        return screened
    
    def calculate_score(self, stock_data: pd.Series) -> float:
        """마법공식 점수 계산 (ROIC + 수익수익률 기반)"""
        # ROIC 점수 (높을수록 좋음)
        roic_score = min(1.0, stock_data['roic'] / 0.50)  # 50% ROIC를 최대값으로 정규화
        
        # 수익수익률 점수 (높을수록 좋음)
        ey_score = min(1.0, stock_data['earnings_yield'] / 0.25)  # 25% 수익률을 최대값으로 정규화
        
        # EV/EBIT 점수 (낮을수록 좋음)
        ev_ebit_score = max(0, 1 - (stock_data['ev_ebit'] - 5) / 20)  # 5배를 기준으로 정규화
        
        # 종합 점수 (마법공식은 ROIC와 수익수익률을 동등 가중)
        total_score = (0.4 * roic_score + 0.4 * ey_score + 0.2 * ev_ebit_score)
        
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

        stock_list_text = "| Ticker | ROIC | Earnings Yield | EV/EBIT | Score |\n"
        stock_list_text += "|--------|------|----------------|---------|-------|\n"
        
        for _, row in top.iterrows():
            roic = row.get('roic', 0)
            earnings_yield = row.get('earnings_yield', 0)
            ev_ebit = row.get('ev_ebit', 0)
            score = row.get('score', 0)
            
            stock_list_text += f"| {row['ticker']} | {roic:.1%} | {earnings_yield:.1%} | {ev_ebit:.1f} | {score:.2f} |\n"

        prompt = self.explanation_prompt.format(
            history=self.memory.chat_memory.messages or "No prior reasoning.",
            top_n=top_n,
            stock_list=stock_list_text
        )
        
        # 프롬프트와 테이블 저장
        os.makedirs('data/greenblatt_script', exist_ok=True)
        with open(f'data/greenblatt_script/{start_date}_{end_date}_prompt.md', 'w') as f:
            f.write(prompt)

        response = self.llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # 포트폴리오 테이블 추출 및 저장
        portfolio_table = self.extract_portfolio_table(response_text)
        with open(f'data/greenblatt_script/{start_date}_{end_date}_table.md', 'w') as f:
            f.write(portfolio_table)

        # 메모리에 저장
        self.memory.save_context(
            {"input": f"Explain Top {top_n} Magic Formula picks"},
            {"output": portfolio_table or "No summary table found."}
        )

        return response_text

    def extract_portfolio_table(self, text: str) -> str:
        """LLM 응답 텍스트에서 포트폴리오 테이블만 추출"""
        pattern = r"(\| *Ticker *\|.*?\n\|[-| ]+\|\n(?:\|.*\n?)+)"
        match = re.search(pattern, text, re.MULTILINE)
        return match.group(1).strip() if match else "" 


# 편의를 위한 별칭
GreenblattLLMAgent = GreenblattAgent 