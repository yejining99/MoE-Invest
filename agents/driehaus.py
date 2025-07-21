"""
Richard Driehaus Growth Investing Agent
리차드 드리하우스의 성장투자 전략을 구현한 에이전트
"""

from .base_agent import BaseAgent, AgentInput, AgentOutput
import pandas as pd
import numpy as np
import os
import re

from langchain.prompts import PromptTemplate
from data.get_ohlcv import get_ohlcv_data


class DriehausAgent(BaseAgent):
    """
    리차드 드리하우스의 성장투자 전략 에이전트 (RunnableSequence 기반)
    
    성장투자 기준:
    - 매출 성장률 > 20%
    - 순이익 성장률 > 25%
    - PEG Ratio < 1.5
    - 최근 모멘텀 positive
    """
    
    def __init__(self, llm=None):
        super().__init__(
            name="Driehaus Growth Agent",
            description="리차드 드리하우스의 성장투자 전략",
            llm=llm
        )
        self.parameters = {
            'min_revenue_growth': 0.20,    # 최소 매출 성장률 20%
            'min_earnings_growth': 0.25,   # 최소 순이익 성장률 25%
            'max_peg_ratio': 1.5,         # 최대 PEG 비율
            'min_momentum': 0.10          # 최소 모멘텀 10%
        }
        
        self.explanation_prompt = PromptTemplate.from_template("""
You are Richard Driehaus's AI assistant.

Your task is to analyze the following top {top_n} stocks based on the growth investing strategy and assign investment weights to construct a portfolio.

## Strategy Background
- Focus on companies with strong growth momentum
- Revenue Growth > 20% (higher is better)
- Earnings Growth > 25% (higher is better)
- PEG Ratio < 1.5 (reasonable valuation for growth)
- Positive price momentum and relative strength
- "Buy high, sell higher" philosophy - momentum over mean reversion

## Past Reasoning History:
{history}

## Candidates:
{stock_list}

### Your task:
1. Analyze the growth characteristics and momentum of each stock.
2. Rank them from best to worst according to growth investing principles.
3. Assign portfolio weights (in %) that sum up to 100, prioritizing strongest growth.
4. Present the final portfolio in the following markdown table format:

| Ticker | Rev Growth | EPS Growth | PEG | 3M Return | Score | Weight (%) | Reason |
|--------|------------|------------|-----|-----------|-------|------------|--------|
| AAPL   | 35%        | 42%        | 1.2 | 18%       | 0.85  | 25         | ...    |
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
        """성장투자 기준으로 스크리닝"""
        screened = data.copy()
        
        # 필요한 컬럼이 없으면 기본값으로 채우기 (성장주 특성)
        required_columns = {
            'revenue_growth': 0.25,      # 매출 성장률 25%
            'earnings_growth': 0.30,     # 순이익 성장률 30%
            'peg_ratio': 1.2,           # PEG 1.2배
            'return_3m': 0.15,          # 3개월 수익률 15%
            'relative_strength': 1.20,   # 상대강도 1.2
            'pe_ratio': 25              # P/E 25배 (성장주)
        }
        
        for col, default_value in required_columns.items():
            if col not in screened.columns:
                screened[col] = default_value
            else:
                screened[col] = screened[col].fillna(default_value)
        
        # 스크리닝 조건
        conditions = [
            screened['revenue_growth'] >= self.parameters['min_revenue_growth'],
            screened['earnings_growth'] >= self.parameters['min_earnings_growth'],
            screened['peg_ratio'] <= self.parameters['max_peg_ratio'],
            screened['return_3m'] >= self.parameters['min_momentum'],
            screened['peg_ratio'] > 0,  # 양의 PEG
            screened['pe_ratio'] > 0    # 양의 P/E
        ]
        
        mask = np.logical_and.reduce(conditions)
        screened = screened[mask]
        
        return screened
    
    def calculate_score(self, stock_data: pd.Series) -> float:
        """성장투자 점수 계산"""
        # 매출 성장 점수 (높을수록 좋음)
        revenue_score = min(1.0, stock_data['revenue_growth'] / 0.50)  # 50% 성장을 최대값으로 정규화
        
        # 순이익 성장 점수 (높을수록 좋음)
        earnings_score = min(1.0, stock_data['earnings_growth'] / 0.60)  # 60% 성장을 최대값으로 정규화
        
        # PEG 점수 (낮을수록 좋음)
        peg_score = max(0, 1 - (stock_data['peg_ratio'] - 0.5) / 2.0)  # 0.5를 최적으로 정규화
        
        # 모멘텀 점수 (높을수록 좋음)
        momentum_score = min(1.0, (stock_data['return_3m'] + 0.10) / 0.50)  # -10%~40% 범위로 정규화
        
        # 종합 점수 (성장률과 모멘텀을 중시)
        total_score = (0.3 * revenue_score + 0.3 * earnings_score + 0.2 * peg_score + 0.2 * momentum_score)
        
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

        stock_list_text = "| Ticker | Rev Growth | EPS Growth | PEG | 3M Return | Score |\n"
        stock_list_text += "|--------|------------|------------|-----|-----------|-------|\n"
        
        for _, row in top.iterrows():
            rev_growth = row.get('revenue_growth', 0)
            eps_growth = row.get('earnings_growth', 0)
            peg_ratio = row.get('peg_ratio', 0)
            return_3m = row.get('return_3m', 0)
            score = row.get('score', 0)
            
            stock_list_text += f"| {row['ticker']} | {rev_growth:.1%} | {eps_growth:.1%} | {peg_ratio:.1f} | {return_3m:.1%} | {score:.2f} |\n"

        prompt = self.explanation_prompt.format(
            history=self.memory.chat_memory.messages or "No prior reasoning.",
            top_n=top_n,
            stock_list=stock_list_text
        )

        # 프롬프트와 테이블 저장
        os.makedirs('data/driehaus_script', exist_ok=True)
        with open(f'data/driehaus_script/{start_date}_{end_date}_prompt.md', 'w') as f:
            f.write(prompt)
            
        response = self.llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # 포트폴리오 테이블 추출 및 저장
        portfolio_table = self.extract_portfolio_table(response_text)
        with open(f'data/driehaus_script/{start_date}_{end_date}_table.md', 'w') as f:
            f.write(portfolio_table)

        # 메모리에 저장
        self.memory.save_context(
            {"input": f"Explain Top {top_n} Growth picks"},
            {"output": portfolio_table or "No summary table found."}
        )

        return response_text

    def extract_portfolio_table(self, text: str) -> str:
        """LLM 응답 텍스트에서 포트폴리오 테이블만 추출"""
        pattern = r"(\| *Ticker *\|.*?\n\|[-| ]+\|\n(?:\|.*\n?)+)"
        match = re.search(pattern, text, re.MULTILINE)
        return match.group(1).strip() if match else "" 


# 편의를 위한 별칭
DriehausLLMAgent = DriehausAgent 