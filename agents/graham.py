"""
Benjamin Graham Value Investing Agent
벤자민 그레이엄의 가치투자 전략을 구현한 에이전트
"""

from .base_agent import BaseAgent, AgentInput, AgentOutput
import pandas as pd
import numpy as np
import os
import re

from langchain.prompts import PromptTemplate
from langchain.tools import tool
from data.get_ohlcv import get_ohlcv_data


class GrahamAgent(BaseAgent):
    """
    벤자민 그레이엄의 가치투자 전략 에이전트 (RunnableSequence 기반)
    
    주요 스크리닝 조건:
    1. P/E < 15
    2. P/B < 1.5
    3. Current Ratio > 2
    4. Debt/Equity < 0.5
    5. ROE > 10%
    """
    
    def __init__(self, llm=None):
        super().__init__(
            name="Graham Value Agent",
            description="벤자민 그레이엄의 가치투자 전략",
            llm=llm
        )
        self.parameters = {
            'max_pe': 15,
            'max_pb': 1.5,
            'min_current_ratio': 2.0,
            'max_debt_equity': 0.5,
            'min_roe': 0.10
        }
        
        # 설명 생성용 프롬프트 템플릿
        self.explanation_prompt = PromptTemplate.from_template("""
You are Benjamin Graham's AI assistant.

Your task is to analyze the following top {top_n} stocks based on the value investing strategy and assign investment weights to construct a portfolio.

## Strategy Background
- P/E Ratio < 15 (lower is better - indicates undervaluation)
- P/B Ratio < 1.5 (lower is better - indicates trading below book value)
- Current Ratio > 2 (higher is better - indicates financial strength)
- Debt/Equity < 0.5 (lower is better - indicates conservative debt levels)
- ROE > 10% (higher is better - indicates profitable operations)
- Focus on fundamentally sound companies with strong balance sheets trading at discount to intrinsic value.

## Past Reasoning History:
{history}

## Candidates:
{stock_list}

### Your task:
1. Briefly explain the value characteristics and financial strength of each stock.
2. Rank them from best to worst according to value investing principles.
3. Assign portfolio weights (in %) that sum up to 100, considering both value metrics and diversification.
4. Present the final portfolio in the following markdown table format:

| Ticker | P/E | P/B | Current Ratio | ROE | Score | Weight (%) | Reason |
|--------|-----|-----|---------------|-----|-------|------------|--------|
| AAPL   | 12.5| 1.2 | 2.3          | 15% | 0.82  | 25         | ...    |
...

Explain your reasoning step-by-step before showing the table.
""")

    def enrich_data(self, tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
        """
        종목 리스트에 대해 OHLCV 데이터를 수집
        """
        self.start_date = start_date
        self.end_date = end_date
        enriched = []
        for t in tickers:
            stock = get_ohlcv_data(t, start_date, end_date)
            enriched.append(stock)
        return pd.DataFrame(enriched)
    
    def screen_stocks(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        그레이엄의 가치투자 기준으로 스크리닝
        """
        screened = data.copy()
        
        # 필요한 컬럼이 없으면 기본값으로 채우기
        required_columns = {
            'pe_ratio': 12,                # 12배 기본값 (Graham 기준에 부합)
            'pb_ratio': 1.2,               # 1.2배 기본값 (Graham 기준에 부합)
            'current_ratio': 2.5,          # 2.5배 기본값 (Graham 기준 충족)
            'debt_equity_ratio': 0.3,      # 0.3배 기본값 (Graham 기준 충족)
            'roe': 0.12,                   # 12% 기본값 (Graham 기준 충족)
        }
        
        for col, default_value in required_columns.items():
            if col not in screened.columns:
                screened[col] = default_value
            else:
                screened[col] = screened[col].fillna(default_value)
        
        # 기본 스크리닝 조건
        conditions = [
            screened['pe_ratio'] < self.parameters['max_pe'],
            screened['pb_ratio'] < self.parameters['max_pb'],
            screened['current_ratio'] > self.parameters['min_current_ratio'],
            screened['debt_equity_ratio'] < self.parameters['max_debt_equity'],
            screened['roe'] > self.parameters['min_roe'],
            screened['pe_ratio'] > 0,  # 음수 P/E 제외
            screened['pb_ratio'] > 0,  # 음수 P/B 제외
            screened['current_ratio'] > 0,  # 음수 Current Ratio 제외
            screened['roe'].notna(),  # ROE 데이터 있는 종목만
        ]
        
        # 모든 조건을 만족하는 종목 필터링
        mask = np.logical_and.reduce(conditions)
        screened = screened[mask]
        
        return screened
    
    def calculate_score(self, stock_data: pd.Series) -> float:
        """
        그레이엄 점수 계산
        낮은 P/E, P/B일수록 높은 점수
        """
        # P/E 점수 (낮을수록 높은 점수)
        pe_score = max(0, 1 / stock_data['pe_ratio']) if stock_data['pe_ratio'] > 0 else 0
        
        # P/B 점수 (낮을수록 높은 점수)
        pb_score = max(0, 1 / stock_data['pb_ratio']) if stock_data['pb_ratio'] > 0 else 0
        
        # ROE 점수 (높을수록 높은 점수)
        roe_score = min(1.0, stock_data['roe'] / 0.20)  # 20% ROE를 최대값으로 정규화
        
        # Current Ratio 점수 (높을수록 높은 점수, 단 너무 높으면 불리)
        cr_score = min(1.0, stock_data['current_ratio'] / 3.0)
        
        # 종합 점수 (가중 평균)
        total_score = (0.4 * pe_score + 0.3 * pb_score + 0.2 * roe_score + 0.1 * cr_score)
        
        return total_score

    def explain_topN_with_llm(self, screened_df: pd.DataFrame, top_n: int = 10, 
                             start_date: str = None, end_date: str = None) -> str:
        """
        상위 N개 종목에 대한 LLM 설명 생성
        """
        if len(screened_df) == 0:
            return "스크리닝된 종목이 없습니다."
            
        screened_df_copy = screened_df.copy()
        if 'score' not in screened_df_copy.columns:
            screened_df_copy['score'] = screened_df_copy.apply(self.calculate_score, axis=1)
            
        top = screened_df_copy.sort_values(by="score", ascending=False).head(top_n)

        stock_list_text = "| Ticker | P/E | P/B | Current Ratio | Debt/Equity | ROE | Score |\n"
        stock_list_text += "|--------|-----|-----|---------------|-------------|-----|-------|\n"
        
        for _, row in top.iterrows():
            pe_ratio = row.get('pe_ratio', 0)
            pb_ratio = row.get('pb_ratio', 0)
            current_ratio = row.get('current_ratio', 0)
            debt_equity = row.get('debt_equity_ratio', 0)
            roe = row.get('roe', 0)
            score = row.get('score', 0)
            
            stock_list_text += f"| {row['ticker']} | {pe_ratio:.1f} | {pb_ratio:.1f} | {current_ratio:.1f} | {debt_equity:.2f} | {roe:.1%} | {score:.2f} |\n"

        prompt = self.explanation_prompt.format(
            history=self.memory.chat_memory.messages or "No prior reasoning.",
            top_n=top_n,
            stock_list=stock_list_text
        )

        # 프롬프트와 테이블 저장
        os.makedirs('data/graham_script', exist_ok=True)
        with open(f'data/graham_script/{start_date}_{end_date}_prompt.md', 'w') as f:
            f.write(prompt)

        response = self.llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # 포트폴리오 테이블 추출 및 저장
        portfolio_table = self.extract_portfolio_table(response_text)
        with open(f'data/graham_script/{start_date}_{end_date}_table.md', 'w') as f:
            f.write(portfolio_table)

        # 메모리에 저장
        self.memory.save_context(
            {"input": f"Explain Top {top_n} Graham value picks"},
            {"output": portfolio_table or "No summary table found."}
        )

        return response_text

    def extract_portfolio_table(self, text: str) -> str:
        """
        LLM 응답 텍스트에서 포트폴리오 테이블만 추출
        """
        pattern = r"(\| *Ticker *\|.*?\n\|[-| ]+\|\n(?:\|.*\n?)+)"
        match = re.search(pattern, text, re.MULTILINE)
        return match.group(1).strip() if match else "" 


# 편의를 위한 별칭
GrahamLLMAgent = GrahamAgent 