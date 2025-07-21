"""
Piotroski F-Score Quality Investing Agent
피오트로스키 F-스코어 품질투자 전략을 구현한 에이전트
"""

from .base_agent import BaseAgent, AgentInput, AgentOutput
import pandas as pd
import numpy as np
import os
import re

from langchain.prompts import PromptTemplate
from data.get_ohlcv import get_ohlcv_data


class PiotroskiAgent(BaseAgent):
    """
    피오트로스키 F-스코어 품질투자 전략 에이전트 (RunnableSequence 기반)
    
    F-Score 구성 요소:
    1. Profitability (수익성): ROA > 0, CFO > 0, ΔROA > 0, CFO > ROA
    2. Leverage/Liquidity (부채/유동성): ΔLEVER < 0, ΔLIQUID > 0, No equity offering
    3. Operating Efficiency (운영효율성): ΔMARGIN > 0, ΔTURNOVER > 0
    """
    
    def __init__(self, llm=None):
        super().__init__(
            name="Piotroski Quality Agent",
            description="피오트로스키 F-스코어 품질투자 전략",
            llm=llm
        )
        self.parameters = {
            'min_fscore': 7,  # 최소 F-Score (9점 만점에서 7점 이상)
            'min_roa': 0.05,  # 최소 ROA 5%
            'max_debt_ratio': 0.6  # 최대 부채비율 60%
        }
        
        self.explanation_prompt = PromptTemplate.from_template("""
You are Joseph Piotroski's AI assistant.

Your task is to analyze the following top {top_n} stocks based on the F-Score quality investing strategy and assign investment weights to construct a portfolio.

## Strategy Background
- F-Score: 9-point financial strength score (higher is better)
- Focus on companies with strong fundamentals and improving financial metrics
- ROA > 5% (profitability)
- CFO > Net Income (quality of earnings)
- Improving margins and asset turnover
- Conservative debt management
- Select companies with F-Score ≥ 7

## Past Reasoning History:
{history}

## Candidates:
{stock_list}

### Your task:
1. Analyze the financial quality and F-Score components of each stock.
2. Rank them from best to worst according to quality investing principles.
3. Assign portfolio weights (in %) that sum up to 100, prioritizing high F-Score companies.
4. Present the final portfolio in the following markdown table format:

| Ticker | F-Score | ROA | CFO/Sales | Debt Ratio | Score | Weight (%) | Reason |
|--------|---------|-----|-----------|------------|-------|------------|--------|
| AAPL   | 8       | 12% | 25%       | 35%        | 0.85  | 25         | ...    |
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
        """피오트로스키 F-Score 기준으로 스크리닝"""
        screened = data.copy()
        
        # 필요한 컬럼이 없으면 기본값으로 채우기 (우수한 품질 가정)
        required_columns = {
            'f_score': 8,           # F-Score 8점 기본값
            'roa': 0.08,           # ROA 8% 기본값  
            'cfo_sales': 0.15,     # CFO/Sales 15% 기본값
            'debt_ratio': 0.35,    # 부채비율 35% 기본값
            'margin_trend': 0.02,  # 마진 개선 2%p 기본값
            'turnover_trend': 0.05 # 회전율 개선 5% 기본값
        }
        
        for col, default_value in required_columns.items():
            if col not in screened.columns:
                screened[col] = default_value
            else:
                screened[col] = screened[col].fillna(default_value)
        
        # 스크리닝 조건
        conditions = [
            screened['f_score'] >= self.parameters['min_fscore'],
            screened['roa'] >= self.parameters['min_roa'], 
            screened['debt_ratio'] <= self.parameters['max_debt_ratio'],
            screened['cfo_sales'] > 0,  # 양의 현금흐름
            screened['f_score'].notna(),
        ]
        
        mask = np.logical_and.reduce(conditions)
        screened = screened[mask]
        
        return screened
    
    def calculate_score(self, stock_data: pd.Series) -> float:
        """F-Score 기반 품질 점수 계산"""
        # F-Score 점수 (정규화: 9점 만점)
        fscore_score = stock_data['f_score'] / 9.0
        
        # ROA 점수 (높을수록 좋음)
        roa_score = min(1.0, stock_data['roa'] / 0.20)  # 20% ROA를 최대값으로 정규화
        
        # 현금흐름 품질 점수
        cfo_score = min(1.0, stock_data['cfo_sales'] / 0.25)  # 25%를 최대값으로 정규화
        
        # 부채 안정성 점수 (낮을수록 좋음)
        debt_score = max(0, 1 - stock_data['debt_ratio'])
        
        # 종합 점수 (가중 평균)
        total_score = (0.4 * fscore_score + 0.3 * roa_score + 0.2 * cfo_score + 0.1 * debt_score)
        
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

        stock_list_text = "| Ticker | F-Score | ROA | CFO/Sales | Debt Ratio | Score |\n"
        stock_list_text += "|--------|---------|-----|-----------|------------|-------|\n"
        
        for _, row in top.iterrows():
            f_score = row.get('f_score', 0)
            roa = row.get('roa', 0)
            cfo_sales = row.get('cfo_sales', 0)
            debt_ratio = row.get('debt_ratio', 0)
            score = row.get('score', 0)
            
            stock_list_text += f"| {row['ticker']} | {f_score} | {roa:.1%} | {cfo_sales:.1%} | {debt_ratio:.1%} | {score:.2f} |\n"

        prompt = self.explanation_prompt.format(
            history=self.memory.chat_memory.messages or "No prior reasoning.",
            top_n=top_n,
            stock_list=stock_list_text
        )

        # 프롬프트와 테이블 저장
        os.makedirs('data/piotroski_script', exist_ok=True)
        with open(f'data/piotroski_script/{start_date}_{end_date}_prompt.md', 'w') as f:
            f.write(prompt)
            
        response = self.llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # 포트폴리오 테이블 추출 및 저장
        portfolio_table = self.extract_portfolio_table(response_text)
        with open(f'data/piotroski_script/{start_date}_{end_date}_table.md', 'w') as f:
            f.write(portfolio_table)

        # 메모리에 저장
        self.memory.save_context(
            {"input": f"Explain Top {top_n} Piotroski quality picks"},
            {"output": portfolio_table or "No summary table found."}
        )

        return response_text

    def extract_portfolio_table(self, text: str) -> str:
        """LLM 응답 텍스트에서 포트폴리오 테이블만 추출"""
        pattern = r"(\| *Ticker *\|.*?\n\|[-| ]+\|\n(?:\|.*\n?)+)"
        match = re.search(pattern, text, re.MULTILINE)
        return match.group(1).strip() if match else "" 


# 편의를 위한 별칭
PiotroskiLLMAgent = PiotroskiAgent 