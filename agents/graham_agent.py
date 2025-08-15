# langchain 기반 그레이엄 에이전트
from langchain.prompts import PromptTemplate

import pandas as pd
import json
import numpy as np


template="""
    ## Role
You are Benjamin Graham, the father of value investing. 
Evaluate a universe of stocks using fundamental analysis, intrinsic value logic, and the Margin of Safety principle. 
Avoid speculation and short-term noise.

## Available Tools (call them BEFORE concluding)
You can call these Python tools to compute metrics **on the full DataFrame**:

- `metric_current_ratio(df) -> pd.Series`
  - Computes **Current Ratio = Total Current Assets / Total Current Liabilities**.
  - Uses columns: `Total Current Assets` (fallback: `Cash And Equivalents` + `Other Current Assets, Total`), `Total Current Liabilities`.

- `metric_debt_to_equity(df) -> pd.Series`
  - Computes **Debt-to-Equity = Total Liabilities / Total Shareholders Equity (As Reported)**.
  - Uses columns: `Total Liabilities`, `Total Shareholders Equity (As Reported)`.

- `metric_interest_coverage(df) -> pd.Series`
  - Computes **Interest Coverage = EBIT / |Interest Expense|**.
  - Uses columns: `EBIT` (fallback: `Gross Profit (As Reported)` - `Operating Expenses (As Reported)`), `Interest Expense`.

- `metric_roe(df) -> pd.Series`
  - Computes **ROE = Net Income / Total Shareholders Equity** (Return on Equity in %).
  - Uses columns: `Net Income - (IS)` (or `Net Income`), `Total Shareholders Equity (As Reported)`.

- `metric_asset_turnover(df) -> pd.Series`
  - Computes **Asset Turnover = Total Revenues / Total Assets** (Efficiency ratio).
  - Uses columns: `Total Revenues` (or `Revenues`), `Total Assets`.

- `metric_profit_margin(df) -> pd.Series`
  - Computes **Profit Margin = Net Income / Total Revenues** (Profitability in %).
  - Uses columns: `Net Income - (IS)`, `Total Revenues`.

- `metric_working_capital_ratio(df) -> pd.Series`
  - Computes **Working Capital Ratio = (Current Assets - Current Liabilities) / Total Assets** (in %).
  - Uses columns: `Total Current Assets`, `Total Current Liabilities`, `Total Assets`.

Additionally, compute **ROE** directly as:
- **ROE = Net Income / Total Shareholders Equity (As Reported)** (express as percent). If any field is missing or zero, leave ROE blank.

## Defensive Rules (apply as flags, not strict hard-stops for ranking)
- Current Ratio ≥ 2 preferred
- Debt-to-Equity ≤ 0.5 preferred
- Interest Coverage ≥ 5 preferred
- Graham combo rule: (P/E × P/B ≤ 22.5) preferred
- Net-Net bonus if (Total Current Assets – Total Liabilities) > market_cap

## Scoring (fundamental analysis based on available data)
1) After computing metrics via tools: Current Ratio, D/E, Interest Coverage, ROE, Asset Turnover, Profit Margin, Working Capital Ratio.
2) Build a **Fundamental Score** per stock (higher is better):
   - Normalize each metric across the universe with robust scaling (winsorize to 5th–95th percentiles, then min–max 0–1).
   - Score = 0.25·CurrentRatio_norm + 0.20·ROE_norm + 0.20·ProfitMargin_norm + 0.15·AssetTurnover_norm + 0.10·WorkingCapitalRatio_norm + 0.10·InterestCoverage_norm
   - Apply **penalties**: 
       - If D/E > 0.5, subtract 0.05
       - If Interest Coverage < 5, subtract 0.05
       - If ROE < 5%, subtract 0.05
   - Apply **bonus**:
       - If Working Capital Ratio > 20%, add 0.05
       - If Current Ratio ≥ 2, add 0.05
   - Clip Score to [0, 1].
3) Rank descending by Score.

## Portfolio construction (weights sum to 100)
- Select the top K = 10 names by Score (or all with Score > 0 if fewer than 10).
- Base weights proportional to Score.
- Soft diversification: cap each weight at 25%; if any cap applies, renormalize to sum 100.
- Round weights to whole percentages that sum to 100 (last item absorbs rounding remainder).

## Your task
1) Briefly explain the value characteristics and financial strength of each stock.
2) Rank them from best to worst according to value investing principles (the Score above).
3) Assign portfolio weights (in %) that sum to 100, considering both value metrics and diversification.
4) Present the final portfolio **ONLY** in the following markdown table format (no extra text outside the table):

| Ticker | Score | Weight (%) | Reason |
|--------|-------|------------|--------|

- Score with 2 decimals.
- Reason = one concise sentence capturing why it scores well/poorly (e.g., cheap on P/B, strong liquidity; dinged for high D/E).

## Reasoning style (important)
- Think step-by-step **privately**. Use the tools to compute metrics **before** ranking.
- In the final output, DO NOT reveal your internal step-by-step thoughts or raw intermediate calculations.
- Provide only the required table with concise “Reason” phrases.

## Now do this:
- Call the metric tools as needed on the provided DataFrame.
- Compute ROE as specified.
- Build scores, rank, construct weights, then output the final table exactly in the required format.
    """

from langchain.chains import LLMChain
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool
from typing import List

class GrahamInvestmentAnalyzer:
    """순수 LLMChain 기반 Graham 투자 분석기"""
    
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(temperature=0, model="gpt-4o")
        self.name = "Graham Value Analyzer"
        self.df = pd.read_csv("C:/Users/unist/Desktop/MOE-Invest/MoE-Invest/agent_data_20250813/agent_data_20250813/nasdaq100_bs_cf_is.csv")
        
        # Agent용 프롬프트 템플릿 정의
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", template),
            ("user", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])
        
        # 툴 정의
        self.tools = [
            Tool(
                name="metric_current_ratio",
                description="Current Ratio = Current Assets / Current Liabilities",
                func=self.metric_current_ratio
            ),
            Tool(
                name="metric_debt_to_equity", 
                description="Debt-to-Equity = Total Liabilities / Total Shareholders Equity",
                func=self.metric_debt_to_equity
            ),
            Tool(
                name="metric_interest_coverage",
                description="Interest Coverage = EBIT / Interest Expense",
                func=self.metric_interest_coverage
            ),
            Tool(
                name="metric_roe",
                description="ROE = Net Income / Total Shareholders Equity (Return on Equity)", 
                func=self.metric_pe
            ),
            Tool(
                name="metric_asset_turnover",
                description="Asset Turnover = Total Revenues / Total Assets (Efficiency)",
                func=self.metric_pb
            ),
            Tool(
                name="metric_profit_margin",
                description="Profit Margin = Net Income / Total Revenues (Profitability)",
                func=self.metric_pe_x_pb
            ),
            Tool(
                name="metric_working_capital_ratio",
                description="Working Capital Ratio = (Current Assets - Current Liabilities) / Total Assets",
                func=self.metric_net_net_surplus
            )
        ]
        
        self.agent = self._create_agent()

    def safe_div(self,a, b):
        return (a / b).replace([np.inf, -np.inf], np.nan)

    # Current Ratio
    def metric_current_ratio(self, *args, **kwargs) -> pd.Series:
        """
        Current Ratio = Current Assets / Current Liabilities
        - Current Assets: 'Total Current Assets' (fallback: Cash And Equivalents + Other Current Assets, Total)
        - Current Liabilities: 'Total Current Liabilities'
        """
        current_assets = self.df["Total Current Assets"].fillna(
            self.df["Cash And Equivalents"].fillna(0) + self.df["Other Current Assets, Total"].fillna(0)
        )
        current_liabilities = self.df["Total Current Liabilities"]
        self.df["current_ratio"] = self.safe_div(current_assets, current_liabilities)
        return self.df[["TICKERSYMBOL", "current_ratio"]]

    # Debt-to-Equity  
    def metric_debt_to_equity(self, *args, **kwargs) -> pd.Series:
        """
        Debt-to-Equity = Total Liabilities / Total Shareholders Equity (As Reported)
        """
        total_liabilities = self.df["Total Liabilities & Shareholders Equity (As Reported)"]
        total_equity = self.df["Total Shareholders Equity (As Reported)"]
        self.df["debt_to_equity"] = self.safe_div(total_liabilities, total_equity)
        return self.df[["TICKERSYMBOL", "debt_to_equity"]]

    # Interest Coverage
    def metric_interest_coverage(self, *args, **kwargs) -> pd.Series:
        """
        Interest Coverage = EBIT / |Interest Expense|
        - EBIT: 'EBIT' (fallback: Gross Profit (As Reported) - Operating Expenses (As Reported))
        """
        ebit = self.df["EBIT"].fillna(
            self.df["Gross Profit (As Reported)"] - self.df["Operating Expenses (As Reported)"]
        )
        interest_expense = self.df["Interest Expense"].abs()
        self.df["interest_coverage"] = self.safe_div(ebit, interest_expense)
        return self.df[["TICKERSYMBOL", "interest_coverage"]]

    # ROE (Return on Equity) - 시장 데이터 없이 계산 가능
    def metric_pe(self, *args, **kwargs) -> pd.Series:
        """
        ROE = Net Income / Total Shareholders Equity
        P/E와 P/B는 시장 데이터(가격, 주식수)가 없어 ROE로 대체
        """
        net_income = self.df.get("Net Income - (IS)", self.df.get("Net Income", 0))
        total_equity = self.df["Total Shareholders Equity (As Reported)"]
        self.df["roe"] = self.safe_div(net_income, total_equity) * 100  # 퍼센트로 표시
        return self.df[["TICKERSYMBOL", "roe"]]

    # Asset Turnover - 시장 데이터 없이 계산 가능
    def metric_pb(self, *args, **kwargs) -> pd.Series:
        """
        Asset Turnover = Total Revenues / Total Assets
        P/B 대신 자산 회전율로 대체 (효율성 지표)
        """
        total_revenues = self.df.get("Total Revenues", self.df.get("Revenues", 0))
        total_assets = self.df["Total Assets"]
        self.df["asset_turnover"] = self.safe_div(total_revenues, total_assets)
        return self.df[["TICKERSYMBOL", "asset_turnover"]]

    # Profit Margin - 시장 데이터 없이 계산 가능
    def metric_pe_x_pb(self, *args, **kwargs) -> pd.Series:
        """
        Profit Margin = Net Income / Total Revenues
        P/E × P/B 대신 이익률로 대체 (수익성 지표)
        """
        net_income = self.df.get("Net Income - (IS)", self.df.get("Net Income", 0))
        total_revenues = self.df.get("Total Revenues", self.df.get("Revenues", 1))
        self.df["profit_margin"] = self.safe_div(net_income, total_revenues) * 100  # 퍼센트로 표시
        return self.df[["TICKERSYMBOL", "profit_margin"]]

    # Working Capital Ratio - 시장 데이터 없이 계산 가능
    def metric_net_net_surplus(self, *args, **kwargs) -> pd.Series:
        """
        Working Capital Ratio = (Current Assets - Current Liabilities) / Total Assets
        Net-Net Surplus 대신 운전자본 비율로 대체 (유동성 지표)
        """
        current_assets = self.df["Total Current Assets"].fillna(
            self.df["Cash And Equivalents"].fillna(0) + self.df["Other Current Assets, Total"].fillna(0)
        )
        current_liabilities = self.df["Total Current Liabilities"]
        total_assets = self.df["Total Assets"]
        working_capital = current_assets - current_liabilities
        self.df["working_capital_ratio"] = self.safe_div(working_capital, total_assets) * 100  # 퍼센트로 표시
        return self.df[["TICKERSYMBOL", "working_capital_ratio"]]
    
    def _create_agent(self):
        """Agent Executor 생성"""
        agent = create_openai_tools_agent(self.llm, 
                                          self.tools, 
                                          self.prompt
                                          )
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)

    
    def analyze(self, start_date: str, end_date: str, top_n: int = 10) -> str:
        """직접적인 분석 메서드 - BaseAgent 인터페이스 불필요"""
        request = f"""
        Analyze these stocks using Graham's principles:
        - Period: {start_date} to {end_date}
        - Target: Top {top_n} recommendations
        """

        # 원본 데이터 백업 후 필터링된 복사본 생성
        original_df = self.df.copy()
        self.df = self.df[self.df["QUARTER"] == end_date].copy()
        
        try:
            result = self.agent.invoke({"input": request})
            return result["output"]
        finally:
            # 분석 완료 후 원본 데이터 복원
            self.df = original_df




