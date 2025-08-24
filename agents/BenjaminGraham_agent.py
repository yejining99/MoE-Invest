# langchain 기반 그레이엄 에이전트
from langchain.prompts import PromptTemplate

import pandas as pd
import json
import numpy as np
import os
from datetime import datetime


template="""
## Role
You are **Benjamin Graham**, father of value investing. Your creed:
- “The individual investor should act consistently as an investor and not as a speculator.” 
- Insist that the buyer “has a margin of safety.”
- Prefer simple, testable selection rules; judge results at the **portfolio** level.
- Exploit deep value when available (e.g., **net-nets**); avoid over-elaborate analysis.
- Expect markets to overshoot: stocks “advance too far and decline too far.”
- Our policy places “relatively little stress” on forecasting markets; focus on **intrinsic value** and **financial strength**.

Your tone is prudent, skeptical, and independent. Favor strong liquidity, low leverage, durable profitability, and a clear margin of safety.

## Data (tabular fundamentals)
- One row per ticker for a quarter; identifiers: `TICKERSYMBOL`, `QUARTER` (e.g., “2023Q4”).
- Metrics may be missing; treat divide-by-zero or missing denominators as NA.

## Tools (call BEFORE ranking; once each on the full DataFrame)
- `metric_current_ratio(df)` → [{{ticker, current_ratio}}]
- `metric_debt_to_equity(df)` → [{{ticker, debt_to_equity}}]
- `metric_interest_coverage(df)` → [{{ticker, interest_coverage}}]
- `metric_roe(df)` → [{{ticker, roe}}]
- `metric_asset_turnover(df)` → [{{ticker, asset_turnover}}]
- `metric_profit_margin(df)` → [{{ticker, profit_margin}}]
- `metric_working_capital_ratio(df)` → [{{ticker, working_capital_ratio}}]
- `metric_valuation(df)` → [{{ticker, price, mktcap, pe, pb, pe_x_pb, ncav, is_netnet}}]
Use only these tool outputs to build the per-ticker metrics table.

## Scoring & Portfolio (concise, deterministic)
- **Scale** each metric across the universe via winsorize(5th–95th) → min–max to [0,1]. If a metric has no spread, set all scaled values to 0.50.
- **Handle NAs** per ticker by dropping only missing metrics and **renormalizing that ticker's metric weights** proportionally.
- **Score** = 0.25·CurrentRatio + 0.20·ROE + 0.20·ProfitMargin + 0.15·AssetTurnover + 0.10·WorkingCapital + 0.10·InterestCoverage  
  **Penalties:** D/E>0.5 −0.05; InterestCoverage<5 −0.05; ROE<5% −0.05  
  **Bonuses:** WorkingCapital>20% +0.05; CurrentRatio≥2 +0.05  
  **Clip** to [0,1]. **Tie-breakers:** higher CurrentRatio, lower D/E, higher ProfitMargin, then ticker alphabetical.
- **Portfolio** Include **all Eligible** tickers; weights ∝ **Score**; renormalize; round to whole % (**last row absorbs remainder**).  

## Output (STRICT)
Return **only** this markdown table:

| Ticker | Score | Weight (%) | Reason |
|--------|-------|------------|--------|

- Score: 2 decimals in [0.00, 1.00].
- Weight: integers summing to 100.
- Reason: one short sentence (e.g., "strong liquidity & margins; dinged for high D/E").

Think step-by-step privately, calling the above tools before ranking.
"""

from langchain.chains import LLMChain
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import MessagesPlaceholder
from langchain.tools import Tool
from typing import List
import re

class GrahamInvestmentAnalyzer:
    """LLMChain 기반 Graham 투자 분석기"""
    
    def __init__(self, llm=None, save_results=True, results_dir="results/graham_agent"):
        self.llm = llm or ChatOpenAI(temperature=0, model="gpt-4o")
        self.name = "Graham Value Analyzer"
        self.save_results = save_results
        self.results_dir = results_dir
        
        # 결과 저장 디렉토리 생성
        if self.save_results:
            os.makedirs(self.results_dir, exist_ok=True)
        
        self.df = pd.read_csv("data/nasdaq100_bs_cf_is.csv")
        self.df_ohlcv = pd.read_csv("data/nasdaq100_ohlcv.csv")
        self.df_ohlcv['EVAL_D'] = pd.to_datetime(self.df_ohlcv['EVAL_D'])
        
        # Agent용 프롬프트 템플릿 정의
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", template),
            ("user", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
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
                func=self.metric_roe
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
            ),
            Tool(
                name="metric_valuation",
                description="Compute valuation metrics (P/E, P/B, PExPB, NCAV, net-net flag) using price snapshot merged to fundamentals.",
                func=self.metric_valuation
            ),
        ]
        
        self.agent = self._create_agent()

    def safe_div(self,a, b):
        return (a / b).replace([np.inf, -np.inf], np.nan)
    
    @staticmethod
    def _quarter_end_date(q: str) -> pd.Timestamp:
        # "YYYYQx" → 분기 말일
        y = int(q[:4]); qi = int(q[-1])
        month_end = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}[qi]
        return pd.Timestamp(year=y, month=month_end[0], day=month_end[1])
    
    def _build_price_snapshot(self, q_end: str) -> pd.DataFrame:
        """
        분기말(또는 그 이전 최근 거래일)의 가격/시총/주식수 스냅샷을 티커별 1행으로 생성.
        우선순위: DIV_ADJ_CLOSE → CLOSE
        MKTCAP이 없으면 price * NUM_SHARES 로 추정
        """
        if self.df_ohlcv is None:
            return pd.DataFrame(columns=["TICKERSYMBOL","PX","NUM_SHARES_SNP","MKTCAP_SNP","PX_DATE"])

        qt_end = self._quarter_end_date(q_end)
        px = self.df_ohlcv[self.df_ohlcv["EVAL_D"] <= qt_end].copy()
        if px.empty:
            return pd.DataFrame(columns=["TICKERSYMBOL","PX","NUM_SHARES_SNP","MKTCAP_SNP","PX_DATE"])

        px["PX"] = px["DIV_ADJ_CLOSE"].fillna(px["CLOSE_"])
        # 최신 거래일만 남기기
        px = px.sort_values(["TICKERSYMBOL","EVAL_D"]).drop_duplicates(["TICKERSYMBOL"], keep="last")
        # 시총 보강
        est_mktcap = px["PX"] * px["NUM_SHARES"]
        px["MKTCAP_SNP"] = px["MKTCAP"].fillna(est_mktcap)
        out = px.rename(columns={"NUM_SHARES":"NUM_SHARES_SNP","EVAL_D":"PX_DATE"})[
            ["TICKERSYMBOL","PX","NUM_SHARES_SNP","MKTCAP_SNP","PX_DATE"]
        ].reset_index(drop=True)
        return out
    
    def _attach_ttm_and_bvps(self, q_end: str) -> None:
        base = self._original_df.copy() if hasattr(self, "_original_df") else self.df.copy()

        def _q_key(q):
            """분기 문자열을 숫자 키로 변환 (NaN과 잘못된 형식 처리)"""
            if pd.isna(q) or not isinstance(q, str):
                return -1  # NaN이나 잘못된 형식은 -1로 처리 (필터링될 값)
            
            try:
                q = str(q).strip()
                if len(q) < 5 or 'Q' not in q:
                    return -1
                y, qi = int(q[:4]), int(q[-1])
                return y*4 + qi
            except (ValueError, IndexError):
                return -1
        
        end_key = _q_key(q_end)

        # NaN이 아닌 QUARTER 값만 필터링하고 유효한 분기만 선택
        valid_quarters = base['QUARTER'].notna() & (base['QUARTER'].astype(str).str.contains('Q', na=False))
        base_filtered = base[valid_quarters].copy()
        
        recent4 = base_filtered[base_filtered["QUARTER"].apply(lambda x: _q_key(x) <= end_key and _q_key(x) > 0)].copy()
        
        if recent4.empty:
            print(f"경고: {q_end} 이전의 유효한 분기 데이터를 찾을 수 없습니다.")
            # 빈 데이터프레임으로 기본값 설정
            self.df = self.df.assign(
                NET_INCOME_TTM=np.nan,
                EQUITY_END=np.nan,
                NUM_SHARES_SNP=np.nan,
                EPS_TTM=np.nan,
                BVPS=np.nan
            )
            return
        
        recent4["QKEY"] = recent4["QUARTER"].apply(_q_key)
        recent4 = recent4.sort_values(["TICKERSYMBOL","QKEY"]).groupby("TICKERSYMBOL").tail(4)

        ni_col = "Net Income - (IS)" if "Net Income - (IS)" in recent4.columns else "Net Income"
        ttm = recent4.groupby("TICKERSYMBOL", as_index=False)[ni_col].sum().rename(columns={ni_col:"NET_INCOME_TTM"})

        # q_end 분기의 데이터도 유효성 검사
        end_quarter_data = base_filtered[base_filtered["QUARTER"]==q_end]
        if end_quarter_data.empty:
            print(f"경고: {q_end} 분기의 데이터를 찾을 수 없습니다.")
            eq = pd.DataFrame(columns=["TICKERSYMBOL", "EQUITY_END"])
        else:
            eq = end_quarter_data[["TICKERSYMBOL","Total Shareholders Equity (As Reported)"]]\
                    .rename(columns={"Total Shareholders Equity (As Reported)":"EQUITY_END"})

        shares = (self._pxsnap[["TICKERSYMBOL","NUM_SHARES_SNP"]]
                if hasattr(self, "_pxsnap") and not self._pxsnap.empty
                else pd.DataFrame(columns=["TICKERSYMBOL","NUM_SHARES_SNP"]))

        # 병합으로 정렬/정합 보장
        self.df = (self.df.merge(ttm, on="TICKERSYMBOL", how="left")
                        .merge(eq, on="TICKERSYMBOL", how="left")
                        .merge(shares, on="TICKERSYMBOL", how="left"))

        def _sd(a,b):
            """안전한 나눗셈"""
            if a is None or b is None:
                return pd.Series([np.nan] * len(self.df))
            s = pd.to_numeric(a, errors='coerce') / pd.to_numeric(b, errors='coerce')
            return s.replace([np.inf,-np.inf], np.nan)

        self.df["EPS_TTM"] = _sd(self.df.get("NET_INCOME_TTM"), self.df.get("NUM_SHARES_SNP"))
        self.df["BVPS"]    = _sd(self.df.get("EQUITY_END"),     self.df.get("NUM_SHARES_SNP"))

    def metric_valuation(self, *_args, **_kwargs) -> str:
        if not hasattr(self, "_pxsnap"):
            return json.dumps([])

        merged = self.df.merge(self._pxsnap, on="TICKERSYMBOL", how="left")

        # 총부채: 위 헬퍼 재사용 (merged 기준으로도 동작하도록 약식 처리)
        tl = (merged.get("Total Liabilities - (Standard / Utility Template)")
            if "Total Liabilities - (Standard / Utility Template)" in merged.columns else None)
        if tl is None:
            ta = merged.get("Total Assets") if "Total Assets" in merged.columns else merged.get("Total Assets (As Reported)")
            te = merged.get("Total Shareholders Equity (As Reported)")
            tl = ta - te if (ta is not None and te is not None) else pd.Series([np.nan]*len(merged))

        ca = merged.get("Total Current Assets")
        ncav = ca - tl if ca is not None else pd.Series([np.nan]*len(merged))

        price  = merged["PX"]
        mktcap = merged["MKTCAP_SNP"]
        eps    = merged.get("EPS_TTM")
        bvps   = merged.get("BVPS")
        equity = merged.get("EQUITY_END")
        ni_ttm = merged.get("NET_INCOME_TTM")

        def _sd(a,b):
            s = a.astype(float)/b.astype(float)
            return s.replace([np.inf,-np.inf], np.nan)

        pe_px = _sd(price, eps) if eps is not None else pd.Series([np.nan]*len(merged))
        # EPS<=0이면 NaN
        if eps is not None:
            pe_px = pe_px.where(eps > 0, np.nan)

        pe_mc = _sd(mktcap, ni_ttm) if ni_ttm is not None else pd.Series([np.nan]*len(merged))
        if ni_ttm is not None:
            pe_mc = pe_mc.where(ni_ttm > 0, np.nan)

        pe = pe_px.combine_first(pe_mc)

        pb_px = _sd(price, bvps) if bvps is not None else pd.Series([np.nan]*len(merged))
        if bvps is not None:
            pb_px = pb_px.where(bvps > 0, np.nan)

        pb_mc = _sd(mktcap, equity) if equity is not None else pd.Series([np.nan]*len(merged))
        if equity is not None:
            pb_mc = pb_mc.where(equity > 0, np.nan)

        pb = pb_px.combine_first(pb_mc)

        combo = pe * pb
        is_netnet = (mktcap < ncav)

        out = pd.DataFrame({
            "ticker": merged["TICKERSYMBOL"],
            "price": price,
            "mktcap": mktcap,
            "pe": pe,
            "pb": pb,
            "pe_x_pb": combo,
            "ncav": ncav,
            "is_netnet": is_netnet.astype("boolean")
        })
        return out.to_json(orient="records")


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
        return self.df[["TICKERSYMBOL", "current_ratio"]].to_json(orient="records")

    # Debt-to-Equity  
    def metric_debt_to_equity(self, *args, **kwargs) -> pd.Series:
        """
        Debt-to-Equity = Total Liabilities / Total Shareholders Equity (As Reported)
        """
        total_liabilities = self.df["Total Liabilities - (Standard / Utility Template)"]
        total_equity = self.df["Total Shareholders Equity (As Reported)"]
        self.df["debt_to_equity"] = self.safe_div(total_liabilities, total_equity)
        return self.df[["TICKERSYMBOL", "debt_to_equity"]].to_json(orient="records")

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
        return self.df[["TICKERSYMBOL", "interest_coverage"]].to_json(orient="records")

    # ROE (Return on Equity) - 시장 데이터 없이 계산 가능
    def metric_roe(self, *args, **kwargs) -> pd.Series:
        """
        ROE = Net Income / Total Shareholders Equity
        P/E와 P/B는 시장 데이터(가격, 주식수)가 없어 ROE로 대체
        """
        net_income = self.df.get("Net Income - (IS)", self.df.get("Net Income", 0))
        total_equity = self.df["Total Shareholders Equity (As Reported)"]
        self.df["roe"] = self.safe_div(net_income, total_equity) * 100  # 퍼센트로 표시
        return self.df[["TICKERSYMBOL", "roe"]].to_json(orient="records")

    # Asset Turnover - 시장 데이터 없이 계산 가능
    def metric_pb(self, *args, **kwargs) -> pd.Series:
        """
        Asset Turnover = Total Revenues / Total Assets
        P/B 대신 자산 회전율로 대체 (효율성 지표)
        """
        total_revenues = self.df.get("Total Revenues", self.df.get("Revenues", 0))
        total_assets = self.df["Total Assets"]
        self.df["asset_turnover"] = self.safe_div(total_revenues, total_assets)
        return self.df[["TICKERSYMBOL", "asset_turnover"]].to_json(orient="records")

    # Profit Margin - 시장 데이터 없이 계산 가능
    def metric_pe_x_pb(self, *args, **kwargs) -> pd.Series:
        """
        Profit Margin = Net Income / Total Revenues
        P/E × P/B 대신 이익률로 대체 (수익성 지표)
        """
        net_income = self.df.get("Net Income - (IS)", self.df.get("Net Income", 0))
        total_revenues = self.df.get("Total Revenues", self.df.get("Revenues", 1))
        self.df["profit_margin"] = self.safe_div(net_income, total_revenues) * 100  # 퍼센트로 표시
        return self.df[["TICKERSYMBOL", "profit_margin"]].to_json(orient="records")

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
        return self.df[["TICKERSYMBOL", "working_capital_ratio"]].to_json(orient="records")
    
    def _create_agent(self):
        """Agent Executor 생성"""
        agent = create_openai_tools_agent(self.llm, 
                                          self.tools, 
                                          self.prompt
                                          )
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True, return_intermediate_steps=True)

    def save_analysis_results(self, start_date: str, end_date: str, result: dict, execution_time: float):
        """분석 결과를 파일로 저장"""
        if not self.save_results:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 전체 결과를 JSON으로 저장
        result_data = {
            "timestamp": timestamp,
            "start_date": start_date,
            "end_date": end_date,
            "execution_time_seconds": execution_time,
            "final_output": result["output"],
            "intermediate_steps": []
        }
        
        # 2. 중간 단계들 저장
        steps = result.get("intermediate_steps", [])
        for i, (action, observation) in enumerate(steps):
            step_data = {
                "step_number": i + 1,
                "tool_name": action.tool if hasattr(action, 'tool') else str(action),
                "tool_input": action.tool_input if hasattr(action, 'tool_input') else {},
                "observation": observation,
                "timestamp": datetime.now().isoformat()
            }
            result_data["intermediate_steps"].append(step_data)
        
        # JSON 파일로 저장
        json_filename = f"{self.results_dir}/graham_analysis_{start_date}_{end_date}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2, default=str)
        
        # 3. 최종 포트폴리오 결과만 별도 CSV로 저장 (파싱 가능한 경우)
        try:
            portfolio_csv = self.extract_portfolio_table(result["output"])
            if portfolio_csv is not None:
                csv_filename = f"{self.results_dir}/graham_portfolio_{start_date}_{end_date}.csv"
                portfolio_csv.to_csv(csv_filename, index=False, encoding='utf-8')
                print(f"포트폴리오 결과 저장됨: {csv_filename}")
        except Exception as e:
            print(f"포트폴리오 CSV 저장 실패: {e}")
        
        # 4. 실행 로그 저장
        log_filename = f"{self.results_dir}/graham_execution_log_{start_date}_{end_date}.txt"
        with open(log_filename, 'w', encoding='utf-8') as f:
            f.write(f"Graham Investment Analysis Execution Log\n")
            f.write(f"{'='*50}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Period: {start_date} to {end_date}\n")
            f.write(f"Execution Time: {execution_time:.2f} seconds\n\n")
            
            f.write("Intermediate Steps:\n")
            f.write("-" * 30 + "\n")
            for i, (action, observation) in enumerate(steps):
                f.write(f"Step {i+1}:\n")
                f.write(f"Action: {action}\n")
                f.write(f"Observation: {observation}\n")
                f.write("-" * 50 + "\n")
            
            f.write(f"\nFinal Output:\n")
            f.write("=" * 30 + "\n")
            f.write(result["output"])
        
        print(f"분석 결과 저장 완료:")
        print(f"  - JSON: {json_filename}")
        print(f"  - 로그: {log_filename}")
        
        return {
            "json_file": json_filename,
            "log_file": log_filename,
            "csv_file": csv_filename if 'csv_filename' in locals() else None
        }
    
    def extract_portfolio_table(self, output: str) -> pd.DataFrame:
        """최종 출력에서 포트폴리오 테이블을 추출하여 DataFrame으로 변환"""
        try:
            lines = output.split('\n')
            table_lines = []
            in_table = False
            
            for line in lines:
                if '| Ticker |' in line or '|--------|' in line:
                    in_table = True
                    continue
                elif in_table and line.strip().startswith('|') and '|' in line.strip()[1:]:
                    table_lines.append(line.strip())
                elif in_table and not line.strip().startswith('|'):
                    break
            
            if table_lines:
                data = []
                for line in table_lines:
                    parts = [part.strip() for part in line.split('|')[1:-1]]  # 첫번째와 마지막 빈 부분 제거
                    if len(parts) >= 4:
                        data.append(parts)
                
                if data:
                    df = pd.DataFrame(data, columns=['Ticker', 'Score', 'Weight (%)', 'Reason'])
                    return df
        except Exception as e:
            print(f"테이블 추출 오류: {e}")
        
        return None
    
    def load_previous_results(self, limit=10):
        """이전 분석 결과들을 로드"""
        if not os.path.exists(self.results_dir):
            return []
        
        json_files = [f for f in os.listdir(self.results_dir) if f.startswith('graham_analysis_') and f.endswith('.json')]
        json_files.sort(reverse=True)  # 최신순 정렬
        
        results = []
        for filename in json_files[:limit]:
            try:
                with open(os.path.join(self.results_dir, filename), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results.append({
                        "filename": filename,
                        "timestamp": data.get("timestamp"),
                        "period": f"{data.get('start_date')} to {data.get('end_date')}",
                        "execution_time": data.get("execution_time_seconds"),
                        "steps_count": len(data.get("intermediate_steps", []))
                    })
            except Exception as e:
                print(f"결과 파일 로드 실패 {filename}: {e}")
        
        return results
    
    def analyze(self, start_date: str, end_date: str) -> str:
        """개선된 analyze 메서드 - 실행 시간 측정 및 결과 저장 포함"""
        start_time = datetime.now()
        
        # 원본 저장
        self._original_df = self.df.copy()

        # 분기 문자열로 정규화
        def _to_quarter(s: str) -> str:
            s = str(s).strip()
            if re.fullmatch(r"\d{4}Q[1-4]", s):
                return s
            dt = pd.to_datetime(s)
            q = (dt.month-1)//3 + 1
            return f"{dt.year}Q{q}"

        q_end = _to_quarter(end_date)
        print(f"start_date: {start_date}, end_date: {end_date}, quarter: {q_end}")

        # end 분기만 필터링(기존 로직 유지)
        if "QUARTER" in self.df.columns:
            self.df = self.df[self.df["QUARTER"] == q_end].copy()

        # 가격 스냅샷 생성 & 보관
        self._pxsnap = self._build_price_snapshot(q_end)

        # TTM/BVPS 부착
        self._attach_ttm_and_bvps(q_end)

        try:
            request = f"""
    Analyze these stocks using Graham's principles:
    - Period start: {start_date}
    - Period end: {q_end}
    """
            result = self.agent.invoke({"input": request})
            
            # 실행 시간 계산
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # 중간 단계 출력 (기존 유지)
            steps = result.get("intermediate_steps", [])
            for action, observation in steps:
                print(f"Action: {action.tool_calls if hasattr(action, 'tool_calls') else action}")
                print(f"Observation: {observation}")
                print("-"*100)
            
            # 결과 저장
            if self.save_results:
                saved_files = self.save_analysis_results(start_date, end_date, result, execution_time)
                print(f"실행 시간: {execution_time:.2f}초")
            
            return result["output"]
            
        finally:
            # 복원
            self.df = self._original_df
            if hasattr(self, "_pxsnap"):
                delattr(self, "_pxsnap")





