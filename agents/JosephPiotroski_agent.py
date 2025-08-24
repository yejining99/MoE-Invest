# langchain 기반 피오트로스키(F-Score) 에이전트 - 개선된 버전
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import MessagesPlaceholder
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.chat_models import ChatOpenAI

import pandas as pd
import numpy as np
import json
import os
import re
from datetime import datetime
from typing import Optional


template = r"""
## Role
You are **Joseph Piotroski**, creator of the **F-Score** (2000).
Your method is a simple, rules-based checklist of **nine binary signals** to separate strong from weak value stocks.
You emphasize **accounting quality** and **recent fundamental improvement**, not forecasting.

## Data
- Fundamentals are quarterly; each row has `TICKERSYMBOL` and `QUARTER` (e.g., "2024Q1").
- Prices/NUM_SHARES come from a separate OHLCV table; we take **quarter-end snapshots** for the current quarter (t) and prior-year same quarter (t-4).
- Divide-by-zero and invalid denominators should become NA (not zero).

## Piotroski Signals (1 if true, else 0; NA if cannot be evaluated)
**Profitability**
1) ROA > 0            (ROA = Net Income / Total Assets)
2) CFO > 0            (CFO = Net Cash from Operating Activities)
3) ΔROA > 0           (ROA_t − ROA_{{t-1y}} > 0)
4) Accruals           (CFO > Net Income)

**Leverage/Liquidity/Source of Funds**
5) ΔLeverage < 0      (Leverage_t − Leverage_{{t-1y}} < 0; Leverage = Long-Term Debt / Total Assets; fallback: Total Liabilities / Total Assets)
6) ΔLiquidity > 0     (Current Ratio_t − Current Ratio_{{t-1y}} > 0; CR = Current Assets / Current Liabilities)
7) No Equity Issuance (Shares Outstanding_t ≤ Shares Outstanding_{{t-1y}})

**Operating Efficiency**
8) ΔGross Margin > 0  (GM_t − GM_{{t-1y}} > 0; GM = Gross Profit / Revenues)
9) ΔAsset Turnover > 0(ATO_t − ATO_{{t-1y}} > 0; ATO = Revenues / Total Assets)

## Tools (call BEFORE ranking; once each on the full DataFrame)
- `metric_profitability(df)` → [{{ticker, roa_t, cfo_t, delta_roa, accrual_signal}}]
- `metric_leverage_liquidity(df)` → [{{ticker, delta_leverage, delta_liquidity, no_equity_issuance}}]
- `metric_efficiency(df)` → [{{ticker, delta_margin, delta_turnover}}]
- `metric_fscore(df)` → [{{ticker, f_score}}]    # Sum of 9 signals using the above tool outputs (NA counts as 0)

Use only these tool outputs to construct your per-ticker table.

## Scoring & Portfolio (deterministic)
- **Eligibility**: at least 4 evaluable signals (adjusted from 6 due to data limitations). Treat NA signals as 0 toward F-score.
- **Primary ranking**: by **F-Score** (desc).
- **Score** (0–1): Score = F-Score / 9.00.
- **Tie-breakers**: higher ROA_t, then higher ΔGross Margin, then ticker alphabetical.
- **Selection**:
  - Prefer tickers with **F-Score ≥ 4** (adjusted threshold). If that yields <15 names, fill up to **K = min(30, ceil(0.3·N))** by continuing down the ranking.
- **Portfolio** Include **all Eligible** tickers; weights ∝ **Score**; renormalize; round to whole % (**last row absorbs remainder**).  

## Output (STRICT)
Return **only** this markdown table:

| Ticker | Score | Weight (%) | Reason |
|--------|-------|------------|--------|

- Score: 2 decimals in [0.00, 1.00].
- Weight: integers summing to 100.
- Reason: one short sentence (e.g., "F=4/9; positive ROA & margins").

Think step-by-step privately, calling the above tools before ranking.
"""


class PiotroskiInvestmentAnalyzer:
    """LLM + Tools 기반 Piotroski(F-Score) 투자 분석기 - 개선된 버전"""

    def __init__(
        self,
        llm=None,
        save_results: bool = True,
        results_dir: str = "results/piotroski_agent",
        fundamentals_csv: str = "data/nasdaq100_bs_cf_is.csv",
        ohlcv_csv: str = "data/nasdaq100_ohlcv.csv",
    ):
        self.llm = llm or ChatOpenAI(temperature=0, model="gpt-4o")
        self.name = "Piotroski F-Score Analyzer (Data-Constrained)"
        self.save_results = save_results
        self.results_dir = results_dir

        if self.save_results:
            os.makedirs(self.results_dir, exist_ok=True)

        # Load data
        self.df = pd.read_csv(fundamentals_csv)
        self.df_ohlcv = pd.read_csv(ohlcv_csv)
        self.df_ohlcv["EVAL_D"] = pd.to_datetime(self.df_ohlcv["EVAL_D"])

        # Identify available cash flow columns (currently none)
        self.cfo_columns = self._identify_cfo_columns()

        # Prompt
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", template),
                ("user", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )

        # Tools
        self.tools = [
            Tool(
                name="metric_profitability",
                description="Compute profitability signals: ROA_t, CFO_t (Net Income proxy), delta_roa, earnings quality signal. Returns JSON records.",
                func=self.metric_profitability,
            ),
            Tool(
                name="metric_leverage_liquidity",
                description="Compute delta_leverage, delta_liquidity, and no_equity_issuance (shares_t <= shares_{{t-1y}}). Returns JSON records.",
                func=self.metric_leverage_liquidity,
            ),
            Tool(
                name="metric_efficiency",
                description="Compute delta_margin and delta_turnover (GM and ATO improvements). Returns JSON records.",
                func=self.metric_efficiency,
            ),
            Tool(
                name="metric_fscore",
                description="Compute Piotroski F-Score (0..9) from prior tool outputs. Returns JSON records.",
                func=self.metric_fscore,
            ),
        ]

        self.agent = self._create_agent()

        # internal caches for tool outputs
        self._cache_profitability = None
        self._cache_levliq = None
        self._cache_eff = None

    def _identify_cfo_columns(self) -> list:
        """실제 데이터에서 현금흐름 관련 컬럼들을 식별"""
        # 원본 CF 데이터에서 확인된 실제 현금흐름 컬럼명들
        cfo_candidates = [
            "Cash from Operations (As Reported)", 
            "Cash from Operations", 
            "Net Cash Provided by Operating Activities", 
            "Net Cash from Operating Activities",
            "Operating Cash Flow", 
            "Cash From Operating Activities"
        ]
        
        # 실제 데이터에서 존재하는 컬럼 확인
        available_cfo_cols = []
        for col in cfo_candidates:
            if col in self.df.columns:
                available_cfo_cols.append(col)
                
        if available_cfo_cols:
            print(f"[DEBUG] 현금흐름 컬럼 발견: {available_cfo_cols}")
            return available_cfo_cols
        else:
            # 실제 데이터에서 Cash 관련 컬럼들 확인
            cash_cols = [c for c in self.df.columns if 'Cash' in c and ('Operations' in c or 'Operating' in c)]
            if cash_cols:
                print(f"[DEBUG] 대안 현금흐름 컬럼 사용: {cash_cols}")
                return cash_cols
            else:
                print("[DEBUG] 현금흐름 데이터 없음. 순이익을 CFO 대용으로 사용")
                return []

    def _build_historical_data(self, q_t: str, q_prev: str) -> tuple:
        """전년 동기 비교를 위한 히스토리컬 데이터 구축"""
        
        # 현재 분기와 전년 동기 데이터를 분리
        df_t = self.df[self.df["QUARTER"] == q_t].copy()
        df_prev = self.df[self.df["QUARTER"] == q_prev].copy()
        
        print(f"[DEBUG] 현재 분기 {q_t} 데이터: {len(df_t)} rows")
        print(f"[DEBUG] 전년 분기 {q_prev} 데이터: {len(df_prev)} rows")
        
        # 각 ticker별로 현재와 과거 데이터를 정렬
        tickers = sorted(set(df_t["TICKERSYMBOL"].unique()) | set(df_prev["TICKERSYMBOL"].unique()))
        print(f"[DEBUG] 전체 ticker 수: {len(tickers)}")
        
        return df_t, df_prev

    # -------------------------
    # Helpers
    # -------------------------
    @staticmethod
    def _quarter_end_date(q: str) -> pd.Timestamp:
        y = int(q[:4])
        qi = int(q[-1])
        month_end = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}[qi]
        return pd.Timestamp(year=y, month=month_end[0], day=month_end[1])

    @staticmethod
    def _to_quarter(s: str) -> str:
        s = str(s).strip()
        if re.fullmatch(r"\d{4}Q[1-4]", s):
            return s
        dt = pd.to_datetime(s)
        q = (dt.month - 1) // 3 + 1
        return f"{dt.year}Q{q}"

    @staticmethod
    def _q_minus_4(q: str) -> str:
        y, qi = int(q[:4]), int(q[-1])
        key = y * 4 + qi
        prev_key = key - 4
        prev_y = prev_key // 4
        prev_q = prev_key % 4
        if prev_q == 0:
            prev_q = 4
            prev_y -= 1
        return f"{prev_y}Q{prev_q}"

    @staticmethod
    def _safe_div(a, b):
        s = pd.to_numeric(a, errors="coerce") / pd.to_numeric(b, errors="coerce")
        return s.replace([np.inf, -np.inf], np.nan)

    def _col(self, df: pd.DataFrame, names: list[str], default=np.nan) -> pd.Series:
        """컬럼 접근 헬퍼 - 여러 후보 이름 중 존재하는 첫 번째를 사용"""
        for n in names:
            if n in df.columns:
                return pd.to_numeric(df[n], errors="coerce")
        return pd.Series(default, index=df.index, dtype="float64")

    def _build_price_snapshot_for(self, q_end: str) -> pd.DataFrame:
        """분기말 가격 스냅샷 구축"""
        if self.df_ohlcv is None:
            return pd.DataFrame(columns=["TICKERSYMBOL", "PX", "NUM_SHARES_SNP", "MKTCAP_SNP", "PX_DATE"])

        qt_end = self._quarter_end_date(q_end)
        px = self.df_ohlcv[self.df_ohlcv["EVAL_D"] <= qt_end].copy()
        if px.empty:
            return pd.DataFrame(columns=["TICKERSYMBOL", "PX", "NUM_SHARES_SNP", "MKTCAP_SNP", "PX_DATE"])

        px["PX"] = px["DIV_ADJ_CLOSE"].fillna(px["CLOSE_"])
        px = px.sort_values(["TICKERSYMBOL", "EVAL_D"]).drop_duplicates(["TICKERSYMBOL"], keep="last")
        est_mktcap = px["PX"] * px["NUM_SHARES"]
        px["MKTCAP_SNP"] = px["MKTCAP"].fillna(est_mktcap)

        out = px.rename(columns={"NUM_SHARES": "NUM_SHARES_SNP", "EVAL_D": "PX_DATE"})[
            ["TICKERSYMBOL", "PX", "NUM_SHARES_SNP", "MKTCAP_SNP", "PX_DATE"]
        ].reset_index(drop=True)
        return out

    # -------------------------
    # Tool implementations (데이터 제약 대응 버전)
    # -------------------------
    def metric_profitability(self, *args, **kwargs) -> str:
        """
        Returns JSON: [{ticker, roa_t, cfo_t, delta_roa, accrual_signal}]
        현금흐름 데이터가 있으면 사용하고, 없으면 순이익을 현금흐름 대용으로 사용
        """
        if not hasattr(self, "_df_t") or not hasattr(self, "_df_prev"):
            return json.dumps([])

        df_t, df_prev = self._df_t, self._df_prev
        
        # 공통 ticker 집합 확인
        tickers = sorted(set(df_t["TICKERSYMBOL"].dropna()) | set(df_prev["TICKERSYMBOL"].dropna()))
        result = pd.DataFrame({"TICKERSYMBOL": tickers})
        
        # 현재 분기 데이터 병합
        current = result.merge(df_t, on="TICKERSYMBOL", how="left")
        
        # 전년 동기 데이터를 별도로 처리 (접미사 없이)
        prev_data = result.merge(df_prev, on="TICKERSYMBOL", how="left")

        print(f"[DEBUG] Current 데이터: {len(current)} rows")
        print(f"[DEBUG] Previous 데이터: {len(prev_data)} rows") 
        print(f"[DEBUG] Previous 컬럼 중 Net Income: {[col for col in prev_data.columns if 'Net Income' in col]}")
        print(f"[DEBUG] Previous 컬럼 중 Total Assets: {[col for col in prev_data.columns if 'Total Assets' in col]}")

        # ROA 계산 (현재)
        ni_t = self._col(current, ["Net Income - (IS)", "Net Income", "Net Income (Loss)", "Net Income Attributable to Parent"])
        ta_t = self._col(current, ["Total Assets", "Total Assets (As Reported)"])
        roa_t = self._safe_div(ni_t, ta_t)

        # ROA 계산 (전년) - 동일한 컬럼명 사용
        ni_prev = self._col(prev_data, ["Net Income - (IS)", "Net Income", "Net Income (Loss)", "Net Income Attributable to Parent"])
        ta_prev = self._col(prev_data, ["Total Assets", "Total Assets (As Reported)"])
        roa_prev = self._safe_div(ni_prev, ta_prev)
        
        delta_roa = roa_t - roa_prev

        print(f"[DEBUG] ROA 현재: {roa_t.notna().sum()}/{len(roa_t)} 기업")
        print(f"[DEBUG] ROA 전년: {roa_prev.notna().sum()}/{len(roa_prev)} 기업")
        print(f"[DEBUG] Delta ROA: {delta_roa.notna().sum()}/{len(delta_roa)} 기업")

        # CFO 계산 - 현금흐름 데이터가 있으면 사용, 없으면 순이익 사용
        cfo_columns = self._identify_cfo_columns()
        if cfo_columns:
            # 실제 현금흐름 데이터 사용
            cfo_t = self._col(current, cfo_columns)
            # Accruals 신호: CFO > NI이면 좋음 (1), 아니면 나쁨 (0)
            accrual_signal = (cfo_t > ni_t).astype(float)
            accrual_signal = accrual_signal.where(~(cfo_t.isna() | ni_t.isna()), np.nan)
        else:
            # 순이익을 CFO 대용으로 사용
            cfo_t = ni_t
            # Accruals 신호: 전통적인 CFO > NI 대신 전년 대비 순이익 개선 사용
            accrual_signal = (ni_t > ni_prev).astype(float)
            accrual_signal = accrual_signal.where(~(ni_t.isna() | ni_prev.isna()), np.nan)

        out = pd.DataFrame({
            "ticker": result["TICKERSYMBOL"],
            "roa_t": roa_t,
            "cfo_t": cfo_t,
            "delta_roa": delta_roa,
            "accrual_signal": accrual_signal,
        })

        self._cache_profitability = out.copy()
        return out.to_json(orient="records")

    def metric_leverage_liquidity(self, *args, **kwargs) -> str:
        """
        Returns JSON: [{ticker, delta_leverage, delta_liquidity, no_equity_issuance}]
        레버리지와 유동성 개선 측정
        """
        if not hasattr(self, "_df_t") or not hasattr(self, "_df_prev") or not hasattr(self, "_pxsnap_t") or not hasattr(self, "_pxsnap_p"):
            return json.dumps([])

        df_t, df_prev = self._df_t, self._df_prev
        
        tickers = sorted(set(df_t["TICKERSYMBOL"].dropna()) | set(df_prev["TICKERSYMBOL"].dropna()))
        result = pd.DataFrame({"TICKERSYMBOL": tickers})
        
        current = result.merge(df_t, on="TICKERSYMBOL", how="left")
        prev_data = result.merge(df_prev, on="TICKERSYMBOL", how="left")

        # Leverage 계산 (현재)
        ltd_t = self._col(current, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation", "Long-term Debt"])
        tl_t = self._col(current, ["Total Liabilities - (Standard / Utility Template)", "Total Liabilities"])
        ta_t = self._col(current, ["Total Assets", "Total Assets (As Reported)"])
        
        lev_t = self._safe_div(ltd_t, ta_t)
        lev_t = lev_t.combine_first(self._safe_div(tl_t, ta_t))  # fallback

        # Leverage 계산 (전년)
        ltd_prev = self._col(prev_data, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation", "Long-term Debt"])
        tl_prev = self._col(prev_data, ["Total Liabilities - (Standard / Utility Template)", "Total Liabilities"])
        ta_prev = self._col(prev_data, ["Total Assets", "Total Assets (As Reported)"])
        
        lev_prev = self._safe_div(ltd_prev, ta_prev)
        lev_prev = lev_prev.combine_first(self._safe_div(tl_prev, ta_prev))
        
        delta_leverage = lev_t - lev_prev

        # Current Ratio 계산
        ca_t = self._col(current, ["Total Current Assets"])
        cl_t = self._col(current, ["Total Current Liabilities"])
        cr_t = self._safe_div(ca_t, cl_t)

        ca_prev = self._col(prev_data, ["Total Current Assets"])
        cl_prev = self._col(prev_data, ["Total Current Liabilities"])
        cr_prev = self._safe_div(ca_prev, cl_prev)
        
        delta_liquidity = cr_t - cr_prev

        # 주식 발행 확인
        px_t = self._pxsnap_t[["TICKERSYMBOL", "NUM_SHARES_SNP"]].rename(columns={"NUM_SHARES_SNP": "SH_T"})
        px_p = self._pxsnap_p[["TICKERSYMBOL", "NUM_SHARES_SNP"]].rename(columns={"NUM_SHARES_SNP": "SH_P"})
        sh = result.merge(px_t, on="TICKERSYMBOL", how="left").merge(px_p, on="TICKERSYMBOL", how="left")
        no_equity_issuance = (sh["SH_T"] <= sh["SH_P"]).astype(float)
        no_equity_issuance = no_equity_issuance.where(~(sh["SH_T"].isna() | sh["SH_P"].isna()), np.nan)

        print(f"[DEBUG] Delta Leverage: {delta_leverage.notna().sum()}/{len(delta_leverage)} 기업")
        print(f"[DEBUG] Delta Liquidity: {delta_liquidity.notna().sum()}/{len(delta_liquidity)} 기업")

        out = pd.DataFrame({
            "ticker": result["TICKERSYMBOL"],
            "delta_leverage": delta_leverage,
            "delta_liquidity": delta_liquidity,
            "no_equity_issuance": no_equity_issuance,
        })

        self._cache_levliq = out.copy()
        return out.to_json(orient="records")

    def metric_efficiency(self, *args, **kwargs) -> str:
        """
        Returns JSON: [{ticker, delta_margin, delta_turnover}]
        운영 효율성 개선 측정
        """
        if not hasattr(self, "_df_t") or not hasattr(self, "_df_prev"):
            return json.dumps([])

        df_t, df_prev = self._df_t, self._df_prev
        
        tickers = sorted(set(df_t["TICKERSYMBOL"].dropna()) | set(df_prev["TICKERSYMBOL"].dropna()))
        result = pd.DataFrame({"TICKERSYMBOL": tickers})
        
        current = result.merge(df_t, on="TICKERSYMBOL", how="left")
        prev_data = result.merge(df_prev, on="TICKERSYMBOL", how="left")

        # Gross Margin 계산 (현재)
        gp_t = self._col(current, ["Gross Profit (As Reported)", "Gross Profit"])
        rev_t = self._col(current, ["Total Revenues", "Revenues", "Revenue"])
        margin_t = self._safe_div(gp_t, rev_t)

        # Gross Margin 계산 (전년)
        gp_prev = self._col(prev_data, ["Gross Profit (As Reported)", "Gross Profit"])
        rev_prev = self._col(prev_data, ["Total Revenues", "Revenues", "Revenue"])
        margin_prev = self._safe_div(gp_prev, rev_prev)
        
        delta_margin = margin_t - margin_prev

        # Asset Turnover 계산
        ta_t = self._col(current, ["Total Assets", "Total Assets (As Reported)"])
        ato_t = self._safe_div(rev_t, ta_t)

        ta_prev = self._col(prev_data, ["Total Assets", "Total Assets (As Reported)"])
        ato_prev = self._safe_div(rev_prev, ta_prev)
        
        delta_turnover = ato_t - ato_prev

        print(f"[DEBUG] Delta Margin: {delta_margin.notna().sum()}/{len(delta_margin)} 기업")
        print(f"[DEBUG] Delta Turnover: {delta_turnover.notna().sum()}/{len(delta_turnover)} 기업")

        out = pd.DataFrame({
            "ticker": result["TICKERSYMBOL"],
            "delta_margin": delta_margin,
            "delta_turnover": delta_turnover,
        })

        self._cache_eff = out.copy()
        return out.to_json(orient="records")

    def metric_fscore(self, *args, **kwargs) -> str:
        """
        Compute F-Score = sum of 9 binary signals (NA → 0).
        Requires prior calls to metric_profitability / metric_leverage_liquidity / metric_efficiency.
        Returns JSON: [{ticker, f_score}]
        """
        # Ensure caches exist
        if self._cache_profitability is None or self._cache_levliq is None or self._cache_eff is None:
            return json.dumps([])

        df = (
            self._cache_profitability[["ticker", "roa_t", "cfo_t", "delta_roa", "accrual_signal"]]
            .merge(self._cache_levliq, on="ticker", how="outer")
            .merge(self._cache_eff, on="ticker", how="outer")
        )

        # Signals:
        roa_pos = (df["roa_t"] > 0).astype(float).where(~df["roa_t"].isna(), np.nan)
        cfo_pos = (df["cfo_t"] > 0).astype(float).where(~df["cfo_t"].isna(), np.nan)
        droa_pos = (df["delta_roa"] > 0).astype(float).where(~df["delta_roa"].isna(), np.nan)
        accrual = df["accrual_signal"]

        dlev_neg = (df["delta_leverage"] < 0).astype(float).where(~df["delta_leverage"].isna(), np.nan)
        dliq_pos = (df["delta_liquidity"] > 0).astype(float).where(~df["delta_liquidity"].isna(), np.nan)
        no_issue = df["no_equity_issuance"]

        dmar_pos = (df["delta_margin"] > 0).astype(float).where(~df["delta_margin"].isna(), np.nan)
        dato_pos = (df["delta_turnover"] > 0).astype(float).where(~df["delta_turnover"].isna(), np.nan)

        signals = pd.DataFrame({
            "ticker": df["ticker"],
            "s1_roa_pos": roa_pos,
            "s2_cfo_pos": cfo_pos,
            "s3_droa_pos": droa_pos,
            "s4_accrual": accrual,
            "s5_dlev_neg": dlev_neg,
            "s6_dliq_pos": dliq_pos,
            "s7_no_issue": no_issue,
            "s8_dmar_pos": dmar_pos,
            "s9_dato_pos": dato_pos,
        })

        # Count F-Score (NA→0)
        bin_cols = [c for c in signals.columns if c.startswith("s")]
        f_score = signals[bin_cols].fillna(0).sum(axis=1)

        out = pd.DataFrame({"ticker": signals["ticker"], "f_score": f_score})
        return out.to_json(orient="records")

    # -------------------------
    # Agent & persistence
    # -------------------------
    def _create_agent(self):
        agent = create_openai_tools_agent(self.llm, self.tools, self.prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True, return_intermediate_steps=True)

    def extract_portfolio_table(self, output: str) -> Optional[pd.DataFrame]:
        try:
            lines = output.split("\n")
            table_lines, in_table, header_seen = [], False, False
            for line in lines:
                if line.strip().startswith("| Ticker |") and not header_seen:
                    header_seen = True
                    in_table = True
                    continue
                if in_table and line.strip().startswith("|--------|"):
                    continue
                if in_table and line.strip().startswith("|") and "|" in line.strip()[1:]:
                    table_lines.append(line.strip())
                elif in_table and not line.strip().startswith("|"):
                    break
            if table_lines:
                data = []
                for ln in table_lines:
                    parts = [p.strip() for p in ln.split("|")[1:-1]]
                    if len(parts) >= 4:
                        data.append(parts[:4])
                if data:
                    return pd.DataFrame(data, columns=["Ticker", "Score", "Weight (%)", "Reason"])
        except Exception as e:
            print(f"테이블 추출 오류: {e}")
        return None

    def save_analysis_results(self, start_date: str, end_date: str, result: dict, execution_time: float):
        if not self.save_results:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_data = {
            "timestamp": timestamp,
            "start_date": start_date,
            "end_date": end_date,
            "execution_time_seconds": execution_time,
            "final_output": result["output"],
            "intermediate_steps": [],
        }

        steps = result.get("intermediate_steps", [])
        for i, (action, observation) in enumerate(steps):
            step_data = {
                "step_number": i + 1,
                "tool_name": action.tool if hasattr(action, "tool") else str(action),
                "tool_input": action.tool_input if hasattr(action, "tool_input") else {},
                "observation": observation,
                "timestamp": datetime.now().isoformat(),
            }
            result_data["intermediate_steps"].append(step_data)

        json_filename = f"{self.results_dir}/piotroski_analysis_{start_date}_{end_date}.json"
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2, default=str)

        try:
            portfolio_csv = self.extract_portfolio_table(result["output"])
            if portfolio_csv is not None:
                csv_filename = f"{self.results_dir}/piotroski_portfolio_{start_date}_{end_date}.csv"
                portfolio_csv.to_csv(csv_filename, index=False, encoding="utf-8")
                print(f"포트폴리오 결과 저장됨: {csv_filename}")
        except Exception as e:
            print(f"포트폴리오 CSV 저장 실패: {e}")

        log_filename = f"{self.results_dir}/piotroski_execution_log_{start_date}_{end_date}.txt"
        with open(log_filename, "w", encoding="utf-8") as f:
            f.write("Piotroski (F-Score) Investment Analysis Execution Log\n")
            f.write("=" * 60 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Period: {start_date} to {end_date}\n")
            f.write(f"Execution Time: {execution_time:.2f} seconds\n\n")
            f.write("Intermediate Steps:\n")
            f.write("-" * 40 + "\n")
            for i, (action, observation) in enumerate(steps):
                f.write(f"Step {i+1}:\n")
                f.write(f"Action: {action}\n")
                f.write(f"Observation: {observation}\n")
                f.write("-" * 50 + "\n")
            f.write("\nFinal Output:\n")
            f.write("=" * 30 + "\n")
            f.write(result["output"])

        print("분석 결과 저장 완료:")
        print(f"  - JSON: {json_filename}")
        print(f"  - 로그: {log_filename}")

        return {
            "json_file": json_filename,
            "log_file": log_filename,
        }

    # -------------------------
    # Public: main entry
    # -------------------------
    def analyze(self, start_date: str, end_date: str) -> str:
        """
        실행: end_date 기준 분기(t)와 전년동기(t-4)를 사용해 F-Score를 산출하고 포트폴리오를 구성.
        데이터 제약 대응 버전
        """
        t0 = datetime.now()

        # Keep original
        self._original_df = self.df.copy()

        # Normalize quarters
        q_t = self._to_quarter(end_date)
        q_p = self._q_minus_4(q_t)
        self._q_t, self._q_prev = q_t, q_p

        print(f"[Piotroski] start_date: {start_date}, end_date: {end_date}, quarter_t: {q_t}, quarter_prev: {q_p}")

        # Build historical comparison data
        df_t, df_prev = self._build_historical_data(q_t, q_p)
        self._df_t, self._df_prev = df_t, df_prev
        
        print(f"[Piotroski] 현재 분기 데이터: {len(df_t)} rows, 전년 동기 데이터: {len(df_prev)} rows")

        # Build price snapshots for t and t-4
        self._pxsnap_t = self._build_price_snapshot_for(q_t)
        self._pxsnap_p = self._build_price_snapshot_for(q_p)

        try:
            request = f"""
Analyze these stocks using Piotroski's F-Score (Data-Constrained Version):
- Period start: {start_date}
- Period end: {q_t} (previous: {q_p})
- Available tickers: {len(set(df_t['TICKERSYMBOL'].dropna()))}
- Historical comparison enabled: {len(df_prev) > 0}
- Note: Using Net Income as CFO proxy due to data limitations
"""
            result = self.agent.invoke({"input": request})

            # Log intermediate steps
            steps = result.get("intermediate_steps", [])
            for action, observation in steps:
                print(f"Action: {getattr(action, 'tool', action)}")
                obs_str = str(observation)
                print(f"Observation: {obs_str[:500]}{'...' if len(obs_str) > 500 else ''}")
                print("-" * 100)

            # Save
            t1 = datetime.now()
            if self.save_results:
                self.save_analysis_results(start_date, end_date, result, (t1 - t0).total_seconds())

            return result["output"]

        finally:
            # Restore
            self.df = self._original_df
            for attr in ["_df_t", "_df_prev", "_pxsnap_t", "_pxsnap_p", "_cache_profitability", "_cache_levliq", "_cache_eff"]:
                if hasattr(self, attr):
                    delattr(self, attr)


# 예시 실행
if __name__ == "__main__":
    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    agent = PiotroskiInvestmentAnalyzer(llm=llm)

    # 예: 한 분기 실행
    start_date = "2024-01-01"
    end_date = "2024-12-31"  # 또는 "2024Q4"
    output = agent.analyze(start_date, end_date)
    print(output)
