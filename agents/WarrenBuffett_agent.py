# langchain 기반 워렌버핏 에이전트
from langchain.prompts import PromptTemplate

import pandas as pd
import json
import numpy as np
import os
from datetime import datetime

template="""
## Role
You are **Warren Buffett**, investor and business owner. Your creed:
- "It's far better to buy a **wonderful company at a fair price** than a fair company at a wonderful price."
- "Our favorite **holding period is forever**."
- "**Price is what you pay; value is what you get**." Keep price and intrinsic value distinct.
- Stay within your **circle of competence**; the boundary matters more than its size.
- Seek **moats** that widen over time; prefer durable advantages to fleeting growth.
- Be **fearful when others are greedy** and **greedy when others are fearful**; temperament beats IQ.
- Shun accounting gimmicks: **EBITDA** chest-thumping is pernicious; focus on owner earnings and cash.
- Ignore short-term market predictions and macro seers; the cemetery for seers is full.
- Intrinsic value is the **discounted cash** that can be taken out of a business. Think like an owner.

Your tone is plainspoken, patient, and business-like. Favor quality, enduring moats, conservative leverage, honest accounting, and long-term compounding grounded in intrinsic value.

## Data
- Universe: one row per ticker per quarter; identifiers: `TICKERSYMBOL`, `QUARTER` (e.g., "2023Q4").
- Merge a **quarter-end price snapshot** (price, shares, market cap) to fundamentals to compute P/E and P/B.
- Prefer **EBIT** (not EBITDA) for coverage and interest tests.
- Metrics may be missing; divide-by-zero → NA. Treat negative denominators as NA unless explicitly stated (e.g., use `|Interest Expense|` for coverage).

## Tools
Call each tool **once** on the full DataFrame **before** ranking. Build the per-ticker metrics table **only** from these outputs:
- `metric_debt_to_equity(df)` → `[{{ticker, debt_to_equity}}]`
- `metric_interest_coverage(df)` → `[{{ticker, interest_coverage}}]` *(EBIT / |Interest Expense|)*
- `metric_roe(df)` → `[{{ticker, roe}}]`
- `metric_profit_margin(df)` → `[{{ticker, profit_margin}}]`
- `metric_asset_turnover(df)` → `[{{ticker, asset_turnover}}]`
- `metric_valuation(df)` → `[{{ticker, price, mktcap, pe, pb, pe_x_pb, ncav, is_netnet}}]`
- `metric_fcf_yield(df)` → `[{{ticker, fcf_ttm, fcf_yield}}]`
- `metric_roce(df)` → `[{{ticker, roce}}]`

## Scoring & Portfolio (concise, deterministic)
- **Scale** each metric via winsorize (5th–95th) → min–max to **[0,1]**. If a metric has no spread, set all scaled values to **0.50**.  
  *(Direction: higher-better → use scaled as is; lower-better (PE, PB, CapExIntensity) → use `1 − scaled_metric` where needed.)*
- **Handle NAs** per ticker by dropping only missing components and **renormalizing that ticker's metric weights** proportionally.
- **Valuation subscore** = `0.55·scaled_FCFYield + 0.25·(1 − scaled_PB) + 0.20·(1 − scaled_PE)`; if none available → **0.50**.
- **Quality add-ons (QualityPlus)** = `+ 0.18·ROCE + 0.10·CashConversion + 0.06·MarginStability + 0.04·BuybackYield − 0.06·CapExIntensity`  
  *(Renormalize if any component is missing.)*
- **Bonuses/Penalties add (raw, unscaled):**  
  - **Bonuses:** `ROE ≥ 15%` **and** `D/E ≤ 0.5` `(+0.05)`; `InterestCoverage ≥ 10` `(+0.03)`; `ProfitMargin ≥ 15%` `(+0.02)`; `OwnerEarningsYield ≥ 5%` `(+0.03)`; `BuybackYield ≥ 2%` `(+0.02)`  
  - **Penalties:** `D/E > 1.0` `(−0.08)` *(additional `−0.05` if `D/E > 2.0`)*; `InterestCoverage < 5` `(−0.05)`; `PE > 35` **or** `PB > 6` `(−0.05)`; `FCF_TTM ≤ 0` `(−0.08)`
- **Score (deterministic):**  
  - **Base** = `0.28·ROE + 0.22·InterestCoverage + 0.18·ProfitMargin + 0.12·AssetTurnover + 0.10·Valuation + 0.05·CurrentRatio + 0.05·WorkingCapitalRatio`  
  - **Final Score** = `Base + QualityPlus + (Bonuses − Penalties)` → **clip to [0,1]**.  
  - **Tie-breakers (raw):** higher **ROE**, higher **InterestCoverage**, lower **D/E**, higher **ProfitMargin**, higher **ROCE**, then ticker alphabetical.
- **Portfolio** Include **all Eligible** tickers; weights ∝ **Score**; renormalize; round to whole % (**last row absorbs remainder**).  

## Output (STRICT)
Return **only** this markdown table:

| Ticker | Score | Weight (%) | Reason |
|--------|-------|------------|--------|

- **Score:** two decimals in **[0.00, 1.00]**.  
- **Weight:** integers summing to **100**.  
- **Reason:** one short sentence (e.g., "high ROE, strong coverage, fair multiple").  

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

class WarrenBuffettInvestmentAnalyzer:
    """LLMChain 기반 Warren Buffett 투자 분석기 (Graham 코드와 동일 구조, 버핏 프롬프트/스코어링만 교체)"""

    def __init__(
        self,
        llm=None,
        save_results=True,
        results_dir="C:/Users/unist/Desktop/MOE-Invest/MoE-Invest/results/buffett_agent",
    ):
        self.llm = llm or ChatOpenAI(temperature=0, model="gpt-4o")
        self.name = "Buffett Quality-Value Analyzer"
        self.save_results = save_results
        self.results_dir = results_dir

        # 결과 저장 디렉토리
        if self.save_results:
            os.makedirs(self.results_dir, exist_ok=True)

        # 데이터 로드
        self.df = pd.read_csv("C:/Users/unist/Desktop/MOE-Invest/MoE-Invest/data/nasdaq100_bs_cf_is.csv")
        self.df_ohlcv = pd.read_csv("C:/Users/unist/Desktop/MOE-Invest/MoE-Invest/data/nasdaq100_ohlcv.csv")
        self.df_ohlcv["EVAL_D"] = pd.to_datetime(self.df_ohlcv["EVAL_D"])

        # Prompt & Agent
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", template),
            ("user", "{input}"),
             MessagesPlaceholder("agent_scratchpad"),
             ])

        self.tools = [
        Tool(
            name="metric_debt_to_equity",
            description="Debt-to-Equity = Total Liabilities / Total Shareholders Equity",
            func=self.metric_debt_to_equity,
        ),
        Tool(
            name="metric_interest_coverage",
            description="Interest Coverage = EBIT / |Interest Expense|",
            func=self.metric_interest_coverage,
        ),
        Tool(
            name="metric_roe",
            description="ROE = Net Income / Total Shareholders Equity (Return on Equity)",
            func=self.metric_roe,
        ),
        Tool(
            name="metric_profit_margin",
            description="Profit Margin = Net Income / Total Revenues (Profitability)",
            func=self.metric_pe_x_pb,  # 이 함수가 profit_margin 계산을 수행
        ),
        Tool(
            name="metric_asset_turnover",
            description="Asset Turnover = Total Revenues / Total Assets (Efficiency)",
            func=self.metric_pb,  # 이 함수가 asset_turnover 계산을 수행
        ),
        Tool(
            name="metric_valuation",
            description="Compute valuation metrics (P/E, P/B, PExPB, NCAV, net-net flag) using price snapshot merged to fundamentals.",
            func=self.metric_valuation,
        ),
        Tool(
            name="metric_fcf_yield",
            description="Free cash flow TTM and FCF yield vs market cap.",
            func=self.metric_fcf_yield,
        ),
        Tool(
            name="metric_roce",
            description="ROCE ≈ NOPAT_TTM / invested capital (assets - current liabilities - cash).",
            func=self.metric_roce,
        ),
    ]

        self.agent = self._create_agent()

    def _pick_first_col(self, df: pd.DataFrame, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def _q_key(self, q):
        try:
            q = str(q).strip()
            return int(q[:4]) * 4 + int(q[-1])
        except:
            return -1

    def _recent_quarters(self, cols, n=4):
        """self._q_end 기준 마지막 n개 분기를 ticker별로 모아 반환"""
        base = getattr(self, "_original_df", self.df).copy()
        end_key = self._q_key(self._q_end)
        valid = base["QUARTER"].astype(str).str.contains("Q", na=False)
        sub = base[valid & (base["QUARTER"].apply(self._q_key) <= end_key)].copy()
        sub["QKEY"] = sub["QUARTER"].apply(self._q_key)
        sub = sub.sort_values(["TICKERSYMBOL", "QKEY"]).groupby("TICKERSYMBOL").tail(n)
        keep = ["TICKERSYMBOL", "QKEY"] + [c for c in cols if c in sub.columns]
        return sub[keep].copy()

    def _safe_div_series(self, a: pd.Series, b: pd.Series):
        s = pd.to_numeric(a, errors="coerce") / pd.to_numeric(b, errors="coerce")
        return s.replace([np.inf, -np.inf], np.nan)                 


    def safe_div(self, a, b):
        return (a / b).replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _quarter_end_date(q: str) -> pd.Timestamp:
        y = int(q[:4])
        qi = int(q[-1])
        month_end = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}[qi]
        return pd.Timestamp(year=y, month=month_end[0], day=month_end[1])

    def _build_price_snapshot(self, q_end: str) -> pd.DataFrame:
        """분기말(또는 그 이전 최근 거래일)의 가격/시총/주식수 스냅샷"""
        if self.df_ohlcv is None:
            return pd.DataFrame(
                columns=["TICKERSYMBOL", "PX", "NUM_SHARES_SNP", "MKTCAP_SNP", "PX_DATE"]
            )

        qt_end = self._quarter_end_date(q_end)
        px = self.df_ohlcv[self.df_ohlcv["EVAL_D"] <= qt_end].copy()
        if px.empty:
            return pd.DataFrame(
                columns=["TICKERSYMBOL", "PX", "NUM_SHARES_SNP", "MKTCAP_SNP", "PX_DATE"]
            )

        px["PX"] = px["DIV_ADJ_CLOSE"].fillna(px["CLOSE_"])
        px = px.sort_values(["TICKERSYMBOL", "EVAL_D"]).drop_duplicates(
            ["TICKERSYMBOL"], keep="last"
        )
        est_mktcap = px["PX"] * px["NUM_SHARES"]
        px["MKTCAP_SNP"] = px["MKTCAP"].fillna(est_mktcap)
        out = px.rename(columns={"NUM_SHARES": "NUM_SHARES_SNP", "EVAL_D": "PX_DATE"})[
            ["TICKERSYMBOL", "PX", "NUM_SHARES_SNP", "MKTCAP_SNP", "PX_DATE"]
        ].reset_index(drop=True)
        return out

    def _attach_ttm_and_bvps(self, q_end: str) -> None:
        base = self._original_df.copy() if hasattr(self, "_original_df") else self.df.copy()

        def _q_key(q):
            if pd.isna(q) or not isinstance(q, str):
                return -1
            try:
                q = str(q).strip()
                if len(q) < 5 or "Q" not in q:
                    return -1
                y, qi = int(q[:4]), int(q[-1])
                return y * 4 + qi
            except (ValueError, IndexError):
                return -1

        end_key = _q_key(q_end)
        valid_quarters = base["QUARTER"].notna() & (
            base["QUARTER"].astype(str).str.contains("Q", na=False)
        )
        base_filtered = base[valid_quarters].copy()
        recent4 = base_filtered[
            base_filtered["QUARTER"].apply(lambda x: _q_key(x) <= end_key and _q_key(x) > 0)
        ].copy()

        if recent4.empty:
            print(f"경고: {q_end} 이전의 유효한 분기 데이터를 찾을 수 없습니다.")
            self.df = self.df.assign(
                NET_INCOME_TTM=np.nan,
                EQUITY_END=np.nan,
                NUM_SHARES_SNP=np.nan,
                EPS_TTM=np.nan,
                BVPS=np.nan,
            )
            return

        recent4["QKEY"] = recent4["QUARTER"].apply(_q_key)
        recent4 = (
            recent4.sort_values(["TICKERSYMBOL", "QKEY"]).groupby("TICKERSYMBOL").tail(4)
        )

        ni_col = (
            "Net Income - (IS)"
            if "Net Income - (IS)" in recent4.columns
            else "Net Income"
        )
        ttm = (
            recent4.groupby("TICKERSYMBOL", as_index=False)[ni_col]
            .sum()
            .rename(columns={ni_col: "NET_INCOME_TTM"})
        )

        end_quarter_data = base_filtered[base_filtered["QUARTER"] == q_end]
        if end_quarter_data.empty:
            print(f"경고: {q_end} 분기의 데이터를 찾을 수 없습니다.")
            eq = pd.DataFrame(columns=["TICKERSYMBOL", "EQUITY_END"])
        else:
            eq = end_quarter_data[
                ["TICKERSYMBOL", "Total Shareholders Equity (As Reported)"]
            ].rename(columns={"Total Shareholders Equity (As Reported)": "EQUITY_END"})

        shares = (
            self._pxsnap[["TICKERSYMBOL", "NUM_SHARES_SNP"]]
            if hasattr(self, "_pxsnap") and not self._pxsnap.empty
            else pd.DataFrame(columns=["TICKERSYMBOL", "NUM_SHARES_SNP"])
        )

        self.df = (
            self.df.merge(ttm, on="TICKERSYMBOL", how="left")
            .merge(eq, on="TICKERSYMBOL", how="left")
            .merge(shares, on="TICKERSYMBOL", how="left")
        )

        def _sd(a, b):
            if a is None or b is None:
                return pd.Series([np.nan] * len(self.df))
            s = pd.to_numeric(a, errors="coerce") / pd.to_numeric(b, errors="coerce")
            return s.replace([np.inf, -np.inf], np.nan)

        self.df["EPS_TTM"] = _sd(self.df.get("NET_INCOME_TTM"), self.df.get("NUM_SHARES_SNP"))
        self.df["BVPS"] = _sd(self.df.get("EQUITY_END"), self.df.get("NUM_SHARES_SNP"))


    def metric_valuation(self, *_args, **_kwargs) -> str:
        if not hasattr(self, "_pxsnap"):
            return json.dumps([])

        merged = self.df.merge(self._pxsnap, on="TICKERSYMBOL", how="left")

        tl = (
            merged.get("Total Liabilities - (Standard / Utility Template)")
            if "Total Liabilities - (Standard / Utility Template)" in merged.columns
            else None
        )
        if tl is None:
            ta = (
                merged.get("Total Assets")
                if "Total Assets" in merged.columns
                else merged.get("Total Assets (As Reported)")
            )
            te = merged.get("Total Shareholders Equity (As Reported)")
            tl = ta - te if (ta is not None and te is not None) else pd.Series([np.nan] * len(merged))

        ca = merged.get("Total Current Assets")
        ncav = ca - tl if ca is not None else pd.Series([np.nan] * len(merged))

        price = merged["PX"]
        mktcap = merged["MKTCAP_SNP"]
        eps = merged.get("EPS_TTM")
        bvps = merged.get("BVPS")
        equity = merged.get("EQUITY_END")
        ni_ttm = merged.get("NET_INCOME_TTM")

        def _sd(a, b):
            s = a.astype(float) / b.astype(float)
            return s.replace([np.inf, -np.inf], np.nan)

        pe_px = _sd(price, eps) if eps is not None else pd.Series([np.nan] * len(merged))
        if eps is not None:
            pe_px = pe_px.where(eps > 0, np.nan)

        pe_mc = _sd(mktcap, ni_ttm) if ni_ttm is not None else pd.Series([np.nan] * len(merged))
        if ni_ttm is not None:
            pe_mc = pe_mc.where(ni_ttm > 0, np.nan)

        pe = pe_px.combine_first(pe_mc)

        pb_px = _sd(price, bvps) if bvps is not None else pd.Series([np.nan] * len(merged))
        if bvps is not None:
            pb_px = pb_px.where(bvps > 0, np.nan)

        pb_mc = _sd(mktcap, equity) if equity is not None else pd.Series([np.nan] * len(merged))
        if equity is not None:
            pb_mc = pb_mc.where(equity > 0, np.nan)

        pb = pb_px.combine_first(pb_mc)

        combo = pe * pb
        is_netnet = mktcap < ncav

        out = pd.DataFrame(
            {
                "ticker": merged["TICKERSYMBOL"],
                "price": price,
                "mktcap": mktcap,
                "pe": pe,
                "pb": pb,
                "pe_x_pb": combo,
                "ncav": ncav,
                "is_netnet": is_netnet.astype("boolean"),
            }
        )
        return out.to_json(orient="records")

    def metric_current_ratio(self, *args, **kwargs) -> str:
        current_assets = self.df["Total Current Assets"].fillna(
            self.df["Cash And Equivalents"].fillna(0)
            + self.df["Other Current Assets, Total"].fillna(0)
        )
        current_liabilities = self.df["Total Current Liabilities"]
        self.df["current_ratio"] = self.safe_div(current_assets, current_liabilities)
        return self.df[["TICKERSYMBOL", "current_ratio"]].to_json(orient="records")

    def metric_debt_to_equity(self, *args, **kwargs) -> str:
        total_liabilities = self.df["Total Liabilities - (Standard / Utility Template)"]
        total_equity = self.df["Total Shareholders Equity (As Reported)"]
        self.df["debt_to_equity"] = self.safe_div(total_liabilities, total_equity)
        return self.df[["TICKERSYMBOL", "debt_to_equity"]].to_json(orient="records")

    def metric_interest_coverage(self, *args, **kwargs) -> str:
        ebit = self.df["EBIT"].fillna(
            self.df["Gross Profit (As Reported)"] - self.df["Operating Expenses (As Reported)"]
        )
        interest_expense = self.df["Interest Expense"].abs()
        self.df["interest_coverage"] = self.safe_div(ebit, interest_expense)
        return self.df[["TICKERSYMBOL", "interest_coverage"]].to_json(orient="records")

    def metric_roe(self, *args, **kwargs) -> str:
        net_income = self.df.get("Net Income - (IS)", self.df.get("Net Income", 0))
        total_equity = self.df["Total Shareholders Equity (As Reported)"]
        self.df["roe"] = self.safe_div(net_income, total_equity) * 100
        return self.df[["TICKERSYMBOL", "roe"]].to_json(orient="records")

    def metric_pb(self, *args, **kwargs) -> str:
        total_revenues = self.df.get("Total Revenues", self.df.get("Revenues", 0))
        total_assets = self.df["Total Assets"]
        self.df["asset_turnover"] = self.safe_div(total_revenues, total_assets)
        return self.df[["TICKERSYMBOL", "asset_turnover"]].to_json(orient="records")

    def metric_pe_x_pb(self, *args, **kwargs) -> str:
        net_income = self.df.get("Net Income - (IS)", self.df.get("Net Income", 0))
        total_revenues = self.df.get("Total Revenues", self.df.get("Revenues", 1))
        self.df["profit_margin"] = self.safe_div(net_income, total_revenues) * 100
        return self.df[["TICKERSYMBOL", "profit_margin"]].to_json(orient="records")

    def metric_net_net_surplus(self, *args, **kwargs) -> str:
        current_assets = self.df["Total Current Assets"].fillna(
            self.df["Cash And Equivalents"].fillna(0)
            + self.df["Other Current Assets, Total"].fillna(0)
        )
        current_liabilities = self.df["Total Current Liabilities"]
        total_assets = self.df["Total Assets"]
        working_capital = current_assets - current_liabilities
        self.df["working_capital_ratio"] = self.safe_div(working_capital, total_assets) * 100
        return self.df[["TICKERSYMBOL", "working_capital_ratio"]].to_json(orient="records")

    def metric_owner_earnings(self, maintenance_capex_ratio: float = 0.6) -> str:
        """Owner Earnings TTM = CFO_TTM - maintenance_capex_ratio * CapEx_TTM
        + OwnerEarningsYield = OwnerEarningsTTM / MarketCap"""
        # FCF와 동일한 컬럼명 사용으로 일관성 확보
        cfo_candidates   = ["Cash from Operations (As Reported)", "Cash from Operations", 
                            "Net Cash Provided by Operating Activities", "Net Cash from Operating Activities",
                            "Operating Cash Flow", "Cash From Operating Activities"]
        capex_candidates = ["Capital Expenditures (CF)", "Capital Expenditures",
                            "Purchase Of Property Plant Equipment"]

        cols = list(set(cfo_candidates + capex_candidates))
        rc = self._recent_quarters(cols, n=4)

        cfo_col   = self._pick_first_col(rc, cfo_candidates)
        capex_col = self._pick_first_col(rc, capex_candidates)
        if cfo_col is None or capex_col is None:
            # 컬럼이 없으면 전부 NA 반환
            print(f"[DEBUG] Owner Earnings: No CFO/CapEx columns found")
            out = pd.DataFrame({"ticker": self.df["TICKERSYMBOL"], "owner_earnings_ttm": np.nan, "owner_earnings_yield": np.nan})
            return out.to_json(orient="records")

        # CapEx는 통상 음수(유출). 절대값으로 집계
        rc["_capex_abs"] = rc[capex_col].abs()
        agg = rc.groupby("TICKERSYMBOL", as_index=False).agg(
            CFO_TTM=(cfo_col, "sum"),
            CAPEX_TTM=("_capex_abs", "sum"),
        )
        agg["OWNER_EARNINGS_TTM"] = agg["CFO_TTM"] - maintenance_capex_ratio * agg["CAPEX_TTM"]

        # 시총 붙이기
        px = getattr(self, "_pxsnap", pd.DataFrame(columns=["TICKERSYMBOL", "MKTCAP_SNP"]))
        merged = self.df[["TICKERSYMBOL"]].drop_duplicates().merge(agg, on="TICKERSYMBOL", how="left")\
                .merge(px[["TICKERSYMBOL", "MKTCAP_SNP"]], on="TICKERSYMBOL", how="left")
        merged["owner_earnings_yield"] = self._safe_div_series(merged["OWNER_EARNINGS_TTM"], merged["MKTCAP_SNP"])

        out = merged.rename(columns={"TICKERSYMBOL":"ticker", "OWNER_EARNINGS_TTM":"owner_earnings_ttm"})[
            ["ticker", "owner_earnings_ttm", "owner_earnings_yield"]
        ]
        return out.to_json(orient="records")


    def metric_fcf_yield(self, *args, **kwargs) -> str:
        """FCF TTM = CFO_TTM - CapEx_TTM; FCF Yield = FCF_TTM / MarketCap"""
        # 원본 CF 데이터에서 확인된 실제 컬럼명들을 포함
        cfo_candidates   = ["Cash from Operations (As Reported)", "Cash from Operations", 
                            "Net Cash Provided by Operating Activities", "Net Cash from Operating Activities",
                            "Operating Cash Flow", "Cash From Operating Activities"]
        capex_candidates = ["Capital Expenditures (CF)", "Capital Expenditures",
                            "Capital Expenditure", "Capital Expenditure - (Template Specific)",
                            "Purchase Of Property Plant Equipment"]

        cols = list(set(cfo_candidates + capex_candidates))
        rc = self._recent_quarters(cols, n=4)

        cfo_col   = self._pick_first_col(rc, cfo_candidates)
        capex_col = self._pick_first_col(rc, capex_candidates)
            
        if cfo_col is None or capex_col is None:
            out = pd.DataFrame({"ticker": self.df["TICKERSYMBOL"], "fcf_ttm": np.nan, "fcf_yield": np.nan})
            return out.to_json(orient="records")

        rc["_capex_abs"] = rc[capex_col].abs()
        agg = rc.groupby("TICKERSYMBOL", as_index=False).agg(
            CFO_TTM=(cfo_col, "sum"),
            CAPEX_TTM=("_capex_abs", "sum"),
        )
        agg["FCF_TTM"] = agg["CFO_TTM"] - agg["CAPEX_TTM"]

        px = getattr(self, "_pxsnap", pd.DataFrame(columns=["TICKERSYMBOL", "MKTCAP_SNP"]))
        merged = self.df[["TICKERSYMBOL"]].drop_duplicates().merge(agg, on="TICKERSYMBOL", how="left")\
                .merge(px[["TICKERSYMBOL", "MKTCAP_SNP"]], on="TICKERSYMBOL", how="left")
        merged["fcf_yield"] = self._safe_div_series(merged["FCF_TTM"], merged["MKTCAP_SNP"])
        
        out = merged.rename(columns={"TICKERSYMBOL":"ticker", "FCF_TTM":"fcf_ttm"})[
            ["ticker", "fcf_ttm", "fcf_yield"]
        ]
        return out.to_json(orient="records")


    def metric_cash_conversion(self, *args, **kwargs) -> str:
        """Cash Conversion = CFO_TTM / NetIncome_TTM (≥1 선호)"""
        cfo_candidates = ["Net Cash Provided by Operating Activities", "Net Cash from Operating Activities",
                        "Operating Cash Flow", "Cash From Operating Activities"]
        rc = self._recent_quarters(cfo_candidates + ["Net Income - (IS)", "Net Income"], n=4)

        cfo_col = self._pick_first_col(rc, cfo_candidates)
        ni_col  = "Net Income - (IS)" if "Net Income - (IS)" in rc.columns else ("Net Income" if "Net Income" in rc.columns else None)
        if cfo_col is None or ni_col is None:
            out = pd.DataFrame({"ticker": self.df["TICKERSYMBOL"], "cash_conversion": np.nan})
            return out.to_json(orient="records")

        agg = rc.groupby("TICKERSYMBOL", as_index=False).agg(
            CFO_TTM=(cfo_col, "sum"),
            NI_TTM=(ni_col, "sum"),
        )
        agg["cash_conversion"] = self._safe_div_series(agg["CFO_TTM"], agg["NI_TTM"])
        out = agg.rename(columns={"TICKERSYMBOL":"ticker"})[["ticker", "cash_conversion"]]
        return out.to_json(orient="records")


    def metric_roce(self, *args, **kwargs) -> str:
        """ROCE ≈ NOPAT_TTM / InvestedCapital_end; NOPAT_TTM = EBIT_TTM * (1 - ETR_TTM)"""
        rc = self._recent_quarters(["EBIT", "Income Tax Expense", "Pretax Income", "Income Before Tax"], n=4)

        # EBIT_TTM
        ebit_col = "EBIT" if "EBIT" in rc.columns else None
        if ebit_col is None:
            out = pd.DataFrame({"ticker": self.df["TICKERSYMBOL"], "roce": np.nan})
            return out.to_json(orient="records")
        ebit_ttm = rc.groupby("TICKERSYMBOL", as_index=False)[ebit_col].sum().rename(columns={ebit_col:"EBIT_TTM"})

        # 유효세율(ETR) 추정
        tax_col    = "Income Tax Expense" if "Income Tax Expense" in rc.columns else None
        pretax_col = "Pretax Income" if "Pretax Income" in rc.columns else ("Income Before Tax" if "Income Before Tax" in rc.columns else None)
        if tax_col and pretax_col:
            tax_agg = rc.groupby("TICKERSYMBOL", as_index=False).agg(TAX_TTM=(tax_col,"sum"), PRETAX_TTM=(pretax_col,"sum"))
            tax_agg["ETR"] = self._safe_div_series(tax_agg["TAX_TTM"], tax_agg["PRETAX_TTM"]).clip(lower=0, upper=0.35)
        else:
            tax_agg = pd.DataFrame({"TICKERSYMBOL": ebit_ttm["TICKERSYMBOL"], "ETR": 0.21})

        # Invested Capital (근사) = Total Assets - Total Current Liabilities - Cash
        ta  = pd.to_numeric(self.df.get("Total Assets"), errors="coerce")
        tcl = pd.to_numeric(self.df.get("Total Current Liabilities"), errors="coerce")
        cash = pd.to_numeric(self.df.get("Cash And Equivalents"), errors="coerce").fillna(0)
        ic = (ta - tcl - cash)
        ic = ic.where(ic > 0, np.nan)
        ic_df = pd.DataFrame({"TICKERSYMBOL": self.df["TICKERSYMBOL"], "IC_END": ic})

        merged = ebit_ttm.merge(tax_agg[["TICKERSYMBOL","ETR"]], on="TICKERSYMBOL", how="left").merge(ic_df, on="TICKERSYMBOL", how="left")
        merged["NOPAT_TTM"] = merged["EBIT_TTM"] * (1 - merged["ETR"].fillna(0.21))
        merged["roce"] = self._safe_div_series(merged["NOPAT_TTM"], merged["IC_END"])
        out = merged.rename(columns={"TICKERSYMBOL":"ticker"})[["ticker","roce"]]
        return out.to_json(orient="records")


    def metric_margin_stability(self, *args, **kwargs) -> str:
        """마진 안정성(높을수록 좋음) = 1 / (1 + std(분기별 ProfitMargin))"""
        rc = self._recent_quarters(["Net Income - (IS)", "Net Income", "Total Revenues", "Revenues"], n=4)
        ni_col = "Net Income - (IS)" if "Net Income - (IS)" in rc.columns else ("Net Income" if "Net Income" in rc.columns else None)
        rev_col = "Total Revenues" if "Total Revenues" in rc.columns else ("Revenues" if "Revenues" in rc.columns else None)
        if ni_col is None or rev_col is None:
            out = pd.DataFrame({"ticker": self.df["TICKERSYMBOL"], "margin_stability": np.nan})
            return out.to_json(orient="records")

        rc["pm"] = self._safe_div_series(rc[ni_col], rc[rev_col])  # ratio
        agg = rc.groupby("TICKERSYMBOL")["pm"].std(ddof=0).reset_index().rename(columns={"pm":"pm_std"})
        agg["margin_stability"] = 1 / (1 + agg["pm_std"].abs())
        out = agg.rename(columns={"TICKERSYMBOL":"ticker"})[["ticker","margin_stability"]]
        return out.to_json(orient="records")


    def metric_buyback_yield(self, *args, **kwargs) -> str:
        """자사주 매입율(대략) = (Shares_{t-4} - Shares_{t}) / Shares_{t-4}
        주당가중평균/유통주식수 등 후보 컬럼에서 사용 가능할 때만 계산."""
        share_candidates = ["Common Shares Outstanding", "Diluted Weighted Average Shares"]
        rc = self._recent_quarters(share_candidates, n=4)
        share_col = self._pick_first_col(rc, share_candidates)
        if share_col is None:
            out = pd.DataFrame({"ticker": self.df["TICKERSYMBOL"], "buyback_yield": np.nan})
            return out.to_json(orient="records")

        first = rc.sort_values(["TICKERSYMBOL","QKEY"]).groupby("TICKERSYMBOL")[share_col].first()
        last  = rc.sort_values(["TICKERSYMBOL","QKEY"]).groupby("TICKERSYMBOL")[share_col].last()
        by = self._safe_div_series((first - last), first)
        out = by.reset_index().rename(columns={"TICKERSYMBOL":"ticker", share_col:"buyback_yield"})
        out = out.rename(columns={0:"buyback_yield"}) if 0 in out.columns else out
        return out.to_json(orient="records")


    def metric_capex_intensity(self, *args, **kwargs) -> str:
        """설비집약도 = CapEx_TTM / Revenues_TTM (낮을수록 자본 효율 ↑)"""
        capex_candidates = ["Capital Expenditures (CF)", "Capital Expenditures",
                            "Purchase Of Property Plant Equipment"]
        rev_candidates   = ["Total Revenues", "Revenues"]

        cols = list(set(capex_candidates + rev_candidates))
        rc = self._recent_quarters(cols, n=4)

        capex_col = self._pick_first_col(rc, capex_candidates)
        rev_col   = self._pick_first_col(rc, rev_candidates)
        if capex_col is None or rev_col is None:
            out = pd.DataFrame({"ticker": self.df["TICKERSYMBOL"], "capex_intensity": np.nan})
            return out.to_json(orient="records")

        rc["_capex_abs"] = rc[capex_col].abs()
        agg = rc.groupby("TICKERSYMBOL", as_index=False).agg(
            CAPEX_TTM=("_capex_abs", "sum"),
            REV_TTM=(rev_col, "sum"),
        )
        agg["capex_intensity"] = self._safe_div_series(agg["CAPEX_TTM"], agg["REV_TTM"])
        out = agg.rename(columns={"TICKERSYMBOL":"ticker"})[["ticker","capex_intensity"]]
        return out.to_json(orient="records")



    def _create_agent(self):
        return AgentExecutor(
            agent=create_openai_tools_agent(self.llm, self.tools, self.prompt),
            tools=self.tools,
            verbose=True,
            return_intermediate_steps=True,
        )

    # ===== 결과 저장/로깅 유틸 (Graham과 동일) =====
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

        json_filename = f"{self.results_dir}/buffett_analysis_{start_date}_{end_date}.json"
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2, default=str)

        try:
            portfolio_csv = self.extract_portfolio_table(result["output"])
            if portfolio_csv is not None:
                csv_filename = f"{self.results_dir}/buffett_portfolio_{start_date}_{end_date}.csv"
                portfolio_csv.to_csv(csv_filename, index=False, encoding="utf-8")
                print(f"포트폴리오 결과 저장됨: {csv_filename}")
        except Exception as e:
            print(f"포트폴리오 CSV 저장 실패: {e}")

        log_filename = f"{self.results_dir}/buffett_execution_log_{start_date}_{end_date}.txt"
        with open(log_filename, "w", encoding="utf-8") as f:
            f.write("Buffett Investment Analysis Execution Log\n")
            f.write("=" * 50 + "\n")
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
            f.write("\nFinal Output:\n")
            f.write("=" * 30 + "\n")
            f.write(result["output"])

        print("분석 결과 저장 완료:")
        print(f"  - JSON: {json_filename}")
        print(f"  - 로그: {log_filename}")

        return {
            "json_file": json_filename,
            "log_file": log_filename,
            "csv_file": csv_filename if "csv_filename" in locals() else None,
        }

    def extract_portfolio_table(self, output: str) -> pd.DataFrame:
        try:
            lines = output.split("\n")
            table_lines = []
            in_table = False
            for line in lines:
                if "| Ticker |" in line or "|--------|" in line:
                    in_table = True
                    continue
                elif in_table and line.strip().startswith("|") and "|" in line.strip()[1:]:
                    table_lines.append(line.strip())
                elif in_table and not line.strip().startswith("|"):
                    break

            if table_lines:
                data = []
                for line in table_lines:
                    parts = [part.strip() for part in line.split("|")[1:-1]]
                    if len(parts) >= 4:
                        data.append(parts)
                if data:
                    df = pd.DataFrame(
                        data, columns=["Ticker", "Score", "Weight (%)", "Reason"]
                    )
                    return df
        except Exception as e:
            print(f"테이블 추출 오류: {e}")
        return None

    def load_previous_results(self, limit=10):
        if not os.path.exists(self.results_dir):
            return []
        json_files = [
            f
            for f in os.listdir(self.results_dir)
            if f.startswith("buffett_analysis_") and f.endswith(".json")
        ]
        json_files.sort(reverse=True)
        results = []
        for filename in json_files[:limit]:
            try:
                with open(os.path.join(self.results_dir, filename), "r", encoding="utf-8") as f:
                    data = json.load(f)
                    results.append(
                        {
                            "filename": filename,
                            "timestamp": data.get("timestamp"),
                            "period": f"{data.get('start_date')} to {data.get('end_date')}",
                            "execution_time": data.get("execution_time_seconds"),
                            "steps_count": len(data.get("intermediate_steps", [])),
                        }
                    )
            except Exception as e:
                print(f"결과 파일 로드 실패 {filename}: {e}")
        return results

    def analyze(self, start_date: str, end_date: str) -> str:
        """버핏 규칙 적용 분석 (분기 스냅샷 + 툴 호출 + 점수 산출은 LLM 프롬프트에서 수행)"""
        start_time = datetime.now()

        # 원본 보관
        self._original_df = self.df.copy()

        def _to_quarter(s: str) -> str:
            s = str(s).strip()
            if re.fullmatch(r"\d{4}Q[1-4]", s):
                return s
            dt = pd.to_datetime(s)
            q = (dt.month - 1) // 3 + 1
            return f"{dt.year}Q{q}"

        q_end = _to_quarter(end_date)
        self._q_end = q_end
        print(f"[Buffett] start_date: {start_date}, end_date: {end_date}, quarter: {q_end}")

        # 해당 분기만 필터
        if "QUARTER" in self.df.columns:
            self.df = self.df[self.df["QUARTER"] == q_end].copy()

        # 가격 스냅샷 & TTM/BVPS 부착
        self._pxsnap = self._build_price_snapshot(q_end)
        self._attach_ttm_and_bvps(q_end)

        try:
            request = f"""
Analyze these stocks as Warren Buffett (rules in system prompt):
- Period start: {start_date}
- Period end: {q_end}
"""
            result = self.agent.invoke({"input": request})

            execution_time = (datetime.now() - start_time).total_seconds()

            # 중간 단계 로깅
            steps = result.get("intermediate_steps", [])
            for action, observation in steps:
                print(f"Action: {getattr(action, 'tool', action)}")
                print(f"Observation: {str(observation)[:500]}...")
                print("-" * 100)

            # 저장
            if self.save_results:
                self.save_analysis_results(start_date, end_date, result, execution_time)
                print(f"실행 시간: {execution_time:.2f}초")

            return result["output"]
        finally:
            # 복원
            self.df = self._original_df
            if hasattr(self, "_pxsnap"):
                delattr(self, "_pxsnap")

