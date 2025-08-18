# langchain 기반 그린블랫(매직 포뮬라) 에이전트
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
from typing import List
from datetime import datetime


template = r"""
## Role
You are **Joel Greenblatt**, author of *The Little Book That Beats the Market* and creator of the **Magic Formula**.
Your core ideas:
- Rank companies by two metrics: **Earnings Yield** (≈ EBIT / Enterprise Value) and **Return on Capital** (≈ EBIT / (Net Working Capital + Net PPE)).
- Prefer **simple, rules-based** selection; evaluate results at the **portfolio** level.
- Avoid over-forecasting; lean on **current operating performance** and **rational prices**.
- Exclude firms with **negative EBIT** or nonsensical denominators (e.g., EV ≤ 0, capital ≤ 0).

Your tone is practical, rules-driven, and disciplined.

## Data (tabular fundamentals)
- One row per ticker for a quarter; identifiers: `TICKERSYMBOL`, `QUARTER` (e.g., “2023Q4”).
- Prices/market cap come from a separate daily OHLCV table; we take a **quarter-end snapshot** (or last trading day ≤ quarter-end).
- Metrics may be missing; treat divide-by-zero or invalid denominators as NA.

## Tools (call BEFORE ranking; once each on the full DataFrame)
- `metric_earnings_yield(df)` → [{{ticker, ebit_ttm, ev, earnings_yield}}]
- `metric_roic(df)` → [{{ticker, roic}}]
- `metric_safety(df)` → [{{ticker, interest_coverage, debt_to_equity}}]
- `metric_size_liquidity(df)` → [{{ticker, price, mktcap}}]
Use only these tool outputs to build the per-ticker metrics table.

### Metric definitions (use tool outputs only)
- **Earnings Yield** = EBIT_TTM / EV.
- **EV** = MarketCap + Debt − Cash & Equivalents (if Debt not available, use a conservative proxy; if EV ≤ 0 → NA).
- **ROIC** = EBIT_TTM / ( (Current Assets − Cash − Current Liabilities) + Net PPE ).
  - Net PPE is preferred; if missing, approximate as Total Assets − Current Assets − (Goodwill + Other Intangibles).

## Scoring & Portfolio (deterministic)
- **Eligibility**: earnings_yield>0 and roic>0 and EV>0 (drop others as NA).
- **Ranking**:
  - Rank by Earnings Yield (desc) → rank_EY
  - Rank by ROIC (desc) → rank_ROIC
  - CombinedRank = rank_EY + rank_ROIC  (lower is better)
- **Score in [0,1]**:
  - Let N = number of eligible tickers.
  - If N==1 → Score=1.00.
  - Else Score = 1 - (CombinedRank - 2) / (2*N - 2).
- **Safety nudges (small, bounded)**:
  - If interest_coverage < 3 → Score -= 0.03
  - If debt_to_equity > 1.0 → Score -= 0.03
  - Clip to [0,1].
- **Selection**: choose top **K = min(30, ceil(0.3·N))** by Score. If N<15, include all eligible.
- **Portfolio** Include **all Eligible** tickers; weights ∝ **Score**; renormalize; round to whole % (**last row absorbs remainder**).  
- **Tie-breakers**: higher Earnings Yield, then higher ROIC, then ticker alphabetical.

## Output (STRICT)
Return **only** this markdown table:

| Ticker | Score | Weight (%) | Reason |
|--------|-------|------------|--------|

- Score: 2 decimals in [0.00, 1.00].
- Weight: integers summing to 100.
- Reason: one short sentence (e.g., "high EY & ROIC; mild D/E penalty").

Think step-by-step privately, calling the above tools before ranking.
"""


class GreenblattInvestmentAnalyzer:
    """LLM + Tools 기반 Greenblatt(Magic Formula) 투자 분석기"""

    def __init__(
        self,
        llm=None,
        save_results=True,
        results_dir="C:/Users/unist/Desktop/MOE-Invest/MoE-Invest/results/greenblatt_agent",
        fundamentals_csv="C:/Users/unist/Desktop/MOE-Invest/MoE-Invest/data/nasdaq100_bs_cf_is.csv",
        ohlcv_csv="C:/Users/unist/Desktop/MOE-Invest/MoE-Invest/data/nasdaq100_ohlcv.csv",
    ):
        self.llm = llm or ChatOpenAI(temperature=0, model="gpt-4o")
        self.name = "Greenblatt Magic Formula Analyzer"
        self.save_results = save_results
        self.results_dir = results_dir

        # 결과 저장 디렉토리
        if self.save_results:
            os.makedirs(self.results_dir, exist_ok=True)

        # 데이터 적재
        self.df = pd.read_csv(fundamentals_csv)
        self.df_ohlcv = pd.read_csv(ohlcv_csv)
        self.df_ohlcv["EVAL_D"] = pd.to_datetime(self.df_ohlcv["EVAL_D"])

        # 프롬프트
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", template),
                ("user", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )

        # 툴 정의
        self.tools = [
            Tool(
                name="metric_earnings_yield",
                description="Compute EBIT_TTM, EV, Earnings Yield (EBIT_TTM/EV). Returns JSON records [{ticker, ebit_ttm, ev, earnings_yield}].",
                func=self.metric_earnings_yield,
            ),
            Tool(
                name="metric_roic",
                description="Compute ROIC = EBIT_TTM / ( (CA - Cash - CL) + NetPPE_approx ). Returns JSON records [{ticker, roic}].",
                func=self.metric_roic,
            ),
            Tool(
                name="metric_safety",
                description="Compute safety metrics: Interest Coverage (EBIT/|Interest Expense|), Debt-to-Equity. Returns JSON records [{ticker, interest_coverage, debt_to_equity}].",
                func=self.metric_safety,
            ),
            Tool(
                name="metric_size_liquidity",
                description="Return price snapshot & size: [{ticker, price, mktcap}].",
                func=self.metric_size_liquidity,
            ),
        ]

        self.agent = self._create_agent()

    # -------------------------
    # Helpers
    # -------------------------
    def safe_div(self, a, b):
        s = pd.to_numeric(a, errors="coerce") / pd.to_numeric(b, errors="coerce")
        return s.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _quarter_end_date(q: str) -> pd.Timestamp:
        # "YYYYQx" → quarter-end date
        y = int(q[:4])
        qi = int(q[-1])
        month_end = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}[qi]
        return pd.Timestamp(year=y, month=month_end[0], day=month_end[1])

    def _build_price_snapshot(self, q_end: str) -> pd.DataFrame:
        """
        분기말(또는 그 이전 최근 거래일)의 가격/시총/주식수 스냅샷.
        우선순위: DIV_ADJ_CLOSE → CLOSE_
        MKTCAP이 없으면 price * NUM_SHARES로 추정.
        """
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

    def _attach_ttm_and_bvps(self, q_end: str) -> None:
        """
        self.df에 TTM 지표(EBIT_TTM, NET_INCOME_TTM)와 자본 관련 보조치(BVPS 등)를 부착.
        """
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

        valid_quarters = base["QUARTER"].notna() & (base["QUARTER"].astype(str).str.contains("Q", na=False))
        base_filtered = base[valid_quarters].copy()

        recent4 = base_filtered[base_filtered["QUARTER"].apply(lambda x: _q_key(x) <= end_key and _q_key(x) > 0)].copy()
        if recent4.empty:
            # 기본 NaN 채우기
            self.df = self.df.assign(
                NET_INCOME_TTM=np.nan,
                EBIT_TTM=np.nan,
                EQUITY_END=np.nan,
                NUM_SHARES_SNP=np.nan,
                EPS_TTM=np.nan,
                BVPS=np.nan,
            )
            return

        recent4["QKEY"] = recent4["QUARTER"].apply(_q_key)
        recent4 = recent4.sort_values(["TICKERSYMBOL", "QKEY"]).groupby("TICKERSYMBOL").tail(4)

        # TTM Net Income
        ni_col = "Net Income - (IS)" if "Net Income - (IS)" in recent4.columns else "Net Income"
        ttm_ni = recent4.groupby("TICKERSYMBOL", as_index=False)[ni_col].sum().rename(columns={ni_col: "NET_INCOME_TTM"})

        # TTM EBIT (fallbacks)
        if "EBIT" in recent4.columns:
            ttm_ebit = recent4.groupby("TICKERSYMBOL", as_index=False)["EBIT"].sum().rename(columns={"EBIT": "EBIT_TTM"})
        else:
            gp = recent4.get("Gross Profit (As Reported)")
            oe = recent4.get("Operating Expenses (As Reported)")
            if gp is not None and oe is not None:
                recent4["_EBIT_ALT"] = pd.to_numeric(gp, errors="coerce") - pd.to_numeric(oe, errors="coerce")
                ttm_ebit = (
                    recent4.groupby("TICKERSYMBOL", as_index=False)["_EBIT_ALT"].sum().rename(columns={"_EBIT_ALT": "EBIT_TTM"})
                )
            else:
                ttm_ebit = pd.DataFrame({"TICKERSYMBOL": recent4["TICKERSYMBOL"].unique(), "EBIT_TTM": np.nan})

        # Quarter-end equity for BVPS
        end_quarter_data = base_filtered[base_filtered["QUARTER"] == q_end]
        if end_quarter_data.empty:
            eq = pd.DataFrame(columns=["TICKERSYMBOL", "EQUITY_END"])
        else:
            eq = end_quarter_data[["TICKERSYMBOL", "Total Shareholders Equity (As Reported)"]].rename(
                columns={"Total Shareholders Equity (As Reported)": "EQUITY_END"}
            )

        shares = (
            self._pxsnap[["TICKERSYMBOL", "NUM_SHARES_SNP"]]
            if hasattr(self, "_pxsnap") and not self._pxsnap.empty
            else pd.DataFrame(columns=["TICKERSYMBOL", "NUM_SHARES_SNP"])
        )

        self.df = (
            self.df.merge(ttm_ni, on="TICKERSYMBOL", how="left")
            .merge(ttm_ebit, on="TICKERSYMBOL", how="left")
            .merge(eq, on="TICKERSYMBOL", how="left")
            .merge(shares, on="TICKERSYMBOL", how="left")
        )

        def _sd(a, b):
            s = pd.to_numeric(a, errors="coerce") / pd.to_numeric(b, errors="coerce")
            return s.replace([np.inf, -np.inf], np.nan)

        self.df["EPS_TTM"] = _sd(self.df.get("NET_INCOME_TTM"), self.df.get("NUM_SHARES_SNP"))
        self.df["BVPS"] = _sd(self.df.get("EQUITY_END"), self.df.get("NUM_SHARES_SNP"))

    def _create_agent(self):
        agent = create_openai_tools_agent(self.llm, self.tools, self.prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True, return_intermediate_steps=True)

    # -------------------------
    # Tool implementations
    # -------------------------
    def _build_debt_proxy(self, merged: pd.DataFrame) -> pd.Series:
        """
        채무(이자부부채) 근사. 우선순위:
        1) 'Total Debt' 또는 동의어
        2) 'Short Term Debt' + 'Long Term Debt' 조합
        3) 보수적 프록시: Total Liabilities - Total Current Liabilities (>=0로 클립)
        """
        # 1) Total Debt 필드
        for td in [
            "Total Debt",
            "Total Debt & Capital Lease Obligations",
            "Total Debt (All) (includes lease)",
        ]:
            if td in merged.columns:
                s = pd.to_numeric(merged[td], errors="coerce")
                return s.clip(lower=0)

        # 2) ST + LT 조합
        st_alts = [
            "Short Term Debt",
            "Short Term Debt & Current Portion of Long Term Debt",
            "Current Portion of Long Term Debt",
        ]
        lt_alts = [
            "Long Term Debt",
            "Long Term Debt And Capital Lease Obligation",
        ]
        st = None
        for c in st_alts:
            if c in merged.columns:
                st = pd.to_numeric(merged[c], errors="coerce") if st is None else st.combine_first(
                    pd.to_numeric(merged[c], errors="coerce")
                )
        lt = None
        for c in lt_alts:
            if c in merged.columns:
                lt = pd.to_numeric(merged[c], errors="coerce") if lt is None else lt.combine_first(
                    pd.to_numeric(merged[c], errors="coerce")
                )
        if st is not None or lt is not None:
            st = st if st is not None else 0.0
            lt = lt if lt is not None else 0.0
            return (st + lt).clip(lower=0)

        # 3) 보수적 프록시
        tl = merged.get("Total Liabilities - (Standard / Utility Template)")
        cl = merged.get("Total Current Liabilities")
        if tl is not None and cl is not None:
            proxy = pd.to_numeric(tl, errors="coerce") - pd.to_numeric(cl, errors="coerce")
            return proxy.clip(lower=0)

        return pd.Series(np.nan, index=merged.index)

    def metric_earnings_yield(self, *args, **kwargs) -> str:
        """
        EBIT_TTM, EV, Earnings Yield 계산.
        Returns JSON records: [{ticker, ebit_ttm, ev, earnings_yield}]
        """
        if not hasattr(self, "_pxsnap"):
            return json.dumps([])

        merged = self.df.merge(self._pxsnap, on="TICKERSYMBOL", how="left")
        price = merged["PX"]
        mktcap = merged["MKTCAP_SNP"]
        cash = pd.to_numeric(merged.get("Cash And Equivalents"), errors="coerce")

        debt = self._build_debt_proxy(merged)
        ev = pd.to_numeric(mktcap, errors="coerce") + debt.fillna(0) - cash.fillna(0)

        ebit_ttm = pd.to_numeric(merged.get("EBIT_TTM"), errors="coerce")
        # 음수 EV 또는 0 EV는 사용 불가
        ev_valid = ev.where(ev > 0, np.nan)
        earnings_yield = self.safe_div(ebit_ttm, ev_valid) * 100.0  # percent 기준

        out = pd.DataFrame(
            {
                "ticker": merged["TICKERSYMBOL"],
                "ebit_ttm": ebit_ttm,
                "ev": ev_valid,
                "earnings_yield": earnings_yield,
            }
        )
        return out.to_json(orient="records")

    def metric_roic(self, *args, **kwargs) -> str:
        """
        ROIC = EBIT_TTM / ( (CA - Cash - CL) + NetPPE_approx )
        NetPPE_approx 우선순위:
        1) 'Net Property Plant And Equipment' 또는 동의어
        2) 대안: Total Assets − Current Assets − (Goodwill + Other Intangible Assets)
        Returns JSON: [{ticker, roic}]
        """
        ca = pd.to_numeric(self.df.get("Total Current Assets"), errors="coerce")
        cash = pd.to_numeric(self.df.get("Cash And Equivalents"), errors="coerce")
        cl = pd.to_numeric(self.df.get("Total Current Liabilities"), errors="coerce")

        # Net PPE - 실제 데이터의 컬럼명 사용
        net_ppe = None
        for c in [
            "Net Property Plant And Equipment",  # 실제 데이터 컬럼명
            "Property/Plant/Equipment, Net",
            "Property, Plant & Equipment (Net)",
            "Property, Plant & Equipment, Net",
        ]:
            if c in self.df.columns:
                net_ppe = pd.to_numeric(self.df[c], errors="coerce")
                break

        if net_ppe is None:
            ta = pd.to_numeric(self.df.get("Total Assets"), errors="coerce")
            goodwill = pd.to_numeric(self.df.get("Goodwill"), errors="coerce").fillna(0)
            other_intangibles = pd.to_numeric(self.df.get("Other Intangible Assets"), errors="coerce").fillna(0)
            net_ppe = ta - ca - (goodwill + other_intangibles)

        nwc_ex_cash = ca - cash - cl
        capital = (nwc_ex_cash + net_ppe)
        capital = capital.where(capital > 0, np.nan)

        ebit_ttm = pd.to_numeric(self.df.get("EBIT_TTM"), errors="coerce")
        roic = self.safe_div(ebit_ttm, capital) * 100.0  # percent

        out = pd.DataFrame({"ticker": self.df["TICKERSYMBOL"], "roic": roic})
        return out.to_json(orient="records")

    def metric_safety(self, *args, **kwargs) -> str:
        """
        안전성 보조지표:
        - Interest Coverage = EBIT / |Interest Expense|
        - Debt-to-Equity = Total Liabilities / Total Shareholders Equity
        Returns JSON: [{ticker, interest_coverage, debt_to_equity}]
        """
        # Interest Coverage (EBIT_TTM / |Interest Expense|)
        # Interest Expense가 분기 단위일 가능성이 있어 4배 근사 고려 없이 그대로 사용 (랭킹 영향 미미)
        ebit = pd.to_numeric(self.df.get("EBIT_TTM"), errors="coerce")
        interest_expense = pd.to_numeric(self.df.get("Interest Expense"), errors="coerce").abs()
        ic = self.safe_div(ebit, interest_expense)

        # D/E
        tl = pd.to_numeric(self.df.get("Total Liabilities - (Standard / Utility Template)"), errors="coerce")
        te = pd.to_numeric(self.df.get("Total Shareholders Equity (As Reported)"), errors="coerce")
        de = self.safe_div(tl, te)

        out = pd.DataFrame(
            {
                "ticker": self.df["TICKERSYMBOL"],
                "interest_coverage": ic,
                "debt_to_equity": de,
            }
        )
        return out.to_json(orient="records")

    def metric_size_liquidity(self, *args, **kwargs) -> str:
        """
        가격/시총 스냅샷 (참고용).
        Returns JSON: [{ticker, price, mktcap}]
        """
        if not hasattr(self, "_pxsnap") or self._pxsnap is None or self._pxsnap.empty:
            return json.dumps([])

        px = self._pxsnap.rename(columns={"PX": "price", "MKTCAP_SNP": "mktcap"})[
            ["TICKERSYMBOL", "price", "mktcap"]
        ].copy()
        px.columns = ["ticker", "price", "mktcap"]
        return px.to_json(orient="records")

    # -------------------------
    # Utility: result persistence
    # -------------------------
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

        json_filename = f"{self.results_dir}/greenblatt_analysis_{start_date}_{end_date}.json"
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2, default=str)

        try:
            portfolio_csv = self.extract_portfolio_table(result["output"])
            if portfolio_csv is not None:
                csv_filename = f"{self.results_dir}/greenblatt_portfolio_{start_date}_{end_date}.csv"
                portfolio_csv.to_csv(csv_filename, index=False, encoding="utf-8")
                print(f"포트폴리오 결과 저장됨: {csv_filename}")
        except Exception as e:
            print(f"포트폴리오 CSV 저장 실패: {e}")

        log_filename = f"{self.results_dir}/greenblatt_execution_log_{start_date}_{end_date}.txt"
        with open(log_filename, "w", encoding="utf-8") as f:
            f.write("Greenblatt (Magic Formula) Investment Analysis Execution Log\n")
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
            "csv_file": (f"{self.results_dir}/greenblatt_portfolio_{start_date}_{end_date}.csv")
            if os.path.exists(f"{self.results_dir}/greenblatt_portfolio_{start_date}_{end_date}.csv")
            else None,
        }

    def extract_portfolio_table(self, output: str) -> pd.DataFrame | None:
        try:
            lines = output.split("\n")
            table_lines = []
            in_table = False
            header_seen = False

            for line in lines:
                if line.strip().startswith("| Ticker |") and not header_seen:
                    header_seen = True
                    in_table = True
                    continue
                if in_table and line.strip().startswith("|--------|"):
                    # skip delimiter row
                    continue
                if in_table and line.strip().startswith("|") and "|" in line.strip()[1:]:
                    table_lines.append(line.strip())
                elif in_table and not line.strip().startswith("|"):
                    break

            if table_lines:
                data = []
                for line in table_lines:
                    parts = [p.strip() for p in line.split("|")[1:-1]]
                    if len(parts) >= 4:
                        data.append(parts[:4])
                if data:
                    df = pd.DataFrame(data, columns=["Ticker", "Score", "Weight (%)", "Reason"])
                    return df
        except Exception as e:
            print(f"테이블 추출 오류: {e}")
        return None

    # -------------------------
    # Public: main entry
    # -------------------------
    def analyze(self, start_date: str, end_date: str) -> str:
        """
        실행: end_date 기준 분기 스냅샷으로 Magic Formula 랭킹 & 포트폴리오 산출.
        """
        start_time = datetime.now()

        # 원본 백업
        self._original_df = self.df.copy()

        # "YYYYQx" 정규화
        def _to_quarter(s: str) -> str:
            s = str(s).strip()
            if re.fullmatch(r"\d{4}Q[1-4]", s):
                return s
            dt = pd.to_datetime(s)
            q = (dt.month - 1) // 3 + 1
            return f"{dt.year}Q{q}"

        q_end = _to_quarter(end_date)
        print(f"[Greenblatt] start_date: {start_date}, end_date: {end_date}, quarter: {q_end}")

        # 해당 분기만 필터
        if "QUARTER" in self.df.columns:
            self.df = self.df[self.df["QUARTER"] == q_end].copy()

        # 가격 스냅샷
        self._pxsnap = self._build_price_snapshot(q_end)

        # TTM 부착
        self._attach_ttm_and_bvps(q_end)

        try:
            request = f"""
Analyze these stocks using Greenblatt's Magic Formula:
- Period start: {start_date}
- Period end: {q_end}
"""
            result = self.agent.invoke({"input": request})

            # 중간 단계 로깅
            steps = result.get("intermediate_steps", [])
            for action, observation in steps:
                print(f"Action: {getattr(action, 'tool', action)}")
                print(f"Observation: {str(observation)[:500]}{'...' if len(str(observation))>500 else ''}")
                print("-" * 100)

            # 결과 저장
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            if self.save_results:
                self.save_analysis_results(start_date, end_date, result, execution_time)
                print(f"실행 시간: {execution_time:.2f}초")

            return result["output"]

        finally:
            # 복원
            self.df = self._original_df
            if hasattr(self, "_pxsnap"):
                delattr(self, "_pxsnap")

