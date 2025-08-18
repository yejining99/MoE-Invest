# langchain 기반 에드워드 알트만(Z-Score) 에이전트
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
You are **Edward Altman**, creator of the **Z-Score** models for default risk.
You apply a rules-based approach to estimate financial distress using accounting ratios.
You do **not** forecast; you classify firms into **Safe / Grey / Distress** zones based on Z variants.

### Altman Variants & Cutoffs (use whichever fits the available data best)
- **Z (1968, public manufacturing)**:  
  Z = 1.2·(WC/TA) + 1.4·(RE/TA) + 3.3·(EBIT/TA) + 0.6·(MVE/TL) + 1.0·(Sales/TA)  
  Zones: **Distress < 1.81**, **Grey 1.81–2.99**, **Safe > 2.99**.
- **Z′ (private manufacturing)**:  
  Z′ = 0.717·(WC/TA) + 0.847·(RE/TA) + 3.107·(EBIT/TA) + 0.420·(MVE/TL) + 0.998·(Sales/TA)  
  Zones: **Distress < 1.23**, **Grey 1.23–2.90**, **Safe > 2.90**.
- **Z″ (non-manufacturing / services)**:  
  Z″ = 6.56·(WC/TA) + 3.26·(RE/TA) + 6.72·(EBIT/TA) + 1.05·(BVE/TL)  
  Zones: **Distress < 1.10**, **Grey 1.10–2.60**, **Safe > 2.60**.

**Variables**
- WC = Current Assets − Current Liabilities
- TA = Total Assets
- RE = Retained Earnings (Accumulated Deficit allowed)
- EBIT = Earnings Before Interest and Taxes (use **TTM**)
- MVE = Market Value of Equity (market cap snapshot at quarter-end)
- TL = Total Liabilities
- Sales = Revenues (use **TTM**)
- BVE = Book Value of Equity (Total Shareholders' Equity)

Your tone is conservative and diagnostic.

## Data
- Fundamentals are quarterly with `TICKERSYMBOL` and `QUARTER` (e.g., "2024Q4").
- Price/share-count from a daily OHLCV table; take a **quarter-end snapshot** (latest trading day ≤ quarter-end).
- Use **TTM** sums for EBIT and Sales by summing the last 4 quarters up to the evaluation quarter.
- Divide-by-zero or invalid denominators ⇒ NA (not zero). Do not impute.

## Tools (call BEFORE ranking; once each on the full DataFrame)
- `metric_altman(df)` → [
  {{ticker, model, z_score, band, wc_ta, re_ta, ebit_ta, mve_tl, sales_ta, bve_tl}}
]
- `metric_extras(df)` → [
  {{ticker, interest_coverage, debt_to_equity, price, mktcap}}
]

Use only these tool outputs to construct the per-ticker table.

## Scoring & Portfolio (deterministic)
- **Eligibility**: z_score is not NA.
- **Normalized Score** in [0,1] using the model's cutoffs:
  - For Z:     score = clip((Z − 1.81) / (2.99 − 1.81), 0, 1)
  - For Z′:    score = clip((Z′ − 1.23) / (2.90 − 1.23), 0, 1)
  - For Z″:    score = clip((Z″ − 1.10) / (2.60 − 1.10), 0, 1)
- **Primary ranking**: higher score (i.e., further into Safe).
- **Tie-breakers**: higher z_score, then higher ebit_ta, then lower debt_to_equity, then ticker alphabetical.
- **Selection**:
  - Include all **Safe** names first; if < 15, add from **Grey** by score until **K = min(30, ceil(0.3·N))** (if N<15, include all eligible).
- **Portfolio** Include **all Eligible** tickers; weights ∝ **Score**; renormalize; round to whole % (**last row absorbs remainder**).  

## Output (STRICT)
Return **only** this markdown table:

| Ticker | Score | Weight (%) | Reason |
|--------|-------|------------|--------|

- Score: 2 decimals in [0.00, 1.00].
- Weight: integers summing to 100.
- Reason: one short sentence (e.g., "Z′=3.1 Safe; strong EBIT/TA; modest D/E").

Think step-by-step privately, calling the above tools before ranking.
"""


class AltmanInvestmentAnalyzer:
    """LLM + Tools 기반 Altman(Z-Score) 투자 분석기"""

    def __init__(
        self,
        llm=None,
        save_results: bool = True,
        results_dir: str = "C:/Users/unist/Desktop/MOE-Invest/MoE-Invest/results/altman_agent",
        fundamentals_csv: str = "C:/Users/unist/Desktop/MOE-Invest/MoE-Invest/data/nasdaq100_bs_cf_is.csv",
        ohlcv_csv: str = "C:/Users/unist/Desktop/MOE-Invest/MoE-Invest/data/nasdaq100_ohlcv.csv",
    ):
        self.llm = llm or ChatOpenAI(temperature=0, model="gpt-4o")
        self.name = "Altman Z-Score Analyzer"
        self.save_results = save_results
        self.results_dir = results_dir

        if self.save_results:
            os.makedirs(self.results_dir, exist_ok=True)

        # Load data
        self.df = pd.read_csv(fundamentals_csv)
        self.df_ohlcv = pd.read_csv(ohlcv_csv)
        self.df_ohlcv["EVAL_D"] = pd.to_datetime(self.df_ohlcv["EVAL_D"])

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
                name="metric_altman",
                description="Compute Altman Z / Z′ / Z″ using available inputs. Returns JSON records [{ticker, model, z_score, band, wc_ta, re_ta, ebit_ta, mve_tl, sales_ta, bve_tl}].",
                func=self.metric_altman,
            ),
            Tool(
                name="metric_extras",
                description="Return safety extras: Interest Coverage (EBIT_TTM/|Interest Expense|), D/E, and size [{ticker, interest_coverage, debt_to_equity, price, mktcap}].",
                func=self.metric_extras,
            ),
        ]

        self.agent = self._create_agent()

    # -------------------------
    # Helpers
    # -------------------------
    @staticmethod
    def _to_quarter(s: str) -> str:
        s = str(s).strip()
        if re.fullmatch(r"\d{4}Q[1-4]", s):
            return s
        dt = pd.to_datetime(s)
        q = (dt.month - 1) // 3 + 1
        return f"{dt.year}Q{q}"

    @staticmethod
    def _quarter_end_date(q: str) -> pd.Timestamp:
        y = int(q[:4]); qi = int(q[-1])
        month_end = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}[qi]
        return pd.Timestamp(year=y, month=month_end[0], day=month_end[1])

    @staticmethod
    def _safe_div(a, b):
        s = pd.to_numeric(a, errors="coerce") / pd.to_numeric(b, errors="coerce")
        return s.replace([np.inf, -np.inf], np.nan)

    def _col(self, df: pd.DataFrame, names: list[str], default=np.nan) -> pd.Series:
        for n in names:
            if n in df.columns:
                return pd.to_numeric(df[n], errors="coerce")
        return pd.Series(default, index=df.index if len(df) else pd.RangeIndex(0), dtype="float64")

    def _build_price_snapshot(self, q_end: str) -> pd.DataFrame:
        """
        Quarter-end (or last trading day ≤ quarter-end): price, shares, market cap snapshot.
        """
        if self.df_ohlcv is None:
            return pd.DataFrame(columns=["TICKERSYMBOL","PX","NUM_SHARES_SNP","MKTCAP_SNP","PX_DATE"])

        qt_end = self._quarter_end_date(q_end)
        px = self.df_ohlcv[self.df_ohlcv["EVAL_D"] <= qt_end].copy()
        if px.empty:
            return pd.DataFrame(columns=["TICKERSYMBOL","PX","NUM_SHARES_SNP","MKTCAP_SNP","PX_DATE"])

        px["PX"] = px["DIV_ADJ_CLOSE"].fillna(px["CLOSE_"])
        px = px.sort_values(["TICKERSYMBOL","EVAL_D"]).drop_duplicates(["TICKERSYMBOL"], keep="last")
        
        print(f"[DEBUG] Unique tickers in price snapshot: {len(px)}")
        print(f"[DEBUG] Sample tickers: {px['TICKERSYMBOL'].head().tolist()}")
        
        # Market cap calculation with better handling
        px["NUM_SHARES"] = pd.to_numeric(px["NUM_SHARES"], errors="coerce")
        px["MKTCAP"] = pd.to_numeric(px["MKTCAP"], errors="coerce")
        
        est_mktcap = px["PX"] * px["NUM_SHARES"]
        px["MKTCAP_SNP"] = px["MKTCAP"].fillna(est_mktcap)
        
        # Check for valid market cap data
        valid_mktcap = px["MKTCAP_SNP"].notna().sum()
        print(f"[DEBUG] Valid market cap entries: {valid_mktcap}/{len(px)}")
        
        out = px.rename(columns={"NUM_SHARES":"NUM_SHARES_SNP","EVAL_D":"PX_DATE"})[
            ["TICKERSYMBOL","PX","NUM_SHARES_SNP","MKTCAP_SNP","PX_DATE"]
        ].reset_index(drop=True)
        return out

    def _attach_ttm(self, q_end: str) -> None:
        """
        Attach TTM EBIT and TTM Sales (Revenues) and quarter-end Equity for ratios.
        """
        base = self._original_df.copy() if hasattr(self, "_original_df") else self.df.copy()

        def _q_key(q):
            try:
                if pd.isna(q): return -1
                q = str(q).strip()
                if "Q" not in q: return -1
                y, qi = int(q[:4]), int(q[-1])
                return y*4 + qi
            except:
                return -1

        end_key = _q_key(q_end)
        valid = base["QUARTER"].notna() & (base["QUARTER"].astype(str).str.contains("Q", na=False))
        base = base[valid].copy()

        # Last 4 quarters up to q_end
        recent = base[base["QUARTER"].apply(lambda x: (k:=_q_key(x))>0 and k<=end_key)].copy()
        if recent.empty:
            # Ensure columns exist
            self.df = self.df.assign(EBIT_TTM=np.nan, REV_TTM=np.nan, EQUITY_END=np.nan)
            return

        recent["QKEY"] = recent["QUARTER"].apply(_q_key)
        recent4 = recent.sort_values(["TICKERSYMBOL","QKEY"]).groupby("TICKERSYMBOL").tail(4)

        # EBIT (with fallback GP - Opex)
        if "EBIT" in recent4.columns:
            ebit_ttm = recent4.groupby("TICKERSYMBOL", as_index=False)["EBIT"].sum().rename(columns={"EBIT":"EBIT_TTM"})
        else:
            gp = pd.to_numeric(recent4.get("Gross Profit (As Reported)"), errors="coerce")
            oe = pd.to_numeric(recent4.get("Operating Expenses (As Reported)"), errors="coerce")
            recent4["_EBIT_ALT"] = gp - oe
            ebit_ttm = recent4.groupby("TICKERSYMBOL", as_index=False)["_EBIT_ALT"].sum().rename(columns={"_EBIT_ALT":"EBIT_TTM"})

        # Revenues TTM
        rev_col = "Total Revenues" if "Total Revenues" in recent4.columns else ("Revenues" if "Revenues" in recent4.columns else None)
        if rev_col:
            rev_ttm = recent4.groupby("TICKERSYMBOL", as_index=False)[rev_col].sum().rename(columns={rev_col:"REV_TTM"})
        else:
            rev_ttm = pd.DataFrame({"TICKERSYMBOL": recent4["TICKERSYMBOL"].unique(), "REV_TTM": np.nan})

        # Quarter-end equity snapshot (for BVE/TL and D/E)
        end_quarter = base[base["QUARTER"] == q_end]
        if end_quarter.empty:
            eq = pd.DataFrame(columns=["TICKERSYMBOL","EQUITY_END"])
        else:
            eq = end_quarter[["TICKERSYMBOL","Total Shareholders Equity (As Reported)"]].rename(
                columns={"Total Shareholders Equity (As Reported)":"EQUITY_END"}
            )

        self.df = (self.df.merge(ebit_ttm, on="TICKERSYMBOL", how="left")
                        .merge(rev_ttm, on="TICKERSYMBOL", how="left")
                        .merge(eq, on="TICKERSYMBOL", how="left"))

    # -------------------------
    # Tools
    # -------------------------
    def metric_altman(self, *args, **kwargs) -> str:
        """
        Compute Altman Z / Z′ / Z″ and choose the best-fit model given available inputs:
        Returns JSON records:
        [{ticker, model, z_score, band, wc_ta, re_ta, ebit_ta, mve_tl, sales_ta, bve_tl}]
        """
        if not hasattr(self, "_q_end") or not hasattr(self, "_pxsnap"):
            return json.dumps([])

        # Work with quarter universe (self.df already filtered in analyze)
        df = self.df.copy()

        # Required base columns / series
        ca = self._col(df, ["Total Current Assets"])
        cl = self._col(df, ["Total Current Liabilities"])
        ta = self._col(df, ["Total Assets", "Total Assets (As Reported)"])
        tl = self._col(df, ["Total Liabilities - (Standard / Utility Template)", "Total Liabilities"])
        re = self._col(df, ["Retained Earnings", "Retained Earnings (Accumulated Deficit)"])
        ebit_ttm = pd.to_numeric(df.get("EBIT_TTM"), errors="coerce")
        rev_ttm = pd.to_numeric(df.get("REV_TTM"), errors="coerce")
        bve = pd.to_numeric(df.get("EQUITY_END"), errors="coerce")

        # WC/TA and re/ta etc.
        wc_ta = self._safe_div(ca - cl, ta)
        re_ta = self._safe_div(re, ta)
        ebit_ta = self._safe_div(ebit_ttm, ta)
        sales_ta = self._safe_div(rev_ttm, ta)
        bve_tl = self._safe_div(bve, tl)

        
        # 더 안전한 병합 방식 사용
        df_with_px = df.merge(self._pxsnap[["TICKERSYMBOL", "MKTCAP_SNP"]], 
                             on="TICKERSYMBOL", how="left")
        mve = pd.to_numeric(df_with_px["MKTCAP_SNP"], errors="coerce")
        
        print(f"[DEBUG] MVE non-null count: {mve.notna().sum()}/{len(mve)}")
        if mve.notna().sum() > 0:
            print(f"[DEBUG] MVE sample values: {mve.dropna().head().tolist()}")
        
        mve_tl = self._safe_div(mve, tl)

        # Compute each Z variant where possible
        z_public = (1.2*wc_ta + 1.4*re_ta + 3.3*ebit_ta + 0.6*mve_tl + 1.0*sales_ta)
        z_private = (0.717*wc_ta + 0.847*re_ta + 3.107*ebit_ta + 0.420*mve_tl + 0.998*sales_ta)
        z_nonmfg = (6.56*wc_ta + 3.26*re_ta + 6.72*ebit_ta + 1.05*bve_tl)

        # Choose model: prefer Z (public) if all terms exist; else Z′ if Sales+RE exist; else Z″ if RE exists; else NA
        has_public = wc_ta.notna() & re_ta.notna() & ebit_ta.notna() & mve_tl.notna() & sales_ta.notna()
        has_private = wc_ta.notna() & re_ta.notna() & ebit_ta.notna() & mve_tl.notna() & sales_ta.notna()
        has_nonmfg = wc_ta.notna() & re_ta.notna() & ebit_ta.notna() & bve_tl.notna()

        # Create chosen_model using pandas Series with object dtype to handle strings and None
        chosen_model = pd.Series(None, index=df.index, dtype="object")
        chosen_model = chosen_model.where(~has_public, "Z")
        chosen_model = chosen_model.where(~has_private, "Z′")  
        chosen_model = chosen_model.where(~has_nonmfg, "Z″")

        z_chosen = pd.Series(np.nan, index=df.index, dtype="float64")

        # Use pandas conditional assignment instead of np.where
        z_chosen = z_chosen.where(chosen_model != "Z", z_public)
        z_chosen = z_chosen.where(chosen_model != "Z′", z_private)
        z_chosen = z_chosen.where(chosen_model != "Z″", z_nonmfg)

        # Band classification
        band = pd.Series(None, index=df.index, dtype="object")
        # Z
        zmask = (chosen_model == "Z")
        band = band.where(~(zmask & (z_chosen < 1.81)), "Distress")
        band = band.where(~(zmask & (z_chosen >= 2.99)), "Safe")
        band = band.where(~(zmask & band.isna()), "Grey")
        # Z′
        zpmask = (chosen_model == "Z′")
        band = band.where(~(zpmask & (z_chosen < 1.23)), "Distress")
        band = band.where(~(zpmask & (z_chosen > 2.90)), "Safe")
        band = band.where(~(zpmask & band.isna()), "Grey")
        # Z″
        z2mask = (chosen_model == "Z″")
        band = band.where(~(z2mask & (z_chosen < 1.10)), "Distress")
        band = band.where(~(z2mask & (z_chosen > 2.60)), "Safe")
        band = band.where(~(z2mask & band.isna()), "Grey")

        out = pd.DataFrame({
            "ticker": df["TICKERSYMBOL"],
            "model": chosen_model,
            "z_score": z_chosen,
            "band": band,
            "wc_ta": wc_ta, "re_ta": re_ta, "ebit_ta": ebit_ta,
            "mve_tl": mve_tl, "sales_ta": sales_ta, "bve_tl": bve_tl
        })
        return out.to_json(orient="records")

    def metric_extras(self, *args, **kwargs) -> str:
        """
        Interest Coverage = EBIT_TTM / |Interest Expense| (no TTM adj to interest)
        D/E = Total Liabilities / Equity
        Also include size snapshot.
        """
        df = self.df.copy()
        # Interest coverage
        ebit_ttm = pd.to_numeric(df.get("EBIT_TTM"), errors="coerce")
        int_exp = pd.to_numeric(df.get("Interest Expense"), errors="coerce").abs()
        ic = self._safe_div(ebit_ttm, int_exp)

        tl = self._col(df, ["Total Liabilities - (Standard / Utility Template)", "Total Liabilities"])
        te = self._col(df, ["Total Shareholders Equity (As Reported)"])
        de = self._safe_div(tl, te)

        # 개선된 가격 데이터 병합
        px_data = self._pxsnap[["TICKERSYMBOL", "PX", "MKTCAP_SNP"]].rename(
            columns={"PX": "price", "MKTCAP_SNP": "mktcap"}
        )

        out = pd.DataFrame({
            "ticker": df["TICKERSYMBOL"],
            "interest_coverage": ic,
            "debt_to_equity": de,
        }).merge(px_data, left_on="ticker", right_on="TICKERSYMBOL", how="left").drop("TICKERSYMBOL", axis=1)

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
                    header_seen = True; in_table = True; continue
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
                    return pd.DataFrame(data, columns=["Ticker","Score","Weight (%)","Reason"])
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

        json_filename = f"{self.results_dir}/altman_analysis_{start_date}_{end_date}.json"
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2, default=str)

        try:
            portfolio_csv = self.extract_portfolio_table(result["output"])
            if portfolio_csv is not None:
                csv_filename = f"{self.results_dir}/altman_portfolio_{start_date}_{end_date}.csv"
                portfolio_csv.to_csv(csv_filename, index=False, encoding="utf-8")
                print(f"포트폴리오 결과 저장됨: {csv_filename}")
        except Exception as e:
            print(f"포트폴리오 CSV 저장 실패: {e}")

        log_filename = f"{self.results_dir}/altman_execution_log_{start_date}_{end_date}.txt"
        with open(log_filename, "w", encoding="utf-8") as f:
            f.write("Altman Z-Score Investment Analysis Execution Log\n")
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
        실행: end_date 기준 분기에서 Altman Z 계열 점수를 산출하고 포트폴리오를 구성.
        """
        t0 = datetime.now()

        # Keep original
        self._original_df = self.df.copy()

        # Normalize quarter and filter
        q_end = self._to_quarter(end_date)
        self._q_end = q_end
        print(f"[Altman] start_date: {start_date}, end_date: {end_date}, quarter: {q_end}")

        if "QUARTER" in self.df.columns:
            self.df = self.df[self.df["QUARTER"] == q_end].copy()

        # Price snapshot and TTM attach
        self._pxsnap = self._build_price_snapshot(q_end)
        self._attach_ttm(q_end)

        try:
            request = f"""
Analyze these stocks using Altman's Z-Score family:
- Period start: {start_date}
- Period end: {q_end}
- Available tickers: {len(self.df)}
"""
            result = self.agent.invoke({"input": request})

            # Log steps briefly
            steps = result.get("intermediate_steps", [])
            for action, observation in steps:
                print(f"Action: {getattr(action, 'tool', action)}")
                obs_str = str(observation)
                print(f"Observation: {obs_str[:500]}{'...' if len(obs_str)>500 else ''}")
                print("-"*100)

            # Save
            t1 = datetime.now()
            if self.save_results:
                self.save_analysis_results(start_date, end_date, result, (t1 - t0).total_seconds())

            return result["output"]

        finally:
            # Restore
            self.df = self._original_df
            if hasattr(self, "_pxsnap"):
                delattr(self, "_pxsnap")


# 예시 실행
if __name__ == "__main__":
    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    agent = AltmanInvestmentAnalyzer(llm=llm)

    # 예: 한 분기 실행
    start_date = "2024-01-01"
    end_date = "2024-12-31"  # 또는 "2024Q4"
    output = agent.analyze(start_date, end_date)
    print(output)
