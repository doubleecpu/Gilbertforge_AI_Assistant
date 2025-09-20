import os
import re
from typing import Dict, Any, Tuple, List
import pandas as pd

class Classifier:
    def __init__(self, app):
        self.app = app

    def classify_intent(self) -> Tuple[str, Dict[str,Any]]:
        """Return (intent, filters) where intent in {"time","product","region","customer","overall"}"""
        q = self.app.app_view.question.lower()
        filters: Dict[str,Any] = {}
        # month detection (YYYY-MM or month names)
        for m in self.app.app_view.months:
            if m.lower() in q:
                self.app.app_view.filters["month"] = m
                break
        # quarter detection
        m_quarter = re.search(r"\bq([1-4])\b", q)
        if m_quarter:
            self.app.app_view.filters["quarter"] = "Q"+m_quarter.group(1)

        # product match
        for p in self.app.app_view.products:
            if p and p.lower() in q:
                self.app.app_view.filters["product"] = p
                break
        # region match
        for r in self.app.app_view.regions:
            if r and r.lower() in q:
                self.app.app_view.filters["region"] = r
                break

        # intent heuristics
        if any(w in q for w in ["trend","time","month","quarter","year","seasonal","forecast"]):
            self.app.app_view.intent = "time"
        elif "product" in q or "sku" in q or "category" in q or "top" in q or "bottom" in q or "best" in q:
            self.app.app_view.intent = "product"
        elif "region" in q or "market" in q or "zone" in q or "area" in q:
            self.app.app_view.intent = "region"
        elif any(w in q for w in ["customer","age","gender","satisfaction","nps"]):
            self.app.app_view.intent = "customer"
        else:
            self.app.app_view.intent = "overall"
        

class MetricsComputer:
    def __init__(self, app):
        self.app = app
        self.pd = pd
        self.Path = app.Path
        self.df = None
        self.Dict = Dict
        self.Tuple = Tuple
        self.Any = Any
        self.List = List

    def load_sales(self) -> pd.DataFrame:
        """Load sales CSV and enforce dtypes. Expects columns:
        Date, Product, Region, Sales, Customer_Age, Customer_Gender, Customer_Satisfaction
        """
        csv_path = self.app.app_view.csv_path
        self.df = self.pd.read_csv(csv_path)
        # Basic typing
        if "Date" in self.df.columns:
            self.df["Date"] = self.pd.to_datetime(self.df["Date"], errors="coerce")
        # numeric conversions
        for col in ["Sales", "Customer_Age", "Customer_Satisfaction"]:
            if col in self.df.columns:
                self.df[col] = self.pd.to_numeric(self.df[col], errors="coerce")
        # clean strings
        for col in ["Product","Region","Customer_Gender"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip()
        # enrich
        if "Date" in self.df.columns:
            self.df["Year"] = self.df["Date"].dt.year
            self.df["Month"] = self.df["Date"].dt.to_period("M").astype(str)
            self.df["Quarter"] = "Q" + self.df["Date"].dt.quarter.astype(str)
        return self.df.dropna(subset=["Date","Sales"])

    def unique_dimensions(self, data_frame: pd.DataFrame) -> Tuple[list, list]:
        """Return unique products and regions for fast matching."""
        products = self.List.sort(data_frame["Product"].dropna().astype(str).unique().tolist()) if "Product" in data_frame else []
        regions = self.List.sort(data_frame["Region"].dropna().astype(str).unique().tolist()) if "Region" in data_frame else []
        return products, regions
        
    def _month_key(self) -> pd.Series:
            return self.df["Date"].dt.to_period("M").dt.to_timestamp()

    def compute_months(self, data_frame: pd.DataFrame):
        return sorted(data_frame["Month"].unique().tolist())


    def compute_time_metrics(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Monthly sales with MoM/YoY growth and moving averages."""
        m = data_frame.groupby(self._month_key())["Sales"].sum().reset_index().rename(columns={"Date":"MonthStart","Sales":"Sales"})
        m["Month"] = m["MonthStart"].dt.to_period("M").astype(str)
        m["Sales_MoM_%"] = m["Sales"].pct_change().mul(100).round(2)
        m["Sales_YoY_%"] = m["Sales"].pct_change(12).mul(100).round(2)
        m["MA_3"] = m["Sales"].rolling(3).mean().round(2)
        m["MA_6"] = m["Sales"].rolling(6).mean().round(2)
        return m

    def compute_product_metrics(self, data_frame: pd.DataFrame, period: str | None = None) -> pd.DataFrame:
        """Total sales by product (optionally within a given Month YYYY-MM)."""
        base = data_frame if period is None else data_frame[data_frame["Month"]==period]
        p = base.groupby("Product")["Sales"].sum().reset_index().sort_values("Sales", ascending=False)
        total = p["Sales"].sum()
        if total > 0:
            p["Contribution_%"] = (p["Sales"]/total*100).round(2)
        else:
            p["Contribution_%"] = 0.0
        return p

    def compute_region_metrics(self, data_frame: pd.DataFrame, period: str | None = None) -> pd.DataFrame:
        """Total sales by region (optionally within a given Month YYYY-MM)."""
        base = data_frame if period is None else data_frame[data_frame["Month"]==period]
        r = base.groupby("Region")["Sales"].sum().reset_index().sort_values("Sales", ascending=False)
        total = r["Sales"].sum()
        if total > 0:
            r["Share_%"] = (r["Sales"]/total*100).round(2)
        else:
            r["Share_%"] = 0.0
        return r

    def compute_customer_metrics(self,data_frame: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Return dict of customer segment views: age bins, gender, satisfaction dist."""
        out: Dict[str, pd.DataFrame] = {}
        if "Customer_Age" in data_frame:
            bins = [0,24,34,44,54,200]
            labels = ["<25","25-34","35-44","45-54","55+"]
            age = data_frame.copy()
            age["Age_Bin"] = pd.cut(age["Customer_Age"], bins=bins, labels=labels, right=True, include_lowest=True)
            out["age"] = age.groupby("Age_Bin").size().reset_index(name="Count")
        if "Customer_Gender" in data_frame:
            g = data_frame.groupby("Customer_Gender").size().reset_index(name="Count")
            out["gender"] = g
        if "Customer_Satisfaction" in data_frame:
            sat = data_frame.copy()
            sat["Satisfaction_Bin"] = pd.cut(sat["Customer_Satisfaction"], bins=[0,2,3,4,5], labels=["Low(<=2)","Med(>2-3]","High(>3-4]","Top(>4-5]"], include_lowest=True)
            out["satisfaction"] = sat.groupby("Satisfaction_Bin").size().reset_index(name="Count")
            out["satisfaction_mean"] = pd.DataFrame({"Mean_Satisfaction":[round(data_frame["Customer_Satisfaction"].mean(),2)]})
        return out

    def get_kpi_tiles(self, data_frame: pd.DataFrame, monthly: pd.DataFrame) -> Dict[str, float]:
        total_sales = float(data_frame["Sales"].sum())
        unique_products = data_frame["Product"].nunique() if "Product" in data_frame else 0
        unique_regions = data_frame["Region"].nunique() if "Region" in data_frame else 0
        avg_sat = round(float(data_frame["Customer_Satisfaction"].mean()),2) if "Customer_Satisfaction" in data_frame else None
        mom = float(monthly["Sales_MoM_%"].iloc[-1]) if len(monthly) >= 2 and pd.notnull(monthly["Sales_MoM_%"].iloc[-1]) else 0.0
        yoy = float(monthly["Sales_YoY_%"].iloc[-1]) if len(monthly) >= 13 and pd.notnull(monthly["Sales_YoY_%"].iloc[-1]) else 0.0
        return {
            "Total Sales": round(total_sales,2),
            "Products": int(unique_products),
            "Regions": int(unique_regions),
            "Avg Satisfaction": avg_sat if avg_sat is not None else 0.0,
            "MoM %": round(mom,2),
            "YoY %": round(yoy,2),
        }
    def get_DataFrame(self, rows):
        return self.pd.DataFrame(rows)


    def build_fact_store(self, data_frame: pd.DataFrame) -> List[Dict[str, Any]]:
        """Build a compact list of fact dicts used by the retriever + LLM.
        Types: time_month, product_total, product_month, region_total, region_month, customer_segment
        """
        facts: List[Dict[str,Any]] = []
        # Time facts (by month)
        tm = self.compute_time_metrics(data_frame)
        for _, r in tm.iterrows():
            facts.append({
                "type": "time_month",
                "month": r["Month"],
                "sales": float(r["Sales"]),
                "mom_pct": None if pd.isna(r["Sales_MoM_%"]) else float(r["Sales_MoM_%"]),
                "yoy_pct": None if pd.isna(r["Sales_YoY_%"]) else float(r["Sales_YoY_%"]),
                "ma_3": None if pd.isna(r["MA_3"]) else float(r["MA_3"]),
                "ma_6": None if pd.isna(r["MA_6"]) else float(r["MA_6"]),
            })

        # Product totals and month splits
        p_all = self.compute_product_metrics(data_frame)
        for _, r in p_all.iterrows():
            facts.append({
                "type":"product_total",
                "product": str(r["Product"]),
                "sales": float(r["Sales"]),
                "contribution_pct": float(r["Contribution_%"])
            })
        for m in (data_frame["Month"].unique().tolist()):
            p_m = self.compute_product_metrics(data_frame, period=m)
            for _, r in p_m.iterrows():
                facts.append({
                    "type":"product_month",
                    "month": m,
                    "product": str(r["Product"]),
                    "sales": float(r["Sales"]),
                    "contribution_pct": float(r["Contribution_%"])
                })
        # Region totals and month splits
        r_all = self.compute_region_metrics(data_frame)
        for _, r in r_all.iterrows():
            facts.append({
                "type":"region_total",
                "region": str(r["Region"]),
                "sales": float(r["Sales"]),
                "share_pct": float(r["Share_%"])
            })
        for m in sorted(data_frame["Month"].unique().tolist()):
            r_m = self.compute_region_metrics(data_frame, period=m)
            for _, r in r_m.iterrows():
                facts.append({
                    "type":"region_month",
                    "month": m,
                    "region": str(r["Region"]),
                    "sales": float(r["Sales"]),
                    "share_pct": float(r["Share_%"])
                })
        # Customer segments
        seg = self.compute_customer_metrics(data_frame)
        if "age" in seg:
            for _, r in seg["age"].iterrows():
                facts.append({
                    "type":"customer_segment",
                    "segment":"age",
                    "age_bin": str(r["Age_Bin"]),
                    "count": int(r["Count"])
                })
        if "gender" in seg:
            for _, r in seg["gender"].iterrows():
                facts.append({
                    "type":"customer_segment",
                    "segment":"gender",
                    "gender": str(r["Customer_Gender"]),
                    "count": int(r["Count"])
                })
        if "satisfaction" in seg:
            for _, r in seg["satisfaction"].iterrows():
                facts.append({
                    "type":"customer_segment",
                    "segment":"satisfaction",
                    "satisfaction_bin": str(r["Satisfaction_Bin"]),
                    "count": int(r["Count"])
                })
        return facts

    def filter_facts(self, facts: List[Dict[str,Any]], month: str | None = None,
                    product: str | None = None, region: str | None = None,
                    segment: str | None = None) -> List[Dict[str,Any]]:
        out = []
        for f in facts:
            if month and f.get("month") != month:
                continue
            if product and f.get("product") != product:
                continue
            if region and f.get("region") != region:
                continue
            if segment and f.get("segment") != segment:
                continue
            out.append(f)
        return out[:50]  # keep it compact



class Evaluator:
    def __init__(self, app):
        self.app = app

    def load_eval_set(self, filepath: str) -> List[Dict[str, Any]]:
        import json
        items = []
        with open(filepath, "r") as f:
            for line in f:
                items.append(json.loads(line))
        return items
    
    def simple_eval(self, pred: str, gold: str) -> float:
        """Extremely simple heuristic: percentage of gold tokens found in pred."""
        g_tokens = [t.lower() for t in gold.split() if t.isalnum()]
        if not g_tokens:
            return 0.0
        hit = sum(1 for t in set(g_tokens) if t in pred.lower())
        return round(100.0 * hit / max(1, len(set(g_tokens))), 2)
    
class Agent:
    def __init__(self, app):
        self.app = app
        self.question = None
        self.facts_block = None
        self.retrieved_facts: List[Dict[str,Any]] = []
        self.SYSTEM_PROMPT = """You are InsightForge, a precise BI copilot.
        Only answer using the provided facts. Do not invent numbers.
        Explain briefly (3-6 sentences) and conclude with a 1-line action insight.
        Always list the facts you used under 'Facts used:' with bullet points.
        """
        self.USER_TEMPLATE = """Question: {self.question}

        Relevant facts:
        {self.facts_block}

        Now answer the question succinctly. If information is missing, say what is missing and suggest the closest alternative insight.
        """

    def run_agent(self, question: str, retrieved_facts: List[Dict[str,Any]]) -> Tuple[str, List[Dict[str,Any]]]:
        """Wrapper to keep an agent-like interface. Returns (answer, used_facts)."""
        self.question = question
        self.retrieved_facts = retrieved_facts
        self.answer_question()
    
class RAG_Chain:
    def __init__(self, app):
        self.app = app
        self.lines = []
        self.llm = None
        self.provider = None
        self.f = None
        

    def _format_facts(self):
        for self.f in self.facts[:20]:
            if self.f["type"] == "time_month":
                self.lines.append(f"- [{self.f['type']}] {self.f['month']}: sales={self.f['sales']}, MoM%={self.f['mom_pct']}, YoY%={self.f['yoy_pct']}")
            elif self.f["type"] in ("product_total","product_month"):
                tag = f"{self.f['type'].replace('_','/') }"
                m = self.f.get("month","ALL")
                self.lines.append(f"- [{tag}] month={m}, product={self.f['product']}, sales={self.f['sales']}, contrib%={self.f.get('contribution_pct')}")
            elif self.f["type"] in ("region_total","region_month"):
                tag = f"{self.f['type'].replace('_','/') }"
                m = self.f.get("month","ALL")
                self.lines.append(f"- [{tag}] month={m}, region={self.f['region']}, sales={self.f['sales']}, share%={self.f.get('share_pct')}")
            elif self.f["type"] == "customer_segment":
                seg = self.f.get("segment")
                if seg=="age":
                    self.lines.append(f"- [customer/age] bin={self.f['age_bin']}, count={self.f['count']}")
                elif seg=="gender":
                    self.lines.append(f"- [customer/gender] gender={self.f['gender']}, count={self.f['count']}")
                else:
                    self.lines.append(f"- [customer/satisfaction] bin={self.f['satisfaction_bin']}, count={self.f['count']}")
        return "\\n".join(self.lines) if self.lines else "(no facts)"

    def init_llm(self):
        """Initialize an LLM if available. Falls back to a rule-based responder if no API keys."""
        self.provider = None
        self.llm = None
        try:
            from langchain_openai import ChatOpenAI
            if os.environ.get("OPENAI_API_KEY"):
                self.llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL","gpt-4o-mini"), temperature=0.1)
                self.provider = "openai"
        except Exception:
            pass
        if self.llm is None:
            try:
                from langchain_groq import ChatGroq
                if os.environ.get("GROQ_API_KEY"):
                    self.llm = ChatGroq(model=os.environ.get("GROQ_MODEL","llama-3.1-70b-versatile"), temperature=0.1)
                    self.provider = "groq"
            except Exception:
                pass

    def _llm_answer(self, question: str, facts: List[Dict[str,Any]]) -> str:
        from langchain_core.messages import SystemMessage, HumanMessage
        self.facts_block = self._format_facts()
        prompt = self.USER_TEMPLATE.format(question=question, facts_block=self.facts_block)
        msgs = [SystemMessage(content=self.SYSTEM_PROMPT), HumanMessage(content=prompt)]
        out = self.llm.invoke(msgs)
        return out.content if hasattr(out, "content") else str(out)

    def _fallback_answer(self):
        self.facts_block = self._format_facts(self.facts)
        return f"""[Fallback, no LLM configured]

Question: {self.question}

Answer (from facts):
- Key observations:
- {(' | '.join([self.f.get('month', self.f.get('product', self.f.get('region','segment'))) for self.f in self.facts[:5]]) )}

Facts used:
{self.facts_block}
"""

    def answer_question(self):
        self.init_llm()
        if self.llm is not None:
            try:
                ans = self._llm_answer(self.llm, self.question, self.facts)
                return ans, self.facts
            except Exception as e:
                return self._fallback_answer(self.question, self.facts) + f"\\n(Note: LLM error: {e})", self.facts
        else:
            return self._fallback_answer()
