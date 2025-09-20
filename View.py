#from __future__ import annotations

import os
from turtle import st
from anyio import Path
from matplotlib import pyplot

class App_View:
    def __init__(self, app):
        self.app = app
        self.st = app.streamlit
        self.upload_dir = app.upload_dir
        self.key_input = None
        self.metrics = app.metrics_computer
        self.eval = app.evaluator
        self.plt = pyplot
        self.csv_path = None
        self.df = None
        self.months = None
        self.time_metrics = None
        self.prod_all = None
        self.reg_all = None
        self.cust_views = None
        self.facts = None
        self.products = None
        self.regions = None
        self.intent = None
        self.filters = {}
        self.question = None
        self.model_name = None
        self.Processed_Texts_Path = None

    def display_app(self, Title):
        self.st.set_page_config(page_title=Title, page_icon="📊", layout="wide")
        self.st.title("GilbertForge — AI-Powered BI Assistant")
        # --- Provider & API key controls in sidebar ---
        self.display_key_status("LLM Settings")
        self.display_settings("Application Settings")
        self.display_PDF_Analyzer("PDF Analyzer")
        page = self.st.sidebar.radio("Navigate", ["💬 Chat", "📈 Overview", "⏱ Time", "📦 Products", "🗺 Regions", "👥 Customers", "🧪 Evaluate"])
        if page == "💬 Chat":
            self.display_chat_interface()
        elif page == "📈 Overview":
            self.display_overview()
        elif page == "⏱ Time":
            self.display_time_analysis()
        elif page == "📦 Products":
            self.display_product_analysis()
        elif page == "🗺 Regions":
            self.display_region_analysis()
        elif page == "👥 Customers":
            self.display_customer_analysis()
        elif page == "🧪 Evaluate":
            self.display_evaluation()


    def display_key_status(self, message):
        with self.st.sidebar.expander(f"🔐 {message}", expanded=True):
            provider = self.st.selectbox("Provider", ["None (fallback)", "OpenAI", "Groq"], index=1 if os.environ.get("OPENAI_API_KEY") else (2 if os.environ.get("GROQ_API_KEY") else 0))
            self.key_input = self.st.text_input("API Key", type="password", help="Paste your API key. It will be set for this session only.")
            # Txt key file uploader
            key_file = self.st.file_uploader("...or upload key (.txt)", type=["txt"], help="File should contain only the API key string.")
            try:
                self.key_input = key_file.read().decode("utf-8").strip()
            except Exception:
                self.st.warning("Could not read uploaded key file.")

            openai_models = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"]
            groq_models = ["llama-3.1-70b-versatile", "llama-3.1-8b-instant"]
            if provider == "OpenAI":    
                self.model_name = self.st.selectbox("Model", openai_models, index=0)
            elif provider == "Groq":
                self.model_name = self.st.selectbox("Model", groq_models, index=0)
            else:
                self.model_name = self.st.selectbox("Model", ["(no LLM)"], index=0, disabled=True)

            if self.st.button("Apply LLM Settings"):
                if provider == "OpenAI":
                    if self.key_input:
                        os.environ["OPENAI_API_KEY"] = self.key_input
                    os.environ.pop("GROQ_API_KEY", None)
                    os.environ["OPENAI_MODEL"] = self.model_name
                    self.st.success(f"OpenAI configured with model: {self.model_name}")
                elif provider == "Groq":
                    if self.key_input:
                        os.environ["GROQ_API_KEY"] = self.key_input
                    os.environ.pop("OPENAI_API_KEY", None)
                    os.environ["GROQ_MODEL"] = self.model_name
                    self.st.success(f"Groq configured with model: {self.model_name}")
                else:
                    # clear
                    os.environ.pop("OPENAI_API_KEY", None)
                    os.environ.pop("GROQ_API_KEY", None)
                    self.st.info("Using fallback responder (no LLM).")

        self.st.sidebar.markdown("### Model Provider")
        self.st.sidebar.write("Set environment variables for OpenAI (`OPENAI_API_KEY`) or Groq (`GROQ_API_KEY`).")
        self.st.sidebar.write("Models default to `gpt-4o-mini` (OpenAI) / `llama-3.1-70b-versatile` (Groq).")
   
        
    def _load(self):
        self.st.cache_data(show_spinner=False)
        self.df = self.metrics.load_sales()
        self.months = self.metrics.compute_months(self.df)
        self.time_metrics = self.metrics.compute_time_metrics(self.df)
        self.products_metrics_all = self.metrics.compute_product_metrics(self.df)
        self.regions_metrics_all = self.metrics.compute_region_metrics(self.df)
        self.customers_views = self.metrics.compute_customer_metrics(self.df)
        self.facts = self.metrics.build_fact_store(self.df)
        self.products, regions = self.metrics.unique_dimensions(self.df)
        
    def display_settings(self, message):
        self.st.sidebar.title("⚙️ {message}")
        default_csv = self.app.Path(__file__).parent / "assets" / "sales_data.csv"
        uploaded = self.st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
        self.csv_path = default_csv
        if uploaded:
            self.upload_dir = self.app.Path(__file__).parent / "assets"
            self.upload_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = self.upload_dir / "uploaded_sales.csv"
            with open(tmp_path, "wb") as f:
                f.write(uploaded.read())
            self.csv_path = tmp_path
            self._load()

    def display_PDF_Analyzer(self, message):
        self.st.sidebar.title("PDF Analysis")
        self.upload_pdf_dir = self.app.Path(__file__).parent / "assets" / "PDF_Folder"
        default_PDF = self.upload_pdf_dir / "Default.pdf"
        self.upload_pdf_dir.mkdir(parents=True, exist_ok=True)
        self.PDF_path = default_PDF
        self.filename = None
        self.Processed_Texts_Path = None

        uploaded = self.st.sidebar.file_uploader("Upload PDF (optional)", type=["pdf"])
        if uploaded:

            self.PDF_path = self.upload_pdf_dir / uploaded.name
            self.filename = self.app.Path(self.PDF_path)
            with open(self.PDF_path, "wb") as f:
                f.write(uploaded.read())
            self.st.sidebar.info(f"Saved as: {self.filename}")
        else:
            # If no upload, check if default exists
            if default_PDF.exists():
                self.filename = self.app.Path(default_PDF)
                self.PDF_path = default_PDF
            else:
                self.filename = None
                self.PDF_path = None
        #PDF Reading and Processing
        if self.st.sidebar.button("Process PDFs"):
            if self.upload_pdf_dir.exists():
                self.st.sidebar.info(f"Processing files: {self.upload_pdf_dir}")
                try:
                    self.Processed_Texts_Path = self.app.model.load_pdfs(self.upload_pdf_dir)
                    if not self.Processed_Texts_Path:
                        self.st.sidebar.warning("No PDFs found or loaded. Please check the file and try again.")
                        self.Processed_Texts_Path = None
                    else:
                        self.st.sidebar.success("PDFs processed. You can now ask questions in the chat.")
                        
                        #st.sidebar.info(f"Processed texts saved to Path:{Processed_Texts_Path}.pkl")
                except Exception as e:
                    self.st.sidebar.error(f"Failed to process PDF: {e}")
            else:
                self.st.sidebar.warning("No PDF selected or file not found. Please upload a PDF first.")

        if self.st.sidebar.button("Analyze PDFs (example)"):
            if self.PDF_path and self.PDF_path.exists():
                example_query = "Summarize the key points from the documents."
                answer = self.app.model.analyze_pdfs(example_query)
                self.st.sidebar.markdown("**Example Analysis Result:**")
                self.st.sidebar.write(answer)
            else:
                self.st.sidebar.warning("No PDF loaded or processed. Please upload and process a PDF first.")

        confirm_clear = self.st.sidebar.checkbox("Confirm clear processed PDF data")
        if self.st.sidebar.button("Clear PDF Data", disabled=not confirm_clear):
            self.Processed_Texts_Path = self.upload_pdf_dir / "processed_texts.pkl"
            if self.Processed_Texts_Path.exists():
                self.Processed_Texts_Path.unlink()
                self.st.sidebar.success("Cleared processed PDF data.")
            else:
                self.st.sidebar.info("No processed PDF data found to clear.")


    def display_chat_interface(self):
        self.st.subheader("Chat with InsightForge")
        q = self.st.text_input("Ask a question (e.g., 'Which region led last month?' or 'Trend in 2024?')")
        if self.st.button("Ask") and q.strip():
            # classify + retrieve
            self.app.classifier.classify_intent()
            # heuristic: choose facts
            retrieved = []
            if self.intent == "time":
                # take recent 12 months time facts
                time_facts = [f for f in self.facts if f["type"]=="time_month"]
                retrieved = time_facts[-12:]
            elif self.intent == "product":
                if "product" in self.filters:
                    retrieved = [f for f in self.facts if f["type"] in ("product_total","product_month") and f.get("product")==self.filters["product"]]
                else:
                    retrieved = [f for f in self.facts if f["type"]=="product_total"][:20]
            elif self.intent == "region":
                if "region" in self.filters:
                    retrieved = [f for f in self.facts if f["type"] in ("region_total","region_month") and f.get("region")==self.filters["region"]]
                else:
                    retrieved = [f for f in self.facts if f["type"]=="region_total"][:20]
            elif self.intent == "customer":
                retrieved = [f for f in self.facts if f["type"]=="customer_segment"][:30]
            else:
                # overall blend
                retrieved = [f for f in self.facts if f["type"] in ("time_month","product_total","region_total")]

            ans, used = self.app.agent.run_agent(q, retrieved)
            self.st.markdown(ans)
            with self.st.expander("Facts used"):
                self.st.json(used)


    def plot_monthly_sales(self):
        fig, ax = self.plt.subplots()
        ax.plot(self.time_metrics["MonthStart"], self.time_metrics["Sales"])
        ax.set_title("Monthly Sales")
        ax.set_xlabel("Month")
        ax.set_ylabel("Sales")
        self.st.pyplot(fig)

    def plot_top_products(self, n=10, period=None):
        data = self.metrics.compute_product_metrics(self.df, period)
        data = data.head(n)
        fig, ax = self.plt.subplots()
        ax.bar(data["Product"], data["Sales"])
        ax.set_title(f"Top Products{' ('+period+')' if period else ''}")
        ax.set_xlabel("Product")
        ax.set_ylabel("Sales")
        self.st.pyplot(fig)

    def plot_regions(self, period=None):
        data = self.metrics.compute_region_metrics(self.df, period)
        fig, ax = self.plt.subplots()
        ax.bar(data["Region"], data["Sales"])
        ax.set_title(f"Regions — Sales{' ('+period+')' if period else ''}")
        ax.set_xlabel("Region")
        ax.set_ylabel("Sales")
        self.st.pyplot(fig)

    def plot_satisfaction(self):
        if "satisfaction" in self.cust_views:
            data = self.cust_views["satisfaction"]
            fig, ax = self.plt.subplots()
            ax.bar(data["Satisfaction_Bin"].astype(str), data["Count"])
            ax.set_title("Customer Satisfaction Distribution")
            ax.set_xlabel("Bucket")
            ax.set_ylabel("Count")
            self.st.pyplot(fig)


    def display_overview(self):
        self.st.subheader("Executive Overview")
        kpis = self.metrics.get_kpi_tiles(self.df, self.time_metrics)
        c1, c2, c3, c4, c5, c6 = self.st.columns(6)
        c1.metric("Total Sales", f"{kpis['Total Sales']:,}")
        c2.metric("Products", kpis["Products"])
        c3.metric("Regions", kpis["Regions"])
        c4.metric("Avg Satisfaction", kpis["Avg Satisfaction"])
        c5.metric("MoM %", kpis["MoM %"])
        c6.metric("YoY %", kpis["YoY %"])

        self.plot_monthly_sales()
        self.st.markdown("---")
        c7, c8 = self.st.columns(2)
        with c7:
            self.st.markdown("**Top Products (All-time)**")
            self.plot_top_products(n=10, period=None)
        with c8:
            self.st.markdown("**Regions (All-time)**")
            self.plot_regions(period=None)
        self.st.markdown("---")
        self.plot_satisfaction()

    def display_time_analysis(self):
        self.st.subheader("Time Analysis")
        self.st.dataframe(self.time_metrics)
        self.plot_monthly_sales()

    def display_product_analysis(self):
        self.st.subheader("Product Analysis")
        period = self.st.selectbox("Month filter (optional)", ["(All)"] + self.months)
        p = None if period == "(All)" else period
        self.table = self.metrics.compute_product_metrics(self.df, p)
        self.st.dataframe(self.table)
        self.plot_top_products(n=15, period=p)

    def display_region_analysis(self):
        self.st.subheader("Region Analysis")
        period = self.st.selectbox("Month filter (optional)", ["(All)"] + self.months, key="regions_period")
        p = None if period == "(All)" else period
        self.table = self.metrics.compute_region_metrics(self.df, p)
        self.st.dataframe(self.table)
        self.plot_regions(period=p)

    def display_customer_analysis(self):
        self.st.subheader("Customer Segments")
        if "age" in self.cust_views:
            self.st.markdown("**Age Segments**")
            self.st.dataframe(self.cust_views["age"])
        if "gender" in self.cust_views:
            self.st.markdown("**Gender Split**")
            self.st.dataframe(self.cust_views["gender"])
        if "satisfaction" in self.cust_views:
            self.st.markdown("**Satisfaction Buckets**")
            self.st.dataframe(self.cust_views["satisfaction"])
            self.plot_satisfaction()

    def display_evaluation(self):
        self.st.subheader("Model Evaluation (Quick Heuristic)")
        eval_file = self.app.Path(__file__).parent / "assets" / "qa.jsonl"
        if eval_file.exists():
            items = self.eval.load_eval_set(str(eval_file))
            rows = []
            for item in items:
                q = item["question"]
                gold = item["answer"]
                # simple retrieval: blend
                retrieved = [f for f in self.facts if f["type"] in ("time_month","product_total","region_total")]            
                pred, _ = self.app.agent.run_agent(q, retrieved)
                score = self.app.agent.simple_eval(pred, gold)
                rows.append({"question": q, "gold": gold, "pred": pred, "score": score})
            self.st.dataframe(self.metrics.get_DataFrame(rows))
        else:
            self.st.info("No qa.jsonl found in assets/. Add some eval questions to run evaluation.")

