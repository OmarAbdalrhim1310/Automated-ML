import streamlit as st
import pandas as pd
import numpy as np
import json

# Direct Groq SDK
try:
    from groq import Groq
    GROQ_AVAILABLE = True
    GROQ_IMPORT_ERROR = None
except Exception as e:
    Groq = None
    GROQ_AVAILABLE = False
    GROQ_IMPORT_ERROR = str(e)

# Optional Plotly
try:
    import plotly.express as px
except Exception:
    px = None

# Project modules
from preprocessing import run_advanced_preprocessing
from automl_engine import (
    detect_data_type,
    analyze_class_balance,
    evaluate_autonomous_agent_pipeline,
)

# ============================================================
# DATA SCIENCE AGENT
# ============================================================

# ============================================================
# DETERMINISTIC AUTOML POLICY
# The LLM may explain results, but it does not choose training settings.
# Same dataset + target => same AutoML policy and reproducible result.
# ============================================================
DETERMINISTIC_AUTOML_POLICY = {
    "training_mode": "Balanced",
    "feature_engineering": False,
    "outlier_method": "No Treatment",
    "feature_selection": False,
    "shap_explainability": False,
    "enable_ensemble": False,
}


class DataScienceAgent:
    """
    Groq-powered Data Science Agent with LOCAL TOOL CALLING.

    Groq decides which analysis tool(s) it needs. The Python application
    executes those tools locally, sends the results back to Groq, and
    obtains a final grounded Data Science report.
    """

    def __init__(self, api_key, model="openai/gpt-oss-120b"):
        if not GROQ_AVAILABLE:
            raise RuntimeError(
                "Groq package is not installed. Run: python -m pip install groq"
            )
        if not api_key:
            raise ValueError("Groq API key is required.")

        self.client = Groq(api_key=api_key)
        self.model = model
        self.df = None
        self.target = None
        self.last_automl_result = None

    # --------------------------------------------------------
    # Local Python tools
    # --------------------------------------------------------

    @staticmethod
    def _json_safe(value):
        if isinstance(value, pd.DataFrame):
            return value.head(20).to_dict(orient="records")
        if isinstance(value, pd.Series):
            return value.head(50).to_dict()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, dict):
            return {
                str(k): DataScienceAgent._json_safe(v)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [DataScienceAgent._json_safe(v) for v in value]
        return value

    @staticmethod
    def profile_dataset(df):
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        missing = df.isna().sum()
        missing_cols = {
            str(k): int(v)
            for k, v in missing[missing > 0].items()
        }

        unique_counts = {
            str(c): int(df[c].nunique(dropna=True))
            for c in df.columns
        }

        return {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "numeric_columns": numeric,
            "categorical_columns": categorical,
            "missing_values": int(missing.sum()),
            "missing_by_column": missing_cols,
            "duplicate_rows": int(df.duplicated().sum()),
            "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
            "unique_values_by_column": unique_counts,
            "constant_columns": [
                str(c)
                for c in df.columns
                if df[c].nunique(dropna=False) <= 1
            ],
        }

    @staticmethod
    def analyze_target(df, target):
        if not target or target == "[Unsupervised Clustering]":
            return {
                "task": "unsupervised",
                "target": None,
                "message": "No supervised target selected."
            }

        if target not in df.columns:
            return {
                "error": f"Target column '{target}' does not exist."
            }

        y = df[target]
        unique = int(y.nunique(dropna=True))

        result = {
            "target": target,
            "unique_values": unique,
            "missing_values": int(y.isna().sum()),
            "dtype": str(y.dtype),
        }

        try:
            balance = analyze_class_balance(y)
            result.update({
                "task": (
                    "classification"
                    if balance.get("is_classification")
                    else "regression"
                ),
                "balance_status": balance.get("status"),
                "majority_minority_ratio": balance.get("ratio"),
                "minority_percentage": balance.get("minority_percentage"),
                "majority_percentage": balance.get("majority_percentage"),
                "class_counts": (
                    DataScienceAgent._json_safe(balance.get("counts"))
                    if balance.get("is_classification")
                    else None
                ),
            })
        except Exception as exc:
            result["task"] = (
                "classification"
                if (y.dtype == "object" or unique <= 20)
                else "regression"
            )
            result["balance_analysis_error"] = str(exc)

        return result

    @staticmethod
    def analyze_data_quality(df):
        missing = df.isna().sum()
        missing_pct = (missing / max(len(df), 1) * 100).round(2)

        high_missing = {
            str(col): float(missing_pct[col])
            for col in df.columns
            if missing_pct[col] >= 30
        }

        possible_id_columns = []
        for col in df.columns:
            try:
                ratio = df[col].nunique(dropna=True) / max(len(df), 1)
                if ratio >= 0.98:
                    possible_id_columns.append(str(col))
            except Exception:
                pass

        return {
            "missing_values_total": int(missing.sum()),
            "columns_with_missing": {
                str(col): {
                    "count": int(missing[col]),
                    "percentage": float(missing_pct[col]),
                }
                for col in df.columns
                if missing[col] > 0
            },
            "high_missing_columns_30pct_plus": high_missing,
            "duplicate_rows": int(df.duplicated().sum()),
            "constant_columns": [
                str(c)
                for c in df.columns
                if df[c].nunique(dropna=False) <= 1
            ],
            "possible_id_columns": possible_id_columns,
        }

    def run_preprocessing_audit(self):
        if self.df is None:
            return {"error": "Dataset is not loaded."}

        try:
            _, report = run_advanced_preprocessing(self.df.copy())
            return {
                "status": "success",
                "report": self._json_safe(report or []),
                "note": (
                    "This is a data-quality audit. The AutoML engine still "
                    "uses its leakage-safe preprocessing pipeline."
                ),
            }
        except Exception as exc:
            return {
                "status": "failed",
                "error": str(exc),
            }

    def run_automl(
        self,
        imbalance_method="No Treatment",
        training_mode="Fast",
        feature_engineering=False,
        outlier_method="No Treatment",
        feature_selection=False,
        shap_explainability=False,
        enable_ensemble=True,
    ):
        if self.df is None:
            return {"error": "Dataset is not loaded."}

        target = (
            self.target
            if self.target
            else "[Unsupervised Clustering]"
        )

        # Never let the LLM choose these values. The engine owns the ML policy.
        training_mode = DETERMINISTIC_AUTOML_POLICY["training_mode"]
        feature_engineering = DETERMINISTIC_AUTOML_POLICY["feature_engineering"]
        outlier_method = DETERMINISTIC_AUTOML_POLICY["outlier_method"]
        feature_selection = DETERMINISTIC_AUTOML_POLICY["feature_selection"]
        shap_explainability = DETERMINISTIC_AUTOML_POLICY["shap_explainability"]
        enable_ensemble = DETERMINISTIC_AUTOML_POLICY["enable_ensemble"]

        # Deterministic imbalance rule: only classification can use treatment.
        if target == "[Unsupervised Clustering]":
            imbalance_method = "No Treatment"
        elif self.df[target].nunique(dropna=True) <= 20:
            balance = analyze_class_balance(self.df[target].dropna())
            imbalance_method = (
                "Class Weighting"
                if balance.get("status") in {"Moderate Imbalance", "Severe Imbalance"}
                else "No Treatment"
            )
        else:
            imbalance_method = "No Treatment"

        try:
            pipeline_result = evaluate_autonomous_agent_pipeline(
                self.df,
                target,
                imbalance_method=imbalance_method,
                training_mode=training_mode,
                feature_engineering=bool(feature_engineering),
                outlier_method=outlier_method,
                feature_selection=bool(feature_selection),
                shap_explainability=bool(shap_explainability),
                enable_ensemble=bool(enable_ensemble),
            )

            if not isinstance(pipeline_result, (tuple, list)):
                raise ValueError(
                    "AutoML pipeline did not return a tuple/list."
                )

            if len(pipeline_result) < 6:
                raise ValueError(
                    f"AutoML pipeline returned {len(pipeline_result)} values; "
                    "at least 6 are required."
                )

            (
                task,
                leaderboard,
                winner,
                score,
                decision_logs,
                tuning_explanations,
            ) = pipeline_result[:6]

            result = {
                "task": task,
                "leaderboard": leaderboard,
                "winner": winner,
                "score": score,
                "logs": decision_logs,
                "explanations": tuning_explanations,
                "target": target,
                "imbalance_method": imbalance_method,
                "training_mode": training_mode,
                "feature_engineering": bool(feature_engineering),
                "outlier_method": outlier_method,
                "feature_selection": bool(feature_selection),
                "shap_explainability": bool(shap_explainability),
                "enable_ensemble": bool(enable_ensemble),
            }

            self.last_automl_result = result

            return {
                "status": "success",
                "task": str(task),
                "winner": str(winner),
                "winner_score": self._json_safe(score),
                "target": target,
                "imbalance_method": imbalance_method,
                "training_mode": training_mode,
                "leaderboard": self._json_safe(leaderboard),
                "decision_logs": self._json_safe(decision_logs),
                "tuning_explanations": self._json_safe(tuning_explanations),
            }

        except Exception as exc:
            return {
                "status": "failed",
                "error": str(exc),
                "target": target,
                "imbalance_method": imbalance_method,
                "training_mode": training_mode,
            }

    # --------------------------------------------------------
    # Groq tool schemas
    # --------------------------------------------------------

    def tool_schemas(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "profile_dataset",
                    "description": (
                        "Inspect dataset shape, column types, missing values, "
                        "duplicates, constants, and cardinality."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_target",
                    "description": (
                        "Analyze the selected target and determine whether the "
                        "problem is classification, regression, or unsupervised; "
                        "also inspect class balance when applicable."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_data_quality",
                    "description": (
                        "Analyze missing data, duplicates, constant columns, "
                        "high-missing columns, and possible ID columns."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_preprocessing_audit",
                    "description": (
                        "Run the project's existing preprocessing audit and "
                        "return its cleaning report."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_automl",
                    "description": (
                        "Run the existing leakage-safe AutoML engine. Use this "
                        "only after inspecting the dataset and target. Choose "
                        "imbalance handling and training depth based on evidence."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "imbalance_method": {
                                "type": "string",
                                "enum": [
                                    "No Treatment",
                                    "Class Weighting",
                                    "SMOTE",
                                    "Random Oversampling",
                                    "Random Undersampling",
                                ],
                            },
                            "training_mode": {
                                "type": "string",
                                "enum": ["Fast", "Balanced", "Deep"],
                            },
                            "feature_engineering": {
                                "type": "boolean",
                            },
                            "outlier_method": {
                                "type": "string",
                                "description": (
                                    "Outlier method supported by the project's "
                                    "AutoML engine. Use No Treatment when unsure."
                                ),
                            },
                            "feature_selection": {
                                "type": "boolean",
                            },
                            "shap_explainability": {
                                "type": "boolean",
                            },
                            "enable_ensemble": {
                                "type": "boolean",
                            },
                        },
                        "required": [
                            "imbalance_method",
                            "training_mode",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
        ]

    # --------------------------------------------------------
    # Tool dispatcher
    # --------------------------------------------------------

    def execute_tool(self, name, arguments):
        arguments = arguments or {}

        if name == "profile_dataset":
            return self.profile_dataset(self.df)

        if name == "analyze_target":
            return self.analyze_target(self.df, self.target)

        if name == "analyze_data_quality":
            return self.analyze_data_quality(self.df)

        if name == "run_preprocessing_audit":
            return self.run_preprocessing_audit()

        if name == "run_automl":
            return self.run_automl(
                imbalance_method=arguments.get(
                    "imbalance_method",
                    "No Treatment",
                ),
                training_mode=arguments.get(
                    "training_mode",
                    "Fast",
                ),
                feature_engineering=arguments.get(
                    "feature_engineering",
                    False,
                ),
                outlier_method=arguments.get(
                    "outlier_method",
                    "No Treatment",
                ),
                feature_selection=arguments.get(
                    "feature_selection",
                    False,
                ),
                shap_explainability=arguments.get(
                    "shap_explainability",
                    False,
                ),
                enable_ensemble=arguments.get(
                    "enable_ensemble",
                    True,
                ),
            )

        return {
            "error": f"Unknown tool: {name}"
        }

    # --------------------------------------------------------
    # Real agent loop
    # --------------------------------------------------------

    def run_agent(
        self,
        df,
        target=None,
        user_goal=None,
        max_rounds=4,
    ):
        self.df = df.copy()
        self.target = target
        self.last_automl_result = None

        tools = self.tool_schemas()

        goal = user_goal or (
            "Analyze this dataset as an autonomous data scientist. "
            "Inspect the dataset and target, check data quality and class "
            "imbalance, decide an appropriate ML strategy, run AutoML when "
            "appropriate, inspect the returned metrics, and produce a concise "
            "final report with evidence-based next steps."
        )

        system_prompt = """
You are an autonomous Data Science Agent inside a Streamlit AutoML application.

You have LOCAL TOOLS that inspect and train on the REAL uploaded dataset.
Use tools instead of guessing.

Required behavior:
1. First inspect the dataset.
2. Inspect the selected target.
3. Inspect data quality.
4. For classification, use target evidence to decide whether imbalance treatment is needed.
5. Do NOT use SMOTE/over/undersampling for regression or clustering.
6. Run AutoML when supervised/unsupervised modeling is appropriate.
7. Do not choose AutoML training settings. The application uses a fixed deterministic AutoML policy.
8. Do not invent columns, metrics, models, or scores.
9. After tools return results, base your final report ONLY on those results.
10. If a tool fails, explain the failure and continue with the evidence that is available.
11. Do not call run_automl more than once unless the previous call failed.
12. Never change the AutoML policy between runs; reproducibility is required.
13. Keep the final answer structured and practical.

The user-selected target is supplied by the application.
"""

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"Selected target: "
                    f"{self.target if self.target else 'UNSUPERVISED'}\n"
                    f"Goal: {goal}"
                ),
            },
        ]

        trace = []
        final_text = None

        for round_number in range(1, max_rounds + 1):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.1,
                max_completion_tokens=1800,
            )

            message = response.choices[0].message
            messages.append(message)

            tool_calls = getattr(message, "tool_calls", None)

            if not tool_calls:
                final_text = message.content or ""
                break

            for tool_call in tool_calls:
                name = tool_call.function.name

                try:
                    args = json.loads(
                        tool_call.function.arguments or "{}"
                    )
                except Exception:
                    args = {}

                result = self.execute_tool(name, args)
                safe_result = self._json_safe(result)

                trace.append({
                    "round": round_number,
                    "tool": name,
                    "arguments": args,
                    "result": safe_result,
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": name,
                    "content": json.dumps(
                        safe_result,
                        default=str,
                    ),
                })

        # If the model used all rounds without returning text,
        # force a final evidence-based synthesis with tools disabled.
        if not final_text:
            final_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages + [{
                    "role": "user",
                    "content": (
                        "Using only the tool results above, produce the final "
                        "Data Science report now. Do not call more tools."
                    ),
                }],
                tools=tools,
                tool_choice="none",
                temperature=0.1,
                max_completion_tokens=1800,
            )
            final_text = (
                final_response.choices[0].message.content
                or "No final report was returned."
            )

        return {
            "final_report": final_text,
            "trace": trace,
            "automl_result": self.last_automl_result,
        }


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Autonomous AI Data Scientist",
    page_icon="🤖",
    layout="wide",
)

st.markdown(
    """
    <h1 style="text-align:center; color:#1E3A8A;">
        🤖 Autonomous AI Data Scientist & Dashboard
    </h1>
    <p style="text-align:center; color:#4B5563; font-size:18px;">
        Leakage-safe preprocessing • Imbalance handling • Multi-model AutoML
    </p>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Agent Configuration")
    st.caption("Agentic Data Science — Groq Tool Calling + AutoML")

    st.markdown("---")
    st.subheader("🤖 Groq Data Science Agent")

    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Your key is used only for the current Streamlit session."
    )

    st.markdown(
        "[Get a Groq API key](https://console.groq.com/keys)"
    )

    st.caption(
        "Groq acts as the decision layer. Local Python tools inspect the dataset "
        "and execute the existing AutoML engine on your machine."
    )

    st.markdown("---")
    st.caption("Supported files: CSV, XLSX")


# ============================================================
# INITIALIZE DATA SCIENCE AGENT
# ============================================================

if groq_api_key and GROQ_AVAILABLE:
    try:
        ds_agent = DataScienceAgent(api_key=groq_api_key)
    except Exception as e:
        ds_agent = None
        st.sidebar.error(f"Agent initialization failed: {e}")
else:
    ds_agent = None

# ============================================================
# DATA UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📂 Upload your dataset",
    type=["csv", "xlsx"],
    help="Upload a CSV or Excel dataset for analysis and AutoML.",
)

if uploaded_file is None:
    st.info("👆 Upload a CSV/XLSX file to start the autonomous pipeline.")
    st.stop()

# ============================================================
# LOAD DATA
# ============================================================

try:
    if uploaded_file.name.lower().endswith(".csv"):
        raw_df = pd.read_csv(uploaded_file)
    else:
        raw_df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error("❌ Failed to read the uploaded dataset.")
    st.exception(e)
    st.stop()

if raw_df.empty:
    st.warning("⚠️ The uploaded dataset is empty.")
    st.stop()

# Normalize duplicate column names slightly
raw_df = raw_df.copy()

st.success(
    f"✅ Dataset loaded: **{uploaded_file.name}** — "
    f"{raw_df.shape[0]:,} rows × {raw_df.shape[1]:,} columns"
)

# ============================================================
# BASIC DATA AUDIT
# ============================================================

total_nulls = int(raw_df.isna().sum().sum())
duplicate_rows = int(raw_df.duplicated().sum())
numeric_cols_raw = raw_df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols_raw = raw_df.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

try:
    detected_type = detect_data_type(raw_df)
except Exception:
    detected_type = "Unknown"

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric("Rows", f"{raw_df.shape[0]:,}")

with k2:
    st.metric("Columns", raw_df.shape[1])

with k3:
    st.metric("Missing Values", f"{total_nulls:,}")

with k4:
    st.metric("Duplicate Rows", f"{duplicate_rows:,}")

st.markdown("---")

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Data Dashboard",
        "🧹 Data Quality",
        "🧠 AutoML Arena",
        "📋 Raw Preview",
        "🤖 Groq AI Chat",
    ]
)

# ============================================================
# TAB 1 — DASHBOARD
# ============================================================

with tab1:
    st.subheader("📊 Dataset Overview")

    st.write(
        f"**Detected data profile:** `{detected_type}`"
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Numerical Columns")
        if numeric_cols_raw:
            st.write(numeric_cols_raw)
        else:
            st.info("No numerical columns detected.")

    with c2:
        st.markdown("### Categorical Columns")
        if categorical_cols_raw:
            st.write(categorical_cols_raw)
        else:
            st.info("No categorical columns detected.")

    if px is not None:
        st.markdown("---")
        st.subheader("📈 Interactive Visualizations")

        if len(numeric_cols_raw) >= 2:
            x_col = st.selectbox(
                "X axis",
                numeric_cols_raw,
                key="dashboard_x",
            )
            y_options = [c for c in numeric_cols_raw if c != x_col]
            y_col = st.selectbox(
                "Y axis",
                y_options,
                key="dashboard_y",
            )

            try:
                fig = px.scatter(
                    raw_df,
                    x=x_col,
                    y=y_col,
                    template="plotly_white",
                    title=f"{x_col} vs {y_col}",
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not create scatter plot: {e}")

        if numeric_cols_raw:
            hist_col = st.selectbox(
                "Distribution column",
                numeric_cols_raw,
                key="hist_col",
            )

            try:
                fig = px.histogram(
                    raw_df,
                    x=hist_col,
                    marginal="box",
                    template="plotly_white",
                    title=f"Distribution of {hist_col}",
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not create histogram: {e}")
    else:
        st.info("Install Plotly to enable interactive charts.")

# ============================================================
# TAB 2 — DATA QUALITY
# ============================================================

with tab2:
    st.subheader("🧹 Data Quality & Preprocessing Audit")

    st.markdown(
        """
        The preprocessing report below is for **data-quality inspection**.
        The AutoML engine performs its own leakage-safe imputation and encoding
        inside the model pipeline.
        """
    )

    if total_nulls > 0:
        st.warning(
            f"Detected **{total_nulls:,} missing values** in the raw dataset."
        )
    else:
        st.success("No missing values detected in the raw dataset.")

    if duplicate_rows > 0:
        st.warning(f"Detected **{duplicate_rows:,} duplicate rows**.")
    else:
        st.success("No duplicate rows detected.")

    if st.button("🔍 Run Data Quality Preprocessing Report"):
        try:
            _, preproc_report = run_advanced_preprocessing(raw_df)

            st.markdown("### Agent Cleaning Report")
            if preproc_report:
                for item in preproc_report:
                    st.write(f"• {item}")
            else:
                st.info("No preprocessing messages were returned.")
        except Exception as e:
            st.error("Preprocessing report failed.")
            st.exception(e)

    st.markdown("---")
    st.markdown("### Missing Values by Column")

    missing_df = (
        raw_df.isna()
        .sum()
        .sort_values(ascending=False)
        .rename("missing_count")
        .reset_index()
        .rename(columns={"index": "column"})
    )

    missing_df["missing_pct"] = (
        missing_df["missing_count"] / len(raw_df) * 100
    ).round(2)

    st.dataframe(missing_df, use_container_width=True)

# ============================================================
# TAB 3 — AUTOML
# ============================================================

with tab3:
    st.subheader("🧠 Autonomous AutoML Training Arena")

    st.markdown(
        """
        **Important:** AutoML receives the raw dataset. Imputation, encoding,
        scaling, and imbalance treatment are fitted inside the ML pipeline
        to reduce preprocessing leakage.
        """
    )

    all_targets = ["[Unsupervised Clustering]"] + raw_df.columns.tolist()

    target_col = st.selectbox(
        "🎯 Select target column",
        all_targets,
        help="Choose a target for supervised learning, or choose clustering.",
    )

    # --------------------------------------------------------
    # AGENTIC DATA SCIENCE — REAL GROQ LOCAL TOOL CALLING
    # --------------------------------------------------------

    st.markdown("---")
    st.subheader("🤖 Agentic Data Science Workflow")
    st.caption(
        "Groq chooses which local Data Science tools to call. "
        "Python executes them on the real dataset and returns the results to Groq."
    )

    agent_goal = st.text_area(
        "Optional goal for the Agent",
        placeholder=(
            "Example: analyze the dataset, detect problems, train a model, "
            "and explain the result."
        ),
        key="agent_goal",
    )

    if ds_agent is None:
        st.info(
            "🔑 Enter a valid Groq API key in the sidebar to activate "
            "the Data Science Agent."
        )
    else:
        run_full_agent = st.button(
            "🧠 Run Full Data Science Agent",
            type="primary",
            use_container_width=True,
            key="run_full_ds_agent",
        )

        if run_full_agent:
            with st.spinner(
                "🤖 Agent is choosing tools, inspecting the dataset, "
                "and running ML actions..."
            ):
                try:
                    agent_run = ds_agent.run_agent(
                        raw_df,
                        target=(
                            None
                            if target_col == "[Unsupervised Clustering]"
                            else target_col
                        ),
                        user_goal=agent_goal.strip() or None,
                    )

                    st.session_state["agentic_run"] = agent_run

                    # Reuse the real AutoML result in the normal result UI below.
                    if agent_run.get("automl_result") is not None:
                        st.session_state["automl_result"] = (
                            agent_run["automl_result"]
                        )

                    st.success("✅ Data Science Agent completed its workflow.")

                except Exception as e:
                    st.error("❌ Data Science Agent workflow failed.")
                    st.exception(e)

        agentic_run = st.session_state.get("agentic_run")

        if agentic_run:
            st.markdown("### 🧠 Agent Final Report")
            st.markdown(agentic_run.get("final_report", "No report returned."))

            trace = agentic_run.get("trace", [])

            with st.expander(
                f"🛠️ Agent Tool Trace ({len(trace)} calls)",
                expanded=False,
            ):
                if not trace:
                    st.info("The model returned an answer without calling a tool.")
                else:
                    for i, item in enumerate(trace, start=1):
                        st.markdown(
                            f"**{i}. {item.get('tool')}** "
                            f"(round {item.get('round')})"
                        )
                        if item.get("arguments"):
                            st.caption("Arguments")
                            st.json(item["arguments"])
                        st.caption("Tool result")
                        st.json(item.get("result"))

            st.markdown("---")
            st.caption(
                "The report above is grounded in the local tool results. "
                "If AutoML was called, its result is also shown below."
            )

    # --------------------------------------------------------
    # Target analysis
    # --------------------------------------------------------

    if target_col != "[Unsupervised Clustering]":
        target_series = raw_df[target_col]

        target_missing = int(target_series.isna().sum())
        unique_count = int(target_series.nunique(dropna=True))

        a1, a2, a3 = st.columns(3)

        with a1:
            st.metric("Target Unique Values", unique_count)

        with a2:
            st.metric("Target Missing", target_missing)

        try:
            balance_info = analyze_class_balance(target_series)
        except Exception:
            balance_info = {
                "is_classification": False,
                "status": "Unknown",
                "ratio": None,
                "minority_percentage": None,
                "majority_percentage": None,
                "counts": pd.Series(dtype=int),
                "percentages": pd.Series(dtype=float),
            }

        with a3:
            st.metric(
                "Detected Task",
                "Classification"
                if balance_info.get("is_classification")
                else "Regression",
            )

        if balance_info.get("is_classification"):
            st.markdown("### ⚖️ Class Balance Analysis")

            b1, b2, b3 = st.columns(3)

            with b1:
                st.metric(
                    "Balance Status",
                    balance_info.get("status", "Unknown"),
                )

            with b2:
                ratio = balance_info.get("ratio")
                st.metric(
                    "Majority / Minority Ratio",
                    "N/A" if ratio is None else f"{ratio:.2f}",
                )

            with b3:
                minority_pct = balance_info.get("minority_percentage")
                st.metric(
                    "Minority %",
                    "N/A"
                    if minority_pct is None
                    else f"{minority_pct:.2f}%",
                )

            counts = balance_info.get("counts")

            if counts is not None and len(counts) > 0:
                balance_df = pd.DataFrame(
                    {
                        "class": counts.index.astype(str),
                        "count": counts.values,
                    }
                )
                balance_df["percentage"] = (
                    balance_df["count"]
                    / balance_df["count"].sum()
                    * 100
                ).round(2)

                st.dataframe(
                    balance_df,
                    use_container_width=True,
                )

                if px is not None:
                    fig = px.bar(
                        balance_df,
                        x="class",
                        y="count",
                        text="count",
                        template="plotly_white",
                        title="Target Class Distribution",
                    )
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

    else:
        balance_info = {
            "is_classification": False,
            "status": "Clustering",
        }
        st.info(
            "🔵 Unsupervised mode selected. "
            "The engine will run KMeans + silhouette analysis."
        )

    # --------------------------------------------------------
    # Imbalance treatment
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("### ⚖️ Imbalance Treatment")

    imbalance_options = [
        "No Treatment",
        "Class Weighting",
        "SMOTE",
        "Random Oversampling",
        "Random Undersampling",
    ]

    if target_col == "[Unsupervised Clustering]":
        imbalance_method = "No Treatment"
        st.info("Imbalance treatment is not used for clustering.")
    elif balance_info.get("is_classification"):
        recommended_imbalance = st.session_state.get("agent_imbalance", "No Treatment")
        imbalance_index = imbalance_options.index(recommended_imbalance) if recommended_imbalance in imbalance_options else 0
        imbalance_method = st.selectbox(
            "Choose treatment",
            imbalance_options,
            index=imbalance_index,
            help=(
                "Treatment is applied inside the training pipeline. "
                "For SMOTE/oversampling/undersampling, the operation is "
                "performed within the training/CV pipeline."
            ),
        )

        if balance_info.get("status") == "Balanced":
            st.caption(
                "The target appears relatively balanced. "
                "You can still select a treatment manually."
            )
    else:
        imbalance_method = "No Treatment"
        st.info(
            "Imbalance treatment is only applicable to classification."
        )

    # --------------------------------------------------------
    # Training mode
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("### 🚀 Training Mode")

    training_modes = ["Fast", "Balanced", "Deep"]
    recommended_mode = st.session_state.get("agent_training_mode", "Fast")
    training_index = training_modes.index(recommended_mode) if recommended_mode in training_modes else 0
    training_mode = st.selectbox(
        "AutoML depth",
        training_modes,
        index=training_index,
        help=(
            "Fast is recommended for the first run. "
            "Balanced and Deep perform more hyperparameter searches."
        ),
    )

    mode_info = {
        "Fast": "Small search budget — quickest iteration.",
        "Balanced": "Moderate search budget — broader model tuning.",
        "Deep": "Largest search budget — slower but more extensive tuning.",
    }

    st.caption(mode_info[training_mode])

    # --------------------------------------------------------
    # Run AutoML
    # --------------------------------------------------------

    run_agentic_automl = st.button(
        "🤖 Run AutoML Using Agent Plan",
        type="primary",
        use_container_width=True,
        disabled=(ds_agent is None or st.session_state.get("agent_result") is None),
    )

    run_automl = st.button(
        "🚀 Run Autonomous AutoML",
        type="secondary",
        use_container_width=True,
    )

    if run_automl or run_agentic_automl:
      with st.spinner(
        "🤖 Agent is preprocessing, tuning models, "
        "and evaluating real validation/test metrics..."
    ):
        try:
            pipeline_result = evaluate_autonomous_agent_pipeline(
                raw_df,
                target_col,
                imbalance_method=imbalance_method,
                training_mode=training_mode,
            )

            if not isinstance(pipeline_result, (tuple, list)):
                raise ValueError(
                    "AutoML pipeline did not return a tuple/list."
                )

            if len(pipeline_result) < 6:
                raise ValueError(
                    f"AutoML pipeline returned {len(pipeline_result)} values, "
                    "but at least 6 are required."
                )

            (
                task,
                leaderboard,
                winner,
                score,
                decision_logs,
                tuning_explanations,
            ) = pipeline_result[:6]

            st.session_state["automl_result"] = {
                "task": task,
                "leaderboard": leaderboard,
                "winner": winner,
                "score": score,
                "logs": decision_logs,
                "explanations": tuning_explanations,
                "target": target_col,
                "imbalance_method": imbalance_method,
                "training_mode": training_mode,
            }

            st.success("✅ AutoML completed successfully.")

        except Exception as e:
            st.error("❌ AutoML pipeline failed.")
            st.exception(e)
        
    # --------------------------------------------------------
    # Display saved result
    # --------------------------------------------------------

    result = st.session_state.get("automl_result")

    if result is not None:
        st.markdown("---")
        st.subheader("🏆 AutoML Results")

        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric("Task", result["task"])

        with r2:
            st.metric("Winner", str(result["winner"]))

        with r3:
            score_value = result["score"]
            if isinstance(score_value, (int, float, np.integer, np.floating)):
                st.metric("Winner Score", f"{float(score_value):.4f}")
            else:
                st.metric("Winner Score", str(score_value))

        st.markdown(
            f"""
            **Target:** `{result["target"]}`  
            **Imbalance Treatment:** `{result["imbalance_method"]}`  
            **Training Mode:** `{result["training_mode"]}`
            """
        )

        st.markdown("### 📈 Real Model Leaderboard")

        leaderboard = result["leaderboard"]

        if leaderboard is not None and not leaderboard.empty:
            st.dataframe(
                leaderboard,
                use_container_width=True,
                hide_index=True,
            )

            # Automatically choose a sensible metric for charting.
            numeric_score_cols = [
                c for c in leaderboard.columns
                if pd.api.types.is_numeric_dtype(leaderboard[c])
                and (
                    "Test" in c
                    or "CV" in c
                    or "Score" in c
                    or "Balanced" in c
                )
            ]

            if px is not None and numeric_score_cols:
                chart_metric = st.selectbox(
                    "Leaderboard metric",
                    numeric_score_cols,
                    key="leaderboard_metric",
                )

                model_col_candidates = [
                    c for c in leaderboard.columns
                    if c.lower() in {"model", "model_name", "estimator"}
                ]

                model_col = (
                    model_col_candidates[0]
                    if model_col_candidates
                    else leaderboard.columns[0]
                )

                try:
                    chart_df = leaderboard.copy()
                    chart_df = chart_df.sort_values(
                        chart_metric,
                        ascending=False,
                    )

                    fig = px.bar(
                        chart_df,
                        x=model_col,
                        y=chart_metric,
                        text=chart_metric,
                        template="plotly_white",
                        title=f"Model Comparison — {chart_metric}",
                    )
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )
                except Exception as e:
                    st.warning(f"Could not create leaderboard chart: {e}")
        else:
            st.warning("No leaderboard results were returned.")

        st.markdown("---")
        st.subheader("🪵 Agent Decision Logs")

        for log in result["logs"]:
            st.write(f"• {log}")

        st.markdown("---")
        st.subheader("🔧 Hyperparameter Tuning Explanations")

        explanations = result["explanations"]

        if explanations:
            for model_name, explanation in explanations.items():
                with st.expander(f"⚙️ {model_name}"):
                    st.write(explanation)
        else:
            st.info("No tuning explanations were returned.")

# ============================================================
# TAB 4 — RAW PREVIEW
# ============================================================

with tab4:
    st.subheader("📋 Raw Dataset Preview")

    st.dataframe(
        raw_df.head(100),
        use_container_width=True,
        height=500,
    )

    st.markdown("---")

    st.subheader("📐 Dataset Shape")

    shape_df = pd.DataFrame(
        {
            "Metric": [
                "Rows",
                "Columns",
                "Missing Values",
                "Duplicate Rows",
            ],
            "Value": [
                raw_df.shape[0],
                raw_df.shape[1],
                total_nulls,
                duplicate_rows,
            ],
        }
    )

    st.dataframe(
        shape_df,
        use_container_width=True,
        hide_index=True,
    )

# ============================================================
# TAB 5 — GROQ AI CHAT
# ============================================================

with tab5:
    st.subheader("🤖 Groq AI Data Scientist Chat")

    if not groq_api_key:
        st.info(
            "🔑 Enter your Groq API key in the sidebar to activate the chat."
        )
    elif not GROQ_AVAILABLE:
        st.error("❌ The Groq Python package is not installed.")
        st.code(GROQ_IMPORT_ERROR, language="text")
        st.info("Run this in the project venv: python -m pip install groq")
    else:
        try:
            groq_client = Groq(api_key=groq_api_key)

            st.caption(
                "Powered directly by Groq using the current GPT-OSS 120B model. "
                "The assistant receives a compact summary of the uploaded dataset "
                "and the latest AutoML result."
            )

            # Build compact context so the whole dataframe is not sent to the API.
            missing_by_column = (
                raw_df.isna().sum()
                .loc[lambda s: s > 0]
                .to_dict()
            )

            try:
                sample_text = raw_df.head(5).to_string(index=False)
            except Exception:
                sample_text = "Sample unavailable."

            automl_context = "No AutoML run has been completed yet."
            saved_result = st.session_state.get("automl_result")

            if saved_result is not None:
                leaderboard = saved_result.get("leaderboard")
                leaderboard_text = "No leaderboard available."

                if leaderboard is not None and not leaderboard.empty:
                    leaderboard_text = leaderboard.head(10).to_string(index=False)

                automl_context = f"""
Latest AutoML result:
Task: {saved_result.get('task')}
Target: {saved_result.get('target')}
Winner: {saved_result.get('winner')}
Winner score: {saved_result.get('score')}
Imbalance treatment: {saved_result.get('imbalance_method')}
Training mode: {saved_result.get('training_mode')}
Leaderboard:
{leaderboard_text}
"""

            dataset_context = f"""
Dataset: {uploaded_file.name}
Rows: {raw_df.shape[0]}
Columns: {raw_df.shape[1]}
Detected profile: {detected_type}
Numeric columns: {numeric_cols_raw}
Categorical columns: {categorical_cols_raw}
Missing values: {total_nulls}
Missing by column: {missing_by_column if missing_by_column else 'None'}
Duplicate rows: {duplicate_rows}

First 5 rows:
{sample_text}

{automl_context}
"""

            system_prompt = """
You are the AI Data Scientist assistant inside an Automated ML application.
You are powered by Groq.
Use the supplied dataset context and AutoML context as the source of truth.
Do not invent columns, metrics, models, or results.
Explain data quality, class imbalance, preprocessing, pipelines, model metrics,
AutoML results, and practical next steps clearly.
If a requested calculation cannot be derived from the supplied context, say so.
Keep responses concise but useful. Use bullets or small tables when helpful.
"""

            if "groq_chat_messages" not in st.session_state:
                st.session_state.groq_chat_messages = []

            clear_chat = st.button(
                "🗑️ Clear Chat",
                key="clear_groq_chat_stage1"
            )

            if clear_chat:
                st.session_state.groq_chat_messages = []
                st.rerun()

            for message in st.session_state.groq_chat_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            prompt = st.chat_input(
                "Ask Groq about your dataset...",
                key="groq_stage1_chat_input"
            )

            if prompt:
                st.session_state.groq_chat_messages.append(
                    {"role": "user", "content": prompt}
                )

                with st.chat_message("user"):
                    st.markdown(prompt)

                messages = [
                    {
                        "role": "system",
                        "content": system_prompt + "\nDATASET CONTEXT:\n" + dataset_context,
                    }
                ]
                messages.extend(st.session_state.groq_chat_messages[-10:])

                with st.chat_message("assistant"):
                    with st.spinner("⚡ Groq is analyzing your request..."):
                        try:
                            completion = groq_client.chat.completions.create(
                                model="openai/gpt-oss-120b",
                                messages=messages,
                                temperature=0.2,
                                max_completion_tokens=1200,
                            )

                            answer = completion.choices[0].message.content
                            answer = answer or "Groq returned an empty response."

                            st.markdown(answer)
                            st.session_state.groq_chat_messages.append(
                                {"role": "assistant", "content": answer}
                            )

                        except Exception as e:
                            st.error("❌ Groq request failed.")
                            st.code(str(e), language="text")

                            if (
                                st.session_state.groq_chat_messages
                                and st.session_state.groq_chat_messages[-1]["role"] == "user"
                            ):
                                st.session_state.groq_chat_messages.pop()

        except Exception as e:
            st.error("❌ Failed to initialize Groq.")
            st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption(
    "Agentic Data Science System | Groq local tool calling • Real AutoML execution • "
    "Leakage-safe preprocessing • Imbalance-aware training"
)
