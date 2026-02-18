#app.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import os
import json
from dotenv import load_dotenv
from groq import Groq

# -----------------------------
# Setup
# -----------------------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("Please set GROQ_API_KEY in a .env file.")
    st.stop()

client = Groq(api_key=api_key)

st.set_page_config(page_title="AI Data Analyst", layout="wide")
st.title("📊 Personal AI Data Analyst")

st.write("Upload a CSV, ask a question in plain English, and get professional insights and charts.")


# -----------------------------
# Helper: build dataset summary for the model
# -----------------------------
def build_dataset_summary(df: pd.DataFrame, max_rows: int = 5) -> str:
    summary = []

    summary.append("Columns and dtypes:")
    summary.append(str(df.dtypes))

    summary.append("\nSample rows:")
    summary.append(df.head(max_rows).to_string())

    # Basic stats for numeric columns
    if not df.select_dtypes(include=[np.number]).empty:
        summary.append("\nNumeric summary (describe):")
        summary.append(df.describe().to_string())

    return "\n\n".join(summary)


# -----------------------------
# Helper: call Groq LLM for insights + chart suggestion
# -----------------------------
def get_ai_analysis(dataset_summary: str, question: str) -> dict:
    """
    Ask the model to act as a senior data analyst and return:
    - insights: natural language explanation
    - chart: JSON with chart_type, x, y, agg, title (optional)
    """
    system_prompt = """
You are a senior data analyst.

You will receive:
1. A dataset summary (columns, dtypes, sample rows, numeric stats)
2. A user question

Your tasks:
- Provide clear, professional insights in natural language.
- If a chart is helpful or requested, propose ONE chart.

IMPORTANT RULES:
- Only use column names EXACTLY as they appear in the dataset summary.
- Only choose from these chart types: "line", "bar", "scatter", "hist".
- If the question is ambiguous, choose the most reasonable chart.
- If no chart is needed, set "needed" to false.

Respond ONLY in this JSON format:

{
  "insights": "<professional explanation>",
  "chart": {
    "needed": true or false,
    "chart_type": "line" | "bar" | "scatter" | "hist" | null,
    "x": "<column name or null>",
    "y": "<column name or null>",
    "agg": "sum" | "mean" | "count" | null,
    "title": "<chart title or null>"
  }
}


If no chart is needed, set "needed" to false and other fields to null.
"""

    user_content = f"""
DATASET SUMMARY:
{dataset_summary}

USER QUESTION:
{question}
"""

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
    )

    raw = completion.choices[0].message.content.strip()

    # Try to parse JSON safely
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback if model adds extra text
        # Try to extract JSON substring
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            try:
                parsed = json.loads(raw[start:end+1])
            except Exception:
                parsed = {
                    "insights": "I could not parse a structured response, but here is a high-level insight based on the data and question.",
                    "chart": {
                        "needed": False,
                        "chart_type": None,
                        "x": None,
                        "y": None,
                        "agg": None,
                        "title": None,
                    },
                }
        else:
            parsed = {
                "insights": "I could not parse a structured response, but here is a high-level insight based on the data and question.",
                "chart": {
                    "needed": False,
                    "chart_type": None,
                    "x": None,
                    "y": None,
                    "agg": None,
                    "title": None,
                },
            }

    # Ensure keys exist
    if "insights" not in parsed:
        parsed["insights"] = "No insights were provided."

    if "chart" not in parsed:
        parsed["chart"] = {
            "needed": False,
            "chart_type": None,
            "x": None,
            "y": None,
            "agg": None,
            "title": None,
        }

    return parsed


# -----------------------------
# Helper: generate chart from AI suggestion
# -----------------------------
def generate_chart(df: pd.DataFrame, chart_info: dict):
    if not chart_info.get("needed"):
        return None

    chart_type = chart_info.get("chart_type")
    x_col = chart_info.get("x")
    y_col = chart_info.get("y")
    agg = chart_info.get("agg")
    title = chart_info.get("title") or "Chart"

    # Validate columns
    if x_col and x_col not in df.columns:
        return None
    if y_col and y_col not in df.columns:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))

    data = df.copy()

    # Apply aggregation if needed
    if agg and x_col and y_col:
        if agg == "sum":
            data = data.groupby(x_col)[y_col].sum().reset_index()
        elif agg == "mean":
            data = data.groupby(x_col)[y_col].mean().reset_index()
        elif agg == "count":
            data = data.groupby(x_col)[y_col].count().reset_index()

    try:
        if chart_type == "line":
            sns.lineplot(data=data, x=x_col, y=y_col, ax=ax)

        elif chart_type == "bar":
            sns.barplot(data=data, x=x_col, y=y_col, ax=ax)

        elif chart_type == "scatter":
            sns.scatterplot(data=data, x=x_col, y=y_col, ax=ax)

        elif chart_type == "hist":
            sns.histplot(data=data, x=y_col, kde=True, ax=ax)

        else:
            plt.close(fig)
            return None

        ax.set_title(title)
        plt.tight_layout()
        return fig

    except Exception:
        plt.close(fig)
        return None



# -----------------------------
# Main UI
# -----------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        st.error("Could not read the CSV file. Check encoding or format.")
        st.stop()

    st.write("### 🔍 Data preview")
    st.dataframe(df.head())

    question = st.text_input("Ask a question about your data (e.g., 'What is the trend over time?', 'Which day has highest value?')")

    analyze_clicked = st.button("Analyze")

    if analyze_clicked:
        if not question.strip():
            st.warning("Please type a question first.")
            st.stop()

        with st.spinner("Analyzing your dataset like a senior data analyst..."):
            dataset_summary = build_dataset_summary(df)
            ai_response = get_ai_analysis(dataset_summary, question)

        # Show insights
        st.write("### 🧠 Insights")
        st.write(ai_response.get("insights", "No insights generated."))

        # Show chart if needed
        chart_info = ai_response.get("chart", {})
        fig = generate_chart(df, chart_info)

        if fig is not None:
            st.write("### 📈 Chart")
            st.pyplot(fig)
        else:
            if chart_info.get("needed"):
                st.info("The AI suggested a chart, but it could not be generated with the given columns.")
else:
    st.info("Upload a CSV file to get started.")
