import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("Please set OPENAI_API_KEY in a .env file.")
    st.stop()

client = OpenAI(api_key=api_key)

st.set_page_config(page_title="AI Data Analyst", layout="wide")
st.title("📊 Personal AI Data Analyst")

st.write("Upload a CSV, ask a question in plain English, and get charts + insights.")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        st.error("Could not read the CSV file. Check encoding or format.")
        st.stop()

    st.write("### 🔍 Data preview")
    st.dataframe(df.head())

    question = st.text_input("Ask a question about your data (e.g., 'show total sales by month')")

    analyze_clicked = st.button("Analyze", key="analyze_button")

    if analyze_clicked:
        if not question.strip():
            st.warning("Please type a question first.")
            st.stop()

        with st.spinner("Thinking..."):
            prompt = f"""
You are a senior data analyst. You are given a pandas DataFrame named df.

Write Python code using ONLY:
- pandas as pd
- matplotlib.pyplot as plt

Rules:
- Do NOT import anything.
- Assume df is already defined.
- If you compute a main result (table/series), store it in a variable named result.
- If you create a plot, use plt to draw it.
- Do NOT show or save the plot (no plt.show(), no savefig).
- Do NOT print anything.
- Just write the code.

User question: {question}

Columns in df: {list(df.columns)}
"""

            try:
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You write safe, concise pandas/matplotlib code for data analysis."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                )

                code = completion.choices[0].message.content

            except Exception as e:
                st.error(f"Error from OpenAI: {e}")
                st.stop()

        st.write("### 🧠 Generated code")
        st.code(code, language="python")

        # Safe execution environment
        local_vars = {"df": df, "pd": pd, "plt": plt}

        try:
            exec(code, {"__builtins__": {}}, local_vars)
        except Exception as e:
            st.error(f"Error while running generated code: {e}")
            st.stop()

        # Show chart if created
        fig_buf = io.BytesIO()
        try:
            plt.tight_layout()
            plt.savefig(fig_buf, format="png")
            fig_buf.seek(0)
            st.write("### 📈 Chart")
            st.image(fig_buf)
        except Exception:
            pass
        finally:
            plt.clf()

        # Show result table/series if exists
        if "result" in local_vars:
            st.write("### 📋 Result")
            st.write(local_vars["result"])

else:
    st.info("Upload a CSV file to get started.")
