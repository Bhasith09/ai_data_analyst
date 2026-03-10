import streamlit as st
import pandas as pd
import numpy as np
import faiss
import requests
import os
from dotenv import load_dotenv

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Please set GROQ_API_KEY in your .env file.")
    st.stop()

# -----------------------------
# 1️⃣ Prepare visualization DataFrame
# -----------------------------
def visual_df(df):
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include=['int64', 'float64']):
        df_copy[col] = df_copy[col].fillna(df_copy[col].median())
    return df_copy

# -----------------------------
# 2️⃣ Generate dataset summary (raw)
# -----------------------------
def generate_summary(df):
    summary_text = ""
    for col in df.columns:
        summary_text += f"Column: {col}\n"
        summary_text += f"Type: {df[col].dtype}, Missing: {df[col].isna().sum()}, Unique: {df[col].nunique()}\n"
        if df[col].dtype in ['int64', 'float64']:
            summary_text += f"Mean: {df[col].mean():.2f}, Median: {df[col].median():.2f}, Min: {df[col].min():.2f}, Max: {df[col].max():.2f}\n"
        summary_text += f"Sample: {df[col].dropna().head(5).tolist()}\n\n"
    return summary_text

# -----------------------------
# 3️⃣ Get embeddings via Groq LLaMA API
# -----------------------------
def get_llama_embedding(text):
    url = "https://api.groq.com/v1/embeddings"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {"model": "llama-3.1-8b-instant", "input": text}

    response = requests.post(url, json=payload, headers=headers).json()

    if "data" not in response or len(response["data"]) == 0:
        st.error(f"Error fetching embedding: {response}")
        return np.zeros(1536, dtype=np.float32)  # fallback zero vector

    return np.array(response["data"][0]["embedding"], dtype=np.float32)

# -----------------------------
# 4️⃣ FAISS RAG store
# -----------------------------
def chunk_text(text, max_chars=1000):
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def build_faiss_index(chunks):
    dim = len(chunks[0]['embedding'])
    index = faiss.IndexFlatL2(dim)
    embeddings = np.array([c['embedding'] for c in chunks])
    index.add(embeddings)
    return index

def retrieve_context(query, chunks, index, top_k=3):
    q_emb = get_llama_embedding(query)
    D, I = index.search(np.array([q_emb]), top_k)
    return "\n".join(chunks[i]['text'] for i in I[0])

# -----------------------------
# 5️⃣ Ask question via LLaMA
# -----------------------------
def ask_llama(question, context):
    url = "https://api.groq.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    prompt = f"Dataset context:\n{context}\n\nQuestion: {question}\nAnswer:"
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "You are a helpful AI data analyst."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(url, json=payload, headers=headers).json()
    return response["choices"][0]["message"]["content"]

# -----------------------------
# 6️⃣ Streamlit UI
# -----------------------------
st.title("📊 AI Data Analyst")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
if uploaded_file:
    # Raw dataset (never altered)
    raw_df = pd.read_csv(uploaded_file)
    st.write("### Raw Dataset (Original)")
    st.dataframe(raw_df.head())

    # Visualization dataset (numeric missing values filled)
    viz_df = visual_df(raw_df)

    # Generate summary from raw data
    summary = generate_summary(raw_df)
    st.write("### Dataset Summary")
    st.text(summary)

    # Create RAG index
    chunked_texts = chunk_text(summary)
    chunks = [{"text": t, "embedding": get_llama_embedding(t)} for t in chunked_texts]
    index = build_faiss_index(chunks)
    st.write("✅ RAG index built using FAISS")

    # Q&A
    st.write("### Ask Questions about Your Dataset")
    question = st.text_input("Enter your question here:")
    if question:
        context = retrieve_context(question, chunks, index)
        answer = ask_llama(question, context)
        st.write("💡 Answer:")
        st.write(answer)

    # Visualization
    st.write("### Data Visualization")
    numeric_cols = viz_df.select_dtypes(include=['int64','float64']).columns.tolist()
    if numeric_cols:
        col_to_plot = st.selectbox("Select numeric column:", numeric_cols)
        st.bar_chart(viz_df[col_to_plot])