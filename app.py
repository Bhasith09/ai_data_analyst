import streamlit as st
import pandas as pd
import numpy as np
import faiss
import requests
import os
from dotenv import load_dotenv


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def visual_df(df):
    visual_df=df.copy()
    for i in visual_df.select_dtype(include=['int64', 'float64']):
        visual_df[i] = visual_df[i].fillna(visual_df[i].median())
    return visual_df

def generate_summary(df):
    summary_text= ""
    for i in df.columns:
        summary_text=summary_text + f"Column: {i}\n"
        summary_text=summary_text + f"Type: {df[i].dtype}, Missing: {df[i].isna().sum()}, Unique: {df[i].nunique()}\n"
        if df[i].dtype in ['int64', 'float64']:
            summary_text=summary_text+f"Mean : {df[i].mean():.2f},Median: {df[i].median():.2f}\n"
        summary_text=summary_text+ f" Sample: {df[i].dropna().head().tolist()}\n\n"   
    return summary_text

def get_llama_embedding(text):
    url= "https://api.groq.com/v1/embeddings"
    headers={"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload={"model": "llama-3.1-8b-instant", "input": text}
    response=requests.post(url,json=payload,headers=headers).json()
    return np.array(response["data"][0]["embedding"], dtype=np.float32)

def chunk_text(text,max_chars=1000):
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]


# Build FAISS index
def build_faiss_index(chunks):
    dim = len(chunks[0]['embedding'])
    index = faiss.IndexFlatL2(dim)
    embeddings = np.array([c['embedding'] for c in chunks])
    index.add(embeddings)
    return index