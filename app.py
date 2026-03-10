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