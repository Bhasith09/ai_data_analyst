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
    df_copy = df.copy()#copy of the original document to prevent altration of the og one
    for col in df_copy.select_dtypes(include=['int64', 'float64']):#include only float and integers
        df_copy[col] = df_copy[col].fillna(df_copy[col].median())#fill the nan with median
    return df_copy
# -----------------------------
# 2️⃣ Generate dataset summary (raw)
# -----------------------------
def generate_summary(df):#follows three steps
    summary_text = ""
    for col in df.columns:
        summary_text += f"Column: {col}\n"
        summary_text += f"Type: {df[col].dtype}, Missing: {df[col].isna().sum()}, Unique: {df[col].nunique()}\n" #get the datatype, null value, and the unique value
        if df[col].dtype in ['int64', 'float64']:
            summary_text += f"Mean: {df[col].mean():.2f}, Median: {df[col].median():.2f}, Min: {df[col].min():.2f}, Max: {df[col].max():.2f}\n"
        summary_text += f"Sample: {df[col].dropna().head(5).tolist()}\n\n"# displays the first 5 rows
    return summary_text


#generate the document of the dataset to store to chromadb

from langchain.docstore.document import Document
def create_document_from_df(df):
    document=[]#document object created to store the documents created from the dataset
    for _, row in df.iterrows():#iterates or visits every row without visiting the column name and index
        metadata={col:row[0] for col in df.columns } #meta data is created for refernce to llm model if it need to get extra information and if it needs to filter
        content=" | ".join([str(val) for val in row.values])#main thing which the llm looks for to generate the answer is the content which is created by joining all the values of the row with a separator " | "
        doc=Document(page_content=content, metadata=metadata)#so now to store values in doc and thepredefined values like page_content and metadata
        document.append(doc)#finally append the doc into the documnet list made 
    return document #print the document



# now embedding the documents created from dataset to store to chromaDB
from langchain.vectorstores import Chroma  #to store to chroma
from langchain_community.embeddings import HuggingFaceEmbeddings#to embedding the vectors

hg_embeddings=HuggingFaceEmbeddings()#loaded the embedding model
persist_directory="/content/chroma_db"#set the location to store the db
def create_embedding(document):
    embedding_from_document=Chroma.from_document(
    document=document,#provide document name created eqarlier in document attribute
    collection_name="df",#give the og_dataset name
    embedding=hg_embeddings,#give the embedding model called
    persist_directory=persist_directory#say the location made to store
    )
print("Chroma DB created and embeddings generated successfully.")


from torch import cuda, bfloat16, float16
import transformers
from transformers import AutoTokenizer
from langchain.llms import HuggingFacePipeline
from sentence_transformers import SentenceTransformer
from time import time


model_id=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  #model name
device=f" cuda:{cuda.current_device()}" if cuda.is_available() else "cpu" #check if gpu is available and set the device accordingly

bnb_config=transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
model=transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

