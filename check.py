import chromadb
from sentence_transformers import SentenceTransformer

model=SentenceTransformer('all-MiniLM-L6-v2')
client =chromadb.Client()

collection=client.create_collection(name="dataset_vectors")

documents= [
    "AI is transforming industries",
    "Machine learning uses data to learn patterns",
    "Python is widely used in data science"
]
embeddings=model.encode(documents)

collection.add(
    documents=documents,
               embeddings=embeddings.tolist(),
               ids=[str(i) for i in range(len(documents))])


from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer=AutoTokenizer.from_pretrained(model_name)
llm_model=AutoModelForCausalLM.from_pretrained(model_name,device_map="auto")
llm=pipeline("text-generation", model=model, tokenizer=tokenizer)
prompt="what are types of AI"

prompt_embedding=model.encode([prompt])

result=collection.query(
    query_embeddings=prompt_embedding.tolist(),
    n_results=2
)

context_docs=result["data"][0]["documents"] 
context_docs   ="\n".join(context_docs)

llm_input=f"Context:\n{context_docs}\n\nQuestion: {prompt}\nAnswer:"

answer=llm(llm_input, max_length=200)[0]['generated_text']
print("LLM answer")
print(answer)