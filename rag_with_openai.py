import os
import openai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Set your API key (hardcoded or via env variable)
openai.api_key = ""  

# Load sentence embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load documents
with open("docs.txt", "r") as f:
    docs = [line.strip() for line in f.readlines()]

# Embed documents
doc_embeddings = embed_model.encode(docs)

# Create FAISS index
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(np.array(doc_embeddings))

# Your question
query = "How did Zaid deploy machine learning models?"
query_vec = embed_model.encode([query])
D, I = index.search(np.array(query_vec), k=1)
retrieved = docs[I[0][0]]

# Build prompt
prompt = f"""
You are a helpful assistant. Use the context below to answer the question.

Context:
{retrieved}

Question:
{query}

Answer:
"""

# Call OpenAI ChatCompletion (classic style)
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3
)

# Print answer
print("\nQ:", query)
print("Context:", retrieved)
print("Answer:", response['choices'][0]['message']['content'])
