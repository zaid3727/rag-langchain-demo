from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
import numpy as np

# Embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Generation model
gen_model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
generator = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)

# Load documents
with open("docs.txt", "r") as f:
    docs = [line.strip() for line in f.readlines()]

# Embed documents
doc_embeddings = embed_model.encode(docs)

# Create FAISS index
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(np.array(doc_embeddings))

# Query
query = input("Ask a question: ")
query_vec = embed_model.encode([query])
D, I = index.search(np.array(query_vec), k=1)

retrieved = docs[I[0][0]]
prompt = f"Context: {retrieved}\nQuestion: {query}\nAnswer:"

# Generate answer
inputs = tokenizer(prompt, return_tensors="pt")
outputs = generator.generate(**inputs, max_length=100)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print results
print(f"\nQ: {query}")
print(f"Retrieved: {retrieved}")
print(f"Answer: {answer}")
