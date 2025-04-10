from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load documents
with open("docs.txt", "r") as f:
    docs = [line.strip() for line in f.readlines()]

# Create embeddings
doc_embeddings = model.encode(docs)

# Build FAISS index
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(np.array(doc_embeddings))

# Map index to text
doc_map = {i: doc for i, doc in enumerate(docs)}

# Query
query = "How did Zaid deploy his machine learning model?"
query_vec = model.encode([query])
D, I = index.search(np.array(query_vec), k=1)

print("\nBest match:")
print(doc_map[I[0][0]])
