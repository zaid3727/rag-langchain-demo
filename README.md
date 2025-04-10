# LangChain RAG Demo (Local Hugging Face Version)

This project demonstrates a simple, interview-ready Retrieval-Augmented Generation (RAG) pipeline using:

- SentenceTransformer for text embeddings
- FAISS for fast semantic retrieval
- LangChain for orchestration
- Local Hugging Face model (flan-t5-base) for generation

---

## How It Works

1. Loads your knowledge base from `docs.txt` (one sentence per line)
2. Embeds each sentence using `all-MiniLM-L6-v2`
3. Uses FAISS to retrieve the most relevant sentence to a question
4. Passes the context and question to a local `flan-t5` model
5. Generates a clean, natural answer

---

## Run It

### Install dependencies:

```bash
pip install langchain langchain-community transformers faiss-cpu sentence-transformers
```

### Run the script:

```bash
python rag_langchain.py
```

Example interaction:

```
Ask your question: How did Zaid deploy ML models?
Answer: FastAPI server
```

---

## Customization Ideas

- Replace `docs.txt` with your resume or project notes
- Add top-k retrieval (multiple chunks)
- Upgrade to GPT-3.5 via OpenAI
- Wrap it into a Streamlit or FastAPI app

---

## Author

Built by [@zaid3727](https://github.com/zaid3727)
