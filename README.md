# LangChain RAG Demo (Local Hugging Face Version)

This project demonstrates a simple, interview-ready Retrieval-Augmented Generation (RAG) pipeline using:

- SentenceTransformer for text embeddings
- FAISS for fast semantic retrieval
- LangChain for orchestration
- Local Hugging Face model (`flan-t5-base`) for generation

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
pip install -r requirements.txt
```

---

### Option 1: Basic RAG with local flan-t5

```bash
python rag_basic.py
```

- Uses SentenceTransformer + FAISS
- Generates answers with Hugging Face's `flan-t5-base`

---

### Option 2: LangChain-powered RAG

```bash
python rag_langchain.py
```

- Uses LangChain’s `RetrievalQA` abstraction
- Combines FAISS + local flan-t5 generation
- Clean, modular orchestration

---

### Option 3: RAG with OpenAI (GPT-3.5)

```bash
export OPENAI_API_KEY="your-openai-key"
python rag_with_openai.py
```

- Retrieval remains local (SentenceTransformer + FAISS)
- Uses OpenAI API for generation

---

## Customization Ideas

- Replace `docs.txt` with your resume or project notes
- Add top-k retrieval (multiple chunks)
- Upgrade to GPT-4 or GPT-3.5 via OpenAI
- Wrap it into a Streamlit or FastAPI app

---

## Author

Built by [@zaid3727](https://github.com/zaid3727)
