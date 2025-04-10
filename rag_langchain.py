from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


# Load HuggingFace sentence transformer for embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load and prepare documents
with open("docs.txt", "r") as f:
    texts = [line.strip() for line in f.readlines()]

# Build FAISS vectorstore from docs
docsearch = FAISS.from_texts(texts, embedding_model)

# Load HuggingFace T5 model for generation
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=100)
llm = HuggingFacePipeline(pipeline=pipe)

# Create LangChain RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever()
)

# Ask a question
query = input("Ask your question: ")
result = qa_chain.invoke({"query": query})

print("\nAnswer:", result["result"])

