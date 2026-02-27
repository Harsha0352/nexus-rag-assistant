import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import (
    HuggingFaceEndpointEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

# ---------------- ENV ----------------
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ---------------- APP ----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield

app = FastAPI(lifespan=lifespan)

# -------- CORS -------- #
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- SERVE FRONTEND -------- #
@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """Serve the frontend HTML file."""
    try:
        with open("frontend/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return """
        <html>
            <body>
                <h1>Welcome to Voice RAG API</h1>
                <p><a href="/docs">API Documentation</a></p>
            </body>
        </html>
        """

# ---------------- GLOBALS ----------------
vectorstore = None
retriever = None
rag_chain = None

# ---------------- REQUEST MODEL ----------------
class QuestionRequest(BaseModel):
    question: str

# ---------------- STARTUP ----------------
def load_models():
    global llm, embeddings, vectorstore, retriever, rag_chain

    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=HF_TOKEN,
    )

    llm_endpoint = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
        task="text-generation",
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.5,
        max_new_tokens=512,
    )

    llm = ChatHuggingFace(llm=llm_endpoint)

    # Load pre-created FAISS index
    try:
        vectorstore = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        rag_chain = build_rag_chain()
    except Exception as e:
        print(f"Warning: Could not load FAISS index: {e}")
        print("The system will work after you upload a PDF.")

# ---------------- HELPERS ----------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain():
    global retriever, llm
    
    if retriever is None:
        raise ValueError("Retriever not initialized. Please load a FAISS index or upload a PDF.")
    
    prompt = ChatPromptTemplate.from_template(
        """
Answer the question based only on the context below.

Context:
{context}

Question:
{question}
"""
    )

    return (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

# ---------------- PDF UPLOAD ----------------
@app.post("/upload-pdf/")
async def upload_pdf(files: List[UploadFile] = File(...)):
    global vectorstore, retriever, rag_chain

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    all_chunks = []

    for file in files:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")

        temp_path = f"temp_{file.filename}"

        with open(temp_path, "wb") as f:
            f.write(await file.read())

        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)
        all_chunks.extend(chunks)
        os.remove(temp_path)

    if all_chunks:
        vectorstore = FAISS.from_documents(all_chunks, embeddings)
        retriever = vectorstore.as_retriever()
        rag_chain = build_rag_chain()

    return {"message": "PDF processed successfully ✅"}

# -------- PDF UPLOAD (Multiple) --------
@app.post("/upload-pdf-multi/")
async def upload_pdf_multi(files: List[UploadFile] = File(...)):
    """Upload and process multiple PDF files."""
    global vectorstore, retriever, rag_chain

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    all_chunks = []

    for file in files:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")

        temp_path = f"temp_{file.filename}"

        with open(temp_path, "wb") as f:
            f.write(await file.read())

        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)
        all_chunks.extend(chunks)
        os.remove(temp_path)

    if all_chunks:
        vectorstore = FAISS.from_documents(all_chunks, embeddings)
        retriever = vectorstore.as_retriever()
        rag_chain = build_rag_chain()

    return {
        "message": "PDFs processed successfully ✅",
        "files": [f.filename for f in files]
    }

# -------- ASK QUESTION -------- #
@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    """Ask a question about the uploaded PDFs."""
    global rag_chain

    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if rag_chain is None:
        raise HTTPException(status_code=400, detail="No documents loaded. Please upload a PDF or ensure FAISS index is created.")

    try:
        response = rag_chain.invoke({"question": question})
        return {"answer": response}
    except Exception as e:
        print(f"Error in ask_question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")